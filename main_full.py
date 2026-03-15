import gc, os
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

import pyarrow
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy  as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score



def objective(trial: optuna.Trial, train: Pool, valid: Pool):
    params = {
        'eval_metric': 'AUC',
        'verbose': 0, # отчёт каждые N новых деревьев
        'use_best_model': True,
        'early_stopping_rounds': 50, # если метрика не растёт 50 деревьев, пруним
        'task_type': 'GPU',
        'thread_count': 20,  # кол-во используемых потоков(-1 макс)
        'iterations': 6_000, # специально много, потому что ставка не на это
        
        'depth': trial.suggest_int('depth', 4, 7),
        'learning_rate': trial.suggest_float('learning_rate', .01, .3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1., 50., log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_strength': trial.suggest_int('random_strength', 1, 7),
    }

    model = CatBoostClassifier(**params)
    model.fit(train, eval_set=valid)

    best_score = model.get_best_score()['validation']['AUC']

    # ДЛЯ ПРУНИНГА
    scores = model.get_evals_result()['validation']['AUC']
    for epoch, score in enumerate(scores, 1):
        trial.report(score, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # ДЛЯ OPTUNA
    return best_score




def main():

    train_path  = 'data/train_main_features_typed.parquet'
    target_path = 'data/train/train_target.parquet'

    train  = pl.read_parquet(train_path).drop('customer_id')
    target = pl.read_parquet(target_path).drop('customer_id')

    all_features = train.columns
    all_cat_features = [col for col in all_features if col.startswith('cat_feature')]

    columns_tar = target.columns


    idx_col = [0, 1] # таргеты которые хочешь обработать за сессию

    time_all = time.time()
    for i in range(idx_col[0], idx_col[1]):

        target_idx = columns_tar[i]
        print(f"TARGET: {target_idx}")
        time_target = time.time()

        print('Этап 1: получаем data')
        time_stage = time.time()

        # Этап 1: загрузка данных
        target_by_idx = target[target_idx].to_numpy()

        # Этап 2: разделяем на индексы на valid/train
        n_samples = len(train)
        train_idx, valid_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.2, random_state=42, stratify=target_by_idx
        )

        # Этап 3: разделим все данные на тренировочные и валидационные
        X_train, y_train = train[train_idx], target_by_idx[train_idx]
        X_valid, y_valid = train[valid_idx], target_by_idx[valid_idx]

        # Этап 5: для catboost pool лучше
        train_pool = Pool(X_train, y_train, all_cat_features)
        valid_pool = Pool(X_valid, y_valid, all_cat_features)

        print(f'Этап 1 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')






        print('Этап 2: получаем лучшие гиперпараметры таргета')
        time_stage = time.time()

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10, # кол-во честных попыток (без прунинга)
                n_warmup_steps=20
            )
        )
        study.optimize(
            lambda trial: objective(trial, train_pool, valid_pool),
            n_trials=50,    # ограничение в 30попыток поиска параметров
            gc_after_trial=True,
        )
        best_params = study.best_params

        print(f'Этап 2 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')







        print('Этап 3: получаем лучшие фичи в датасете для таргета')
        time_stage = time.time()

        min_features: int = 50      # мин. кол-во признаков
        max_features: int = 170     # макс. кол-во признаков
        coverage: float   = 0.95      # оставляем признаки, дающие 95% общей важности
        model = CatBoostClassifier(
            **best_params,
            task_type='GPU',
            verbose=False
        )
        model.fit(train_pool, eval_set=valid_pool)

        # Feature Id    Importances
        feature_importance = model.get_feature_importance(prettified=True)

        feature_importance['cumsum'] = feature_importance['Importances'].cumsum()

        threshold = feature_importance['Importances'].sum() * coverage
        matrix = feature_importance[feature_importance['cumsum'] >= threshold]

        n_features = matrix.index[0] + 1 if len(matrix) > 0 else len(matrix)

        N = max(min_features, min(n_features, max_features))

        best_features = [
            col
            for col in all_features
            if col in feature_importance['Feature Id'].head(N).tolist()
        ]
        best_cat_features = [
            col
            for col in best_features
            if col.startswith("cat_feature")
        ]
        print(f'Выбрано {N} признаков')
        print(best_features)
        print(f'Этап 3 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')







        print('Этап 4: получаем столбец предсказаний по target_id')
        time_stage = time.time()

        X_train = X_train.select(best_features)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds_model = np.zeros(len(X_train))

        for train_idx, valid_idx in kf.split(X_train, y_train):

            X_tr, y_tr   = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[valid_idx], y_train[valid_idx]

            train_fold = Pool(data=X_tr,  label=y_tr,  cat_features=best_cat_features)
            valid_fold = Pool(data=X_val, label=y_val, cat_features=best_cat_features)

            model = CatBoostClassifier(
                **best_params,
                task_type='GPU',
                verbose=False
            )
            model.fit(train_fold, eval_set=valid_fold)
            preds_model[valid_idx] = model.predict_proba(valid_fold)[:,1]

            del model, train_fold, valid_fold, X_tr, X_val, y_tr, y_val
            gc.collect()

        print(f'Этап 4 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')






        print('Этап 5: сохраняем данные о модели')
        time_stage = time.time()

        model = CatBoostClassifier(
            **best_params,
            task_type='GPU',
            verbose=False
        )
        train_pool = Pool(data=X_train, label=y_train, cat_features=best_cat_features)
        model.fit(train_pool)

        Path('snapshots').mkdir(exist_ok=True)
        Path('configs').mkdir(exist_ok=True)

        model.save_model(f'snapshots/{target_idx}.cbm')

        with open(f'configs/{target_idx}.json', 'w', encoding='utf-8') as file:
            json.dump(
                {
                    'target':                 target_idx,
                    'predict':                target_idx.replace('target', 'predict'),
                    'best_score':             roc_auc_score(y_train, preds_model),
                    'best_params':            best_params,
                    'best_features':          best_features,
                    'best_cat_category':      best_cat_features,
                },
                file, indent=4
            )
        print(f'Этап 5 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')







        print('Этап 6: сохраняем preds_target для каждого таргета')
        time_stage = time.time()

        filepath: str = 'meta_data.parquet'

        if os.path.exists(filepath):
            meta_df = pl.read_parquet(filepath)
            meta_df = meta_df.with_columns(pl.Series(target_idx, preds_model))
        else:
            meta_df = pl.DataFrame({target_idx: preds_model})
        
        meta_df.write_parquet(filepath)

        print(f"Cтолбец '{target_idx}' сохранён в {filepath}")
        print(f'Этап 6 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print(f'Время для {target_idx}: {(time.time() - time_target) / 60:.2f} мин.\n')

        gc.collect()

    print(f'Обучение {idx_col[1] - idx_col[0]} моделей: {(time.time() - time_all) / 60:.2f} мин.')


if __name__ == '__main__':
    main()
