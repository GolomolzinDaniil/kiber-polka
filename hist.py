import gc, os
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

import polars as pl
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD  # ✅ Добавлено
import joblib  # ✅ Для сохранения SVD


def elapsed():
    """Вспомогательная функция для замера времени"""
    return time.strftime("%H:%M:%S")


def build_svd_features(seed=42, n_components=100):
    """
    ✅ Построение SVD признаков из extra_features
    Выполняется ОДИН раз перед обучением всех моделей
    """
    print(f"[{elapsed()}] Загрузка данных для SVD...")
    
    # Загрузка основных и дополнительных признаков
    tr_main = pl.read_parquet('data/train/train_main_features.parquet')
    te_main = pl.read_parquet('data/test/test_main_features.parquet')
    tr_extra = pl.read_parquet('data/train/train_extra_features.parquet')
    te_extra = pl.read_parquet('data/test/test_extra_features.parquet')
    
    # Сохраняем customer_id для теста
    test_ids = te_main.select('customer_id').to_numpy().flatten()
    
    # Основные признаки
    X_tr = tr_main.drop('customer_id').to_pandas()
    X_te = te_main.drop('customer_id').to_pandas()
    
    # Дополнительные признаки для SVD
    print(f"[{elapsed()}] Подготовка extra features...")
    X_tr_ex = tr_extra.drop('customer_id').to_pandas().astype(np.float32)
    X_te_ex = te_extra.drop('customer_id').to_pandas().astype(np.float32)
    
    # Замена NaN на 0 для SVD
    print(f"[{elapsed()}] SVD...")
    tr_np = np.nan_to_num(X_tr_ex.values, nan=0.0)
    te_np = np.nan_to_num(X_te_ex.values, nan=0.0)
    
    # SVD разложение
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    tr_svd = svd.fit_transform(tr_np)
    te_svd = svd.transform(te_np)
    
    # Создаем имена колонок для SVD признаков
    svd_cols = [f'svd_{i}' for i in range(n_components)]
    
    # Конкатенация основных признаков + SVD
    print(f"[{elapsed()}] Объединение признаков...")
    X_tr_combined = pd.concat([
        X_tr.reset_index(drop=True),
        pd.DataFrame(tr_svd, columns=svd_cols)
    ], axis=1)
    
    X_te_combined = pd.concat([
        X_te.reset_index(drop=True),
        pd.DataFrame(te_svd, columns=svd_cols)
    ], axis=1)
    
    # Сохранение SVD трансформера для теста
    Path('models/').mkdir(exist_ok=True)
    joblib.dump(svd, 'models/svd_transformer.pkl')
    
    # Сохранение объединённых данных
    print(f"[{elapsed()}] Сохранение данных...")
    X_tr_combined.to_parquet('data/train/train_combined.parquet')
    X_te_combined.to_parquet('data/test/test_combined.parquet')
    
    # Сохранение test_ids
    np.save('data/test/test_ids.npy', test_ids)
    
    del tr_main, te_main, tr_extra, te_extra, X_tr, X_te, X_tr_ex, X_te_ex, tr_np, te_np, tr_svd, te_svd
    gc.collect()
    
    print(f"[{elapsed()}] ✅ SVD признаки созданы: {X_tr_combined.shape}")
    
    return X_tr_combined, X_te_combined, svd_cols


def init_data_start(target_id: str, train, target):
    """Подготовка данных для одного таргета"""
    # Этап 1: загрузка данных
    target_by_id = target[target_id].to_numpy()

    # Этап 2: разделяем на индексы на valid/train
    n_samples = len(train)
    train_idx, valid_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.2,
        random_state=42,
        stratify=target_by_id
    )

    train_pd = train.to_pandas()
    
    X_train = train_pd.iloc[train_idx]  # ✅ pandas .iloc работает с индексами
    y_train = target_by_id[train_idx]
    X_valid = train_pd.iloc[valid_idx]
    y_valid = target_by_id[valid_idx]

    return X_train, y_train, X_valid, y_valid

def objective(trial: optuna.Trial, X_train, y_train, X_valid, y_valid):
    """Функция для оптимизации гиперпараметров"""
    
    # ✅ Объединяем train + valid для использования validation_fraction
    X_combined = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_train, y_valid])
    
    params = {
        'verbose': 0,
        'early_stopping': True,
        'n_iter_no_change': 50,
        'validation_fraction': 0.2,
        'max_iter': 6000,
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.001, 10.0, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'max_bins': trial.suggest_int('max_bins', 32, 255),
        'scoring': 'roc_auc',
    }

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_combined, y_combined)

    # ✅ ПРАВИЛЬНО: Получение лучшей метрики (совместимо со всеми версиями sklearn)
    try:
        # sklearn >= 1.3.0
        if hasattr(model, 'validation_history_') and len(model.validation_history_) > 0:
            best_score = max(score.roc_auc for score in model.validation_history_)
        else:
            # sklearn < 1.3.0 или validation_history_ пуст
            best_score = model.validation_score_
    except AttributeError:
        # Fallback: считаем метрику вручную
        best_score = roc_auc_score(y_combined, model.predict_proba(X_combined)[:, 1])

    # ✅ ДЛЯ ПРУНИНГА (совместимо со всеми версиями)
    try:
        if hasattr(model, 'validation_history_'):
            for epoch, score in enumerate(model.validation_history_, 1):
                trial.report(score.roc_auc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    except AttributeError:
        # Если validation_history_ недоступен, пропускаем прунинг
        pass
    
    return best_score

def get_bst_par(X_train, y_train, X_valid, y_valid):
    """Поиск лучших гиперпараметров"""
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20
        )
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=30,
        gc_after_trial=True,
    )
    
    return study.best_params


def get_bst_features(
        best_params,
        X_train, y_train, X_valid, y_valid,
        min_features: int = 50,
        max_features: int = 170,
        coverage: float = 0.95
    ):
    """Отбор лучших признаков по важности"""
    
    # ✅ Объединяем данные для обучения
    X_combined = pd.concat([X_train, X_valid], axis=0, ignore_index=True)
    y_combined = np.concatenate([y_train, y_valid])
    
    model = HistGradientBoostingClassifier(
        **best_params,
        verbose=False
    )
    model.fit(X_combined, y_combined)

    feature_importance = pd.DataFrame({
        'Feature Id': X_combined.columns,
        'Importances': model.feature_importances_
    })

    feature_importance = feature_importance.sort_values('Importances', ascending=False).reset_index(drop=True)
    feature_importance['cumsum'] = feature_importance['Importances'].cumsum()

    threshold = feature_importance['Importances'].sum() * coverage
    matrix = feature_importance[feature_importance['cumsum'] >= threshold]

    n_features = matrix.index[0] + 1 if len(matrix) > 0 else len(matrix)

    N = max(min_features, min(n_features, max_features))
    best_features = feature_importance['Feature Id'].head(N).tolist()

    print(f'Выбрано {N} признаков')

    return best_features

def oof_one_target(best_features, best_params, X_train, y_train):
    """Генерация OOF предсказаний через кросс-валидацию"""
    X_train = X_train[best_features]
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds_model = np.zeros(len(X_train))

    for train_idx, valid_idx in kf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[valid_idx], y_train[valid_idx]

        model = HistGradientBoostingClassifier(
            **best_params,
            verbose=False
        )
        model.fit(X_tr, y_tr)  # ✅ Нет eval_set!
        preds_model[valid_idx] = model.predict_proba(X_val)[:, 1]

        del model, X_tr, X_val
        gc.collect()

    return preds_model


def save_snapshot(best_params, best_features, target_idx, X_train, y_train, oof_score):
    """Сохранение модели и конфига"""
    
    X_combined = X_train[best_features]
    
    model = HistGradientBoostingClassifier(
        **best_params,
        verbose=False
    )
    model.fit(X_combined, y_train)

    Path('snapshots2').mkdir(exist_ok=True)
    Path('configs2').mkdir(exist_ok=True)

    import joblib
    joblib.dump(model, f'snapshots2/{target_idx}.pkl')

    data = {
        'target': target_idx,
        'best_score': oof_score,
        'best_params': best_params,
        'best_features': best_features,
    }
    
    with open(f'configs2/{target_idx}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def save_oof_column(target_idx: str, preds: np.ndarray, filepath: str = 'meta_data.parquet'):
    """Сохранение OOF предсказаний"""
    if os.path.exists(filepath):
        meta_df = pl.read_parquet(filepath)
        meta_df = meta_df.with_columns(pl.Series(target_idx, preds))
    else:
        meta_df = pl.DataFrame({target_idx: preds})
    
    meta_df.write_parquet(filepath)

    print(f"Cтолбец '{target_idx}' сохранён в {filepath}")


def main():
    """Основная функция обучения"""
    
    # ==========================================
    # ✅ ЭТАП 0: Построение SVD признаков (ОДИН РАЗ)
    # ==========================================
    print("\n" + "="*60)
    print("ЭТАП 0: Построение SVD признаков")
    print("="*60)
    
    # Проверяем, есть ли уже сохранённые данные
    if os.path.exists('data/train/train_combined.parquet'):
        print("[⚡] Загрузка готовых SVD признаков...")
        X_train_combined = pd.read_parquet('data/train/train_combined.parquet')
        X_test_combined = pd.read_parquet('data/test/test_combined.parquet')
        test_ids = np.load('data/test/test_ids.npy')
    else:
        print("[🔨] Построение SVD признаков...")
        X_train_combined, X_test_combined, svd_cols = build_svd_features(seed=42, n_components=50)
        test_ids = np.load('data/test/test_ids.npy')
    
    # Конвертация в Polars для совместимости с остальным кодом
    train = pl.from_pandas(X_train_combined)
    target = pl.read_parquet('data/train/train_target.parquet').drop('customer_id')
    columns_tar = target.columns

    idx_col = [0, 41]  # ✅ Все 41 таргет

    time_all = time.time()
    for i in range(idx_col[0], idx_col[1]):
        target_idx = columns_tar[i]

        print(f"\n{'='*60}")
        print(f"TARGET: {target_idx}")
        print(f"{'='*60}")
        time_target = time.time()

        print('Этап 1: получаем data')
        time_stage = time.time()
        X_train, y_train, X_valid, y_valid = init_data_start(target_idx, train, target)
        print(f'Этап 1 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print('Этап 2: получаем лучшие гиперпараметры таргета')
        time_stage = time.time()
        best_params = get_bst_par(X_train, y_train, X_valid, y_valid)
        print(f'Этап 2 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print('Этап 3: получаем лучшие фичи в датасете для таргета')
        time_stage = time.time()
        best_features = get_bst_features(best_params, X_train, y_train, X_valid, y_valid)
        print(f'Этап 3 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print('Этап 4: получаем столбец предсказаний по target_id')
        time_stage = time.time()
        preds_target = oof_one_target(best_features, best_params, X_train, y_train)
        print(f'Этап 4 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print('Этап 5: сохраняем данные о модели')
        oof_auc = roc_auc_score(y_train, preds_target)
        time_stage = time.time()
        save_snapshot(best_params, best_features, target_idx, X_train, y_train, oof_auc)
        print(f'Этап 5 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print('Этап 6: сохраняем preds_target для каждого таргета')
        time_stage = time.time()
        save_oof_column(target_idx, preds_target, 'meta_data.parquet')
        print(f'Этап 6 выполнен за {(time.time() - time_stage) / 60:.2f} мин.')

        print(f'Время для {target_idx}: {(time.time() - time_target) / 60:.2f} мин.\n')

        gc.collect()

    print(f'\n{"="*60}')
    print(f'Обучение {idx_col[1] - idx_col[0]} моделей: {(time.time() - time_all) / 60:.2f} мин.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()