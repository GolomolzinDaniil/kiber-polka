import os

import polars as pl


def change_types(
        path_data: str,
        tr1: float = 25.0,
        tr2: float = 50.0,
        output_name = None      # либо output_file.parquet
    ):
    '''
    Преобразует данные, для более быстрого дальнейшего использования
    (лучше сразу преобразовать, сохранить, чем расписывать логику в коде и дальше использовать)

    категориальные в Utf8 (str),
        - пропусков нет
    числовые в float32,
        процентное содержание null:
        - если (0, tr1) то не трогаем, catboost сам разберется
        - если [tr1, tr2) то заменим на медиану и добавим столбец-флаг, где 1-была замена null, 0-отстутствие null
        - если (tr2, 100] то нафиг выбрасываем
    '''
    df = pl.read_parquet(path_data)

    category = [col for col in df.columns if col.startswith('cat_feature')]
    numeric  = [col for col in df.columns if col not in category and col != 'customer_id']

    numeric_len = len(df)
    less_tr1        = []
    between_tr1_tr2 = []
    more_tr2        = []
    
    # только числовые, потому что в категориальных нет пропусков
    for col in numeric:

        proc_null = (df[col].null_count() / numeric_len) * 100
        median = df[col].median()

        if proc_null < tr1:
            less_tr1.append(
                pl.col(col).cast(pl.Float32)
            )
        elif tr1 <= proc_null < tr2:
            between_tr1_tr2.extend([
                pl.col(col).cast(pl.Float32).fill_null(median).alias(col),
                pl.col(col).is_null().cast(pl.Int8).alias(f'{col}_is_null')
            ])
        else:
            more_tr2.append(col)

    df = df.with_columns(
        pl.col(category).cast(pl.Utf8),
        *less_tr1,
        *between_tr1_tr2
    )
    df = df.drop(more_tr2)

    basename = os.path.basename(path_data)
    output_name = f'{os.path.splitext(basename)[0]}_typed.parquet' if output_name is None else output_name
    df.write_parquet(f'data/{output_name}', compression='zstd')



change_types(r'data/train/train_main_features.parquet')