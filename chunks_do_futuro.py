import os
import gc
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# codigo principal de processamento e modelo. para concatenação e exposição, veja os outros arquivos!!

# PARÂMETROS
PDVS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
TRANSACTIONS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"
PRODUCTS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"

HORIZONS = [1, 2, 3, 4, 5]
STORE_CHUNK_SIZE = 200
PRODUCT_SAMPLE_RATE = 0.20
WEEKS_THRESHOLD = 12         # semanas sem venda extinto

OUTDIR = Path("output_forecast")
TMPDIR = OUTDIR / "predictions_tmp"
METRICS_DIR = OUTDIR / "metrics"
OUTDIR.mkdir(exist_ok=True, parents=True)
TMPDIR.mkdir(exist_ok=True, parents=True)
METRICS_DIR.mkdir(exist_ok=True, parents=True)

# Funções auxiliare
def wmape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true)) + eps
    return np.sum(np.abs(y_true - y_pred)) / denom

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == 'int64':
            if df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df

def preprocess_transactions(transactions_df: pd.DataFrame,
                            store_col='internal_store_id',
                            prod_col='internal_product_id') -> pd.DataFrame:
    df = transactions_df.copy()
    # datas
    if 'transaction_date' not in df.columns:
        raise ValueError("Coluna 'transaction_date' não encontrada em transactions_df.")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

    # ids
    if store_col not in df.columns:
        raise ValueError(f"Coluna de loja '{store_col}' não encontrada.")
    if prod_col not in df.columns:
        raise ValueError(f"Coluna de produto '{prod_col}' não encontrada.")

    # quantidade/discount
    if 'quantity' not in df.columns:
        raise ValueError("Coluna 'quantity' não encontrada em transactions_df.")
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(np.float32)
    if 'discount' in df.columns:
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0).astype(np.float32)
    else:
        df['discount'] = np.float32(0)

    # semana (segunda-feira)
    df['week_start'] = df['transaction_date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
    return optimize_memory(df[[store_col, prod_col, 'week_start', 'quantity', 'discount']])

def aggregate_weekly(trans_df: pd.DataFrame,
                     store_col='internal_store_id',
                     prod_col='internal_product_id') -> pd.DataFrame:
    df = trans_df.copy()
    agg = df.groupby([store_col, prod_col, 'week_start'], as_index=False).agg(
        quantity=('quantity', 'sum'),
        trans_count=('quantity', 'count'),
        discount_mean=('discount', 'mean'),
        discount_max=('discount', 'max')
    )
    agg['discount_mean'] = agg['discount_mean'].fillna(0.0).astype(np.float32)
    agg['discount_max'] = agg['discount_max'].fillna(0.0).astype(np.float32)
    agg['quantity'] = agg['quantity'].astype(np.float32)
    agg['trans_count'] = agg['trans_count'].astype(np.int32)
    return optimize_memory(agg)

def identify_extinct_pairs(weekly_df: pd.DataFrame,
                           weeks_threshold=12,
                           store_col='internal_store_id',
                           prod_col='internal_product_id') -> set[tuple]:
    # weekly_df já está agregada por semana
    last_sale = weekly_df.groupby([store_col, prod_col])['week_start'].max().reset_index()
    max_week = weekly_df['week_start'].max()
    last_sale['weeks_since_sale'] = ((max_week - last_sale['week_start']).dt.days // 7).astype(int)
    extinct = last_sale[last_sale['weeks_since_sale'] > weeks_threshold]
    return set(extinct[[store_col, prod_col]].itertuples(index=False, name=None))

def select_representative_products(weekly_df: pd.DataFrame,
                                   sample_rate=0.2,
                                   prod_col='internal_product_id') -> list:
    # usa série semanal (muito mais leve) para somar vendas por produto
    product_stats = weekly_df.groupby(prod_col)['quantity'].sum().reset_index()
    product_stats = product_stats[product_stats['quantity'] > 0]
    if len(product_stats) == 0:
        return []
    sampled = product_stats.sample(frac=sample_rate, random_state=42)
    return sampled[prod_col].tolist()

def get_simple_prediction(product_id, product_recent_avg):
    val = float(product_recent_avg.get(product_id, 0.0))
    return 0.0 if not np.isfinite(val) else max(0.0, val)


def generate_store_predictions_one_store(store_id,
                                         all_products,
                                         representative_products,
                                         weekly_df,
                                         extinct_pairs,
                                         product_recent_avg,
                                         store_col='internal_store_id',
                                         prod_col='internal_product_id'):

    # Arrays base
    products_arr = np.asarray(all_products)
    rep_set = representative_products if isinstance(representative_products, set) else set(representative_products)

    # flag extinct (bool) para todos os produtos desta loja
    extinct_mask = np.array([(store_id, p) in extinct_pairs for p in products_arr])

    # Séries recentes: loja e global
    max_week = weekly_df['week_start'].max()
    recent_cutoff = max_week - pd.Timedelta(weeks=4)
    store_hist_recent = weekly_df[(weekly_df[store_col] == store_id) &
                                  (weekly_df['week_start'] >= recent_cutoff)]
    sp_recent = (store_hist_recent
                 .groupby([prod_col])['quantity']
                 .mean()
                 .astype(float))  # index: product_id

    # reindex para alinhar nos 7.092 produtos
    store_recent_vals = pd.Series(index=products_arr, data=np.nan, dtype=float)
    if len(sp_recent) > 0:
        store_recent_vals.loc[sp_recent.index.intersection(store_recent_vals.index)] = sp_recent.values

    global_recent_vals = product_recent_avg.reindex(products_arr).astype(float).fillna(0.0)

    # máscara de representativos
    rep_mask = np.array([p in rep_set for p in products_arr])

    # regra de escolha (vetor):
    # - extinct -> 0
    # - representativo -> usa store_recent se existe senão global_recent
    # - não-representativo -> global_recent
    rep_vals = np.where(np.isfinite(store_recent_vals.values),
                        np.maximum(0.0, store_recent_vals.values),
                        global_recent_vals.values)
    nonrep_vals = global_recent_vals.values

    base_vals = np.where(rep_mask, rep_vals, nonrep_vals)
    base_vals = np.where(extinct_mask, 0.0, base_vals).astype(np.float32)

    # expandir para horizontes (repete valores para cada horizonte)
    horizons_arr = np.repeat(np.array(HORIZONS, dtype=np.int16), len(products_arr))
    preds_arr = np.tile(base_vals, reps=len(HORIZONS))

    # replicar ids
    store_arr = np.full(len(products_arr) * len(HORIZONS), store_id, dtype=object)
    prod_arr = np.tile(products_arr, reps=len(HORIZONS))

    return pd.DataFrame({
        store_col: store_arr,
        prod_col: prod_arr,
        'horizon': horizons_arr,
        'prediction': preds_arr
    })

def process_store_chunks_with_checkpoints(all_stores,
                                          all_products,
                                          representative_products,
                                          weekly_df,
                                          product_recent_avg,
                                          store_col='internal_store_id',
                                          prod_col='internal_product_id'):
    chunk_ids = list(range(0, len(all_stores), STORE_CHUNK_SIZE))
    pbar = tqdm(total=len(all_stores), desc="Lojas processadas", unit="loja")
    total_processed = 0

    # Reuse um set para membership O(1)
    rep_set = representative_products if isinstance(representative_products, set) else set(representative_products)

    for idx, start in enumerate(chunk_ids):
        end = min(start + STORE_CHUNK_SIZE, len(all_stores))
        store_chunk = all_stores[start:end]

        chunk_parquet = TMPDIR / f"chunk_{idx:05d}.parquet"
        chunk_csv = TMPDIR / f"chunk_{idx:05d}.csv"

        if chunk_parquet.exists() and chunk_csv.exists():
            pbar.update(len(store_chunk))
            total_processed += len(store_chunk)
            continue

        all_predictions = []
        for store_id in store_chunk:
            try:
                dfp = generate_store_predictions_one_store(
                    store_id,
                    all_products,
                    rep_set,            # passa o set
                    weekly_df,
                    extinct_pairs,
                    product_recent_avg, #novo
                    store_col=store_col,
                    prod_col=prod_col
                )
                all_predictions.append(dfp)
            except Exception:
                # fallback loja: zeros
                fallback = pd.DataFrame({
                    store_col: [store_id] * (len(all_products) * len(HORIZONS)),
                    prod_col: np.tile(all_products, reps=len(HORIZONS)),
                    'horizon': np.repeat(np.array(HORIZONS, dtype=np.int16), len(all_products)),
                    'prediction': 0.0
                })
                all_predictions.append(fallback)
            finally:
                pbar.update(1)
                total_processed += 1

        if all_predictions:
            chunk_df = pd.concat(all_predictions, ignore_index=True)
            chunk_df.to_parquet(chunk_parquet, index=False)
            chunk_df.to_csv(chunk_csv, index=False)

        if idx % 5 == 0:
            gc.collect()

    pbar.close()
    return total_processed


def finalize_concatenation(output_parquet: Path, output_csv: Path):
    # usa duckdb para concat mais eficiente (inclusive se forem muitos arquivos)
    duckdb.query(f"""
        COPY (
            SELECT * FROM read_parquet('{str(TMPDIR)}/chunk_*.parquet')
        ) TO '{str(output_parquet)}' (FORMAT PARQUET);
    """)
    duckdb.query(f"""
        COPY (
            SELECT * FROM read_parquet('{str(TMPDIR)}/chunk_*.parquet')
        ) TO '{str(output_csv)}' (HEADER, DELIMITER ',');
    """)

def compute_backtest_metrics(weekly_df: pd.DataFrame,
                             store_col='internal_store_id',
                             prod_col='internal_product_id') -> pd.DataFrame:
    """
    Backtest simples: para cada (semana t), predição = média das 4 últimas semanas até t.
    Avalia contra o alvo deslocado em t+h.
    """
    df = weekly_df.sort_values([store_col, prod_col, 'week_start']).copy()
    # time_idx
    min_week = df['week_start'].min()
    df['time_idx'] = ((df['week_start'] - min_week) / np.timedelta64(1, 'W')).astype(np.int32)

    # pred_naive em t
    df['pred_naive'] = (df.groupby([store_col, prod_col])['quantity']
                        .transform(lambda s: s.rolling(4, min_periods=1).mean())
                        .astype(np.float32))

    # targets t+h
    for h in HORIZONS:
        df[f'target_h{h}'] = df.groupby([store_col, prod_col])['quantity'].shift(-h)

    # métricas por horizonte
    rows = []
    for h in HORIZONS:
        sub = df[['pred_naive', f'target_h{h}']].dropna()
        if len(sub) == 0:
            rows.append({'horizon': h, 'wmape': None, 'n': 0})
        else:
            rows.append({'horizon': h,
                         'wmape': float(wmape(sub[f'target_h{h}'].values,
                                              sub['pred_naive'].values)),
                         'n': int(len(sub))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_DIR / "metrics_by_horizon.csv", index=False)
    return metrics

# main

if __name__ == "__main__":

    # leitura de dados
    
    transactions_df = duckdb.read_parquet(TRANSACTIONS_PARQUET).df()
    stores_df = duckdb.read_parquet(PDVS_PARQUET).df()
    products_df = duckdb.read_parquet(PRODUCTS_PARQUET).df()

    # autodetect de colunas auxiliares
    
    store_col_candidates = ['internal_store_id', 'store_id', 'pdv', 'pdv_id']
    prod_col_candidates = ['internal_product_id', 'product_id', 'produto', 'sku']
    store_col = next((c for c in store_col_candidates if c in transactions_df.columns), None)
    prod_col = next((c for c in prod_col_candidates if c in transactions_df.columns), None)
    if store_col is None or prod_col is None:
        raise ValueError("Não encontrei colunas de loja/produto em transactions_df.")

    # listas de lojas/produtos
    
    if 'pdv' in stores_df.columns:
        all_stores = sorted(stores_df['pdv'].dropna().unique().tolist())
    elif store_col in stores_df.columns:
        all_stores = sorted(stores_df[store_col].dropna().unique().tolist())
    else:
        all_stores = sorted(transactions_df[store_col].dropna().unique().tolist())

    if 'produto' in products_df.columns:
        all_products = sorted(products_df['produto'].dropna().unique().tolist())
    elif prod_col in products_df.columns:
        all_products = sorted(products_df[prod_col].dropna().unique().tolist())
    else:
        all_products = sorted(transactions_df[prod_col].dropna().unique().tolist())

    print(f"   Lojas únicas: {len(all_stores):,}")
    print(f"   Produtos únicos: {len(all_products):,}")
    print(f"   Predições esperadas: {len(all_stores) * len(all_products) * len(HORIZONS):,}")

    # pre-processamento
    
    print("2) Preprocessando transações...")
    transactions_df = preprocess_transactions(transactions_df, store_col=store_col, prod_col=prod_col)

    print("3) Agregando por semana (weekly)...")
    
    weekly_df = aggregate_weekly(transactions_df, store_col=store_col, prod_col=prod_col)

        # --- PRECOMPUTE: média 4 semanas por produto (lookup O(1)) ---
    print("3.1) Pré-calculando médias recentes por produto (4 semanas)...")
    max_week = weekly_df['week_start'].max()
    recent_cutoff = max_week - pd.Timedelta(weeks=4)
    product_recent_avg = (weekly_df[weekly_df['week_start'] >= recent_cutoff]
                          .groupby(prod_col)['quantity']
                          .mean()
                          .astype(np.float32))

    # solta memória do bruto (mantém só o semanal, que já tem tudo que precisamos)
    del transactions_df
    gc.collect()

    # extintos
    print("4) Identificando pares extintos...")
    extinct_pairs = identify_extinct_pairs(weekly_df, WEEKS_THRESHOLD, store_col=store_col, prod_col=prod_col)
    print(f"   Pares extintos: {len(extinct_pairs):,}")

    representative_products = select_representative_products(weekly_df, PRODUCT_SAMPLE_RATE, prod_col)
    print(f"   Produtos representativos: {len(representative_products):,}")

    metrics_df = compute_backtest_metrics(weekly_df, store_col=store_col, prod_col=prod_col)
    print(metrics_df)

    # checkpoints
    
    print("7) Gerando previsões com checkpoints por chunk...")
    total_processed = process_store_chunks_with_checkpoints(
        all_stores, all_products, representative_products, weekly_df, product_recent_avg,
        store_col=store_col, prod_col=prod_col
    )

    print(f"   Lojas processadas nesta execução: {total_processed:,}")

