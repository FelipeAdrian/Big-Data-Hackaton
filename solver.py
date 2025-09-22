# ===========================
# GRADIENTE DE QUALIDADE: ZEROS + LGBM TOP PAIRS + MÉDIAS RÁPIDAS
# ===========================
import os
import gc
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- opcional LightGBM ---
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ---------------- PARÂMETROS ----------------
PDVS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
TRANSACTIONS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"
PRODUCTS_PARQUET = r"/home/danieldcs/Save/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"

HORIZONS = [1, 2, 3, 4, 5]
STORE_CHUNK_SIZE = 200
PRODUCT_SAMPLE_RATE = 0.20         # % de produtos que usam média loja×produto
WEEKS_THRESHOLD = 12               # semanas sem venda => extinto
TOP_LGBM_PAIRS = 50000            # nº de pares (store,product) para treinar LGBM
MAX_LGBM_PANEL_ROWS = 4000000    # teto de linhas do painel LGBM (para preservar memória)
LGBM_NUM_BOOST_ROUND = 500
LGBM_EARLY_STOP = 30
LGBM_LEARNING_RATE = 0.05
LGBM_MIN_TRAIN_ROWS = 10000       # mínimo de linhas de treino por horizonte para valer a pena
SEED = 42

OUTDIR = Path("output_forecast")
TMPDIR = OUTDIR / "predictions_tmp"
METRICS_DIR = OUTDIR / "metrics"
OUTDIR.mkdir(exist_ok=True, parents=True)
TMPDIR.mkdir(exist_ok=True, parents=True)
METRICS_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- HELPERS ----------------
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
    if 'transaction_date' not in df.columns:
        raise ValueError("Coluna 'transaction_date' não encontrada.")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

    if store_col not in df.columns:
        raise ValueError(f"Coluna de loja '{store_col}' não encontrada.")
    if prod_col not in df.columns:
        raise ValueError(f"Coluna de produto '{prod_col}' não encontrada.")

    if 'quantity' not in df.columns:
        raise ValueError("Coluna 'quantity' não encontrada.")

    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(np.float32)
    if 'discount' in df.columns:
        df['discount'] = pd.to_numeric(df['discount'], errors='coerce').fillna(0).astype(np.float32)
    else:
        df['discount'] = np.float32(0)

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
    last_sale = weekly_df.groupby([store_col, prod_col])['week_start'].max().reset_index()
    max_week = weekly_df['week_start'].max()
    last_sale['weeks_since_sale'] = ((max_week - last_sale['week_start']).dt.days // 7).astype(int)
    extinct = last_sale[last_sale['weeks_since_sale'] > weeks_threshold]
    return set(extinct[[store_col, prod_col]].itertuples(index=False, name=None))

def select_representative_products(weekly_df: pd.DataFrame,
                                   sample_rate=0.2,
                                   prod_col='internal_product_id') -> list:
    product_stats = weekly_df.groupby(prod_col)['quantity'].sum().reset_index()
    product_stats = product_stats[product_stats['quantity'] > 0]
    if len(product_stats) == 0:
        return []
    sampled = product_stats.sample(frac=sample_rate, random_state=SEED)
    return sampled[prod_col].tolist()

def add_time_idx(df, store_col, prod_col):
    df = df.sort_values([store_col, prod_col, 'week_start'])
    min_week = df['week_start'].min()
    df['time_idx'] = ((df['week_start'] - min_week) / np.timedelta64(1, 'W')).astype(np.int32)
    return df

def add_lag_and_roll_features(df, store_col, prod_col):
    df = df.sort_values([store_col, prod_col, 'time_idx'])
    # Lags
    for lag in [1,2,3,4,8,12,52]:
        df[f'qty_lag_{lag}'] = df.groupby([store_col, prod_col])['quantity'].shift(lag)
    # Rolling simples
    df['qty_4w_sum'] = df.groupby([store_col, prod_col])['quantity'].transform(lambda s: s.rolling(4, min_periods=1).sum())
    df['qty_12w_mean'] = df.groupby([store_col, prod_col])['quantity'].transform(lambda s: s.rolling(12, min_periods=1).mean())
    # Agregados semanais (controle de demanda do store/prod)
    df['store_week_total'] = df.groupby([store_col, 'time_idx'])['quantity'].transform('sum')
    df['product_global_week'] = df.groupby([prod_col, 'time_idx'])['quantity'].transform('sum')

    # NaNs -> 0 e cast
    lag_cols = [c for c in df.columns if c.startswith('qty_lag_')]
    df[lag_cols] = df[lag_cols].fillna(0).astype(np.float32)
    df[['qty_4w_sum','qty_12w_mean','store_week_total','product_global_week']] = \
        df[['qty_4w_sum','qty_12w_mean','store_week_total','product_global_week']].fillna(0).astype(np.float32)
    return df

def prepare_lgb_targets(df, store_col, prod_col, horizons):
    df = df.copy()
    for h in horizons:
        df[f'target_h{h}'] = df.groupby([store_col, prod_col])['quantity'].shift(-h)
    return df

def select_top_pairs_for_lgbm(weekly_df, store_col, prod_col, top_k, max_rows_cap):
    stats = weekly_df.groupby([store_col, prod_col], as_index=False).agg(
        total_sales=('quantity','sum')
    )
    stats = stats.sort_values('total_sales', ascending=False).head(top_k)
    # enforce cap of panel rows
    n_weeks = weekly_df['week_start'].nunique()
    est_rows = len(stats) * n_weeks
    if est_rows > max_rows_cap:
        new_top = max(int(max_rows_cap // max(n_weeks,1)), 1000)
        print(f"[INFO] Painel LGBM estimado {est_rows:,} linhas > cap {max_rows_cap:,}. Ajustando TOP_LGBM_PAIRS -> {new_top}")
        stats = stats.head(new_top)
    return stats[[store_col, prod_col]].drop_duplicates()

def train_lgbm_by_horizon(panel_df, features, horizons, time_col='time_idx'):
    """
    Treina 1 modelo por horizonte em corte temporal.
    Retorna dict horizon->model (ou None se insuficiente).
    """
    models = {}
    max_t = int(panel_df[time_col].max())
    train_cutoff = max_t - max(horizons) - 1  # deixa "respiro" antes do out-of-sample
    print(f"[LGBM] train_cutoff time_idx = {train_cutoff} (max={max_t})")

    for h in horizons:
        target = f'target_h{h}'
        train_df = panel_df[panel_df[time_col] <= train_cutoff].dropna(subset=[target])
        val_df = panel_df[(panel_df[time_col] > train_cutoff) & (panel_df[time_col] <= max_t - h)].dropna(subset=[target])

        print(f"[LGBM] h={h}: train_rows={len(train_df):,}, val_rows={len(val_df):,}")
        if (not HAS_LGBM) or len(train_df) < LGBM_MIN_TRAIN_ROWS or len(val_df) == 0:
            print(f"[LGBM] h={h}: pulado (LightGBM indisponível ou dados insuficientes).")
            models[h] = None
            continue

        lgb_train = lgb.Dataset(train_df[features], train_df[target])
        lgb_val = lgb.Dataset(val_df[features], val_df[target], reference=lgb_train)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': LGBM_LEARNING_RATE,
            'num_leaves': 31,
            'verbose': -1,
            'seed': SEED,
        }
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            callbacks=[lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOP),
                       lgb.log_evaluation(period=50)]
        )
        models[h] = model
        # libera memória
        del lgb_train, lgb_val, train_df, val_df
        gc.collect()
    return models

def infer_lgbm_lastpoint(panel_df, features, horizons, store_col, prod_col, models):
    """
    Gera predição por horizonte para o último ponto conhecido de cada par do painel.
    Retorna dict: overrides_by_store[store_id][product_id] = np.array([h1..h5], dtype=float32)
    """
    # último ponto por par
    last_idx = panel_df.groupby([store_col, prod_col])['time_idx'].idxmax()
    X_last = panel_df.loc[last_idx, [store_col, prod_col] + features].copy()
    overrides_by_store = {}

    for h in horizons:
        model = models.get(h)
        if model is None:
            continue
        yhat = model.predict(X_last[features]).astype(np.float32)
        # clip >=0
        yhat = np.maximum(yhat, 0.0)
        # acumula por linha
        for (s, p, pred) in zip(X_last[store_col].values, X_last[prod_col].values, yhat):
            sdict = overrides_by_store.get(s)
            if sdict is None:
                sdict = {}
                overrides_by_store[s] = sdict
            arr = sdict.get(p)
            if arr is None:
                arr = np.zeros(len(horizons), dtype=np.float32)
                sdict[p] = arr
            idx_h = horizons.index(h)
            arr[idx_h] = pred
    return overrides_by_store

def get_simple_prediction_from_lookup(product_id, product_recent_avg):
    val = float(product_recent_avg.get(product_id, 0.0))
    return 0.0 if not np.isfinite(val) else max(0.0, val)

def generate_store_predictions_one_store(store_id,
                                         all_products,
                                         representative_products_set,
                                         weekly_df,
                                         extinct_pairs,
                                         product_recent_avg,
                                         lgbm_overrides_for_store,
                                         store_col='internal_store_id',
                                         prod_col='internal_product_id'):
    """
    Vetorizado + override de LGBM + zeros de extintos.
    """
    products_arr = np.asarray(all_products)
    extinct_mask = np.array([(store_id, p) in extinct_pairs for p in products_arr])

    # médias recentes
    max_week = weekly_df['week_start'].max()
    recent_cutoff = max_week - pd.Timedelta(weeks=4)
    store_hist_recent = weekly_df[(weekly_df[store_col] == store_id) &
                                  (weekly_df['week_start'] >= recent_cutoff)]
    sp_recent = (store_hist_recent
                 .groupby([prod_col])['quantity']
                 .mean()
                 .astype(float))             # index: product_id

    store_recent_vals = pd.Series(index=products_arr, data=np.nan, dtype=float)
    if len(sp_recent) > 0:
        store_recent_vals.loc[sp_recent.index.intersection(store_recent_vals.index)] = sp_recent.values

    global_recent_vals = product_recent_avg.reindex(products_arr).astype(float).fillna(0.0)

    rep_mask = np.array([p in representative_products_set for p in products_arr])

    rep_vals = np.where(np.isfinite(store_recent_vals.values),
                        np.maximum(0.0, store_recent_vals.values),
                        global_recent_vals.values)
    nonrep_vals = global_recent_vals.values

    base_vals = np.where(rep_mask, rep_vals, nonrep_vals)
    base_vals = np.where(extinct_mask, 0.0, base_vals).astype(np.float32)

    # Replica para horizontes
    N = len(products_arr)
    horizons_arr = np.repeat(np.array(HORIZONS, dtype=np.int16), N)
    preds_arr = np.tile(base_vals, reps=len(HORIZONS))
    store_arr = np.full(N * len(HORIZONS), store_id, dtype=object)
    prod_arr = np.tile(products_arr, reps=len(HORIZONS))

    df = pd.DataFrame({
        store_col: store_arr,
        prod_col: prod_arr,
        'horizon': horizons_arr,
        'prediction': preds_arr
    })

    # Override LGBM (não sobrescreve extinto=0)
    if lgbm_overrides_for_store:
        over_rows = []
        for p, vec in lgbm_overrides_for_store.items():
            if (store_id, p) in extinct_pairs:
                continue  # mantém zero
            for i, h in enumerate(HORIZONS):
                over_rows.append({store_col: store_id, prod_col: p, 'horizon': h, 'prediction': float(vec[i])})
        if over_rows:
            over_df = pd.DataFrame(over_rows)
            df_idx = df.set_index([store_col, prod_col, 'horizon'])
            over_idx = over_df.set_index([store_col, prod_col, 'horizon'])
            df_idx.loc[over_idx.index, 'prediction'] = over_idx['prediction'].values
            df = df_idx.reset_index()

    return df

def process_store_chunks_with_checkpoints(all_stores,
                                          all_products,
                                          representative_products,
                                          weekly_df,
                                          product_recent_avg,
                                          extinct_pairs,
                                          lgbm_overrides_by_store,
                                          store_col='internal_store_id',
                                          prod_col='internal_product_id'):
    chunk_ids = list(range(0, len(all_stores), STORE_CHUNK_SIZE))
    pbar = tqdm(total=len(all_stores), desc="Lojas processadas", unit="loja")
    total_processed = 0

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
                    rep_set,
                    weekly_df,
                    extinct_pairs,
                    product_recent_avg,
                    lgbm_overrides_by_store.get(store_id, {}),
                    store_col=store_col,
                    prod_col=prod_col
                )
                all_predictions.append(dfp)
            except Exception as e:
                # fallback: zeros (não deve acontecer, mas garante robustez)
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
            #chunk_df.to_csv(chunk_csv, index=False)

        if idx % 5 == 0:
            gc.collect()

    pbar.close()
    return total_processed

def finalize_concatenation(output_parquet: Path, output_csv: Path):
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
    df = weekly_df.sort_values([store_col, prod_col, 'week_start']).copy()
    min_week = df['week_start'].min()
    df['time_idx'] = ((df['week_start'] - min_week) / np.timedelta64(1, 'W')).astype(np.int32)

    df['pred_naive'] = (df.groupby([store_col, prod_col])['quantity']
                        .transform(lambda s: s.rolling(4, min_periods=1).mean())
                        .astype(np.float32))
    rows = []
    for h in HORIZONS:
        df[f'target_h{h}'] = df.groupby([store_col, prod_col])['quantity'].shift(-h)
        sub = df[['pred_naive', f'target_h{h}']].dropna()
        if len(sub) == 0:
            rows.append({'horizon': h, 'wmape': None, 'n': 0})
        else:
            rows.append({'horizon': h,
                         'wmape': float(wmape(sub[f'target_h{h}'].values, sub['pred_naive'].values)),
                         'n': int(len(sub))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_DIR / "metrics_by_horizon.csv", index=False)
    return metrics

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("=== GRADIENTE: Zeros + LGBM TopPairs + Médias Rápidas (Checkpoints) ===")

    print("1) Lendo parquets...")
    transactions_df = duckdb.read_parquet(TRANSACTIONS_PARQUET).df()
    stores_df = duckdb.read_parquet(PDVS_PARQUET).df()
    products_df = duckdb.read_parquet(PRODUCTS_PARQUET).df()

    # autodetect de colunas em transactions
    store_col_candidates = ['internal_store_id', 'store_id', 'pdv', 'pdv_id']
    prod_col_candidates = ['internal_product_id', 'product_id', 'produto', 'sku']
    store_col = next((c for c in store_col_candidates if c in transactions_df.columns), None)
    prod_col = next((c for c in prod_col_candidates if c in transactions_df.columns), None)
    if store_col is None or prod_col is None:
        raise ValueError("Não encontrei colunas de loja/produto em transactions_df.")

    # universo de lojas/produtos
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

    print("2) Preprocessando transações...")
    transactions_df = preprocess_transactions(transactions_df, store_col=store_col, prod_col=prod_col)

    print("3) Agregando por semana (weekly)...")
    weekly_df = aggregate_weekly(transactions_df, store_col=store_col, prod_col=prod_col)

    # PRECOMPUTE: média 4 semanas por produto
    print("3.1) Pré-calculando médias recentes por produto (4 semanas)...")
    max_week = weekly_df['week_start'].max()
    recent_cutoff = max_week - pd.Timedelta(weeks=4)
    product_recent_avg = (weekly_df[weekly_df['week_start'] >= recent_cutoff]
                          .groupby(prod_col)['quantity']
                          .mean()
                          .astype(np.float32))

    del transactions_df
    gc.collect()

    print("4) Identificando pares extintos...")
    extinct_pairs = identify_extinct_pairs(weekly_df, WEEKS_THRESHOLD, store_col=store_col, prod_col=prod_col)
    print(f"   Pares extintos: {len(extinct_pairs):,}")

    print(f"5) Selecionando produtos representativos ({PRODUCT_SAMPLE_RATE*100:.1f}%)...")
    representative_products = select_representative_products(weekly_df, PRODUCT_SAMPLE_RATE, prod_col)
    print(f"   Produtos representativos: {len(representative_products):,}")

    # ---------- BLOCO LGBM: só nos TOP_LGBM_PAIRS ----------
    lgbm_overrides_by_store = {}  # store_id -> {product_id: np.array([pred_h1..])}
    if HAS_LGBM and TOP_LGBM_PAIRS > 0:
        print("6) Selecionando top pairs para LGBM...")
        top_pairs = select_top_pairs_for_lgbm(
            weekly_df, store_col, prod_col, TOP_LGBM_PAIRS, MAX_LGBM_PANEL_ROWS
        )
        # Remover extintos (não faz sentido treinar/predizer)
        top_pairs = top_pairs[~top_pairs.apply(lambda r: (r[store_col], r[prod_col]) in extinct_pairs, axis=1)]
        print(f"   Top pairs (após filtrar extintos): {len(top_pairs):,}")

        if len(top_pairs) > 0:
            print("6.1) Construindo painel LGBM...")
            panel = weekly_df.merge(top_pairs, on=[store_col, prod_col], how='inner')
            panel = add_time_idx(panel, store_col, prod_col)
            panel = add_lag_and_roll_features(panel, store_col, prod_col)
            panel = prepare_lgb_targets(panel, store_col, prod_col, HORIZONS)

            # features para LGBM
            lag_cols = [c for c in panel.columns if c.startswith('qty_lag_')]
            features = ['discount_mean','discount_max','trans_count',
                        'qty_4w_sum','qty_12w_mean','store_week_total','product_global_week','time_idx'] + lag_cols
            features = [f for f in features if f in panel.columns]

            print(f"6.2) Treinando LightGBM por horizonte (features={len(features)})...")
            models = train_lgbm_by_horizon(panel, features, HORIZONS, time_col='time_idx')

            print("6.3) Inferindo LGBM no último ponto de cada par...")
            lgbm_overrides_by_store = infer_lgbm_lastpoint(panel, features, HORIZONS,
                                                           store_col, prod_col, models)
            # libera memória do painel
            del panel
            gc.collect()
        else:
            print("   Nenhum par elegível para LGBM (após filtros).")
    else:
        if not HAS_LGBM:
            print("6) LightGBM não disponível — pulando bloco de modelos.")
        else:
            print("6) TOP_LGBM_PAIRS=0 — pulando bloco de modelos.")

    # ---------- Métricas baseline ----------
    print("7) Calculando métricas baseline (WMAPE por horizonte) para referência...")
    metrics_df = compute_backtest_metrics(weekly_df, store_col=store_col, prod_col=prod_col)
    print(metrics_df)

    # ---------- Previsões por chunk de lojas (checkpoints) ----------
    print("8) Gerando previsões com checkpoints por chunk...")
    total_processed = process_store_chunks_with_checkpoints(
        all_stores, all_products, representative_products, weekly_df,
        product_recent_avg, extinct_pairs, lgbm_overrides_by_store,
        store_col=store_col, prod_col=prod_col
    )
    print(f"   Lojas processadas nesta execução: {total_processed:,}")

    # ---------- Concat final ----------
    print("9) Concatenando chunks no arquivo final...")
    final_parquet = OUTDIR / "full_dataset_predictions.parquet"
    final_csv = OUTDIR / "full_dataset_predictions.csv"
    finalize_concatenation(final_parquet, final_csv)

    print("10) Concluído!")
    print(f"   -> {final_parquet}")
    print(f"   -> {final_csv}")
    print(f"   -> {METRICS_DIR/'metrics_by_horizon.csv'}")
