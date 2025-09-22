from __future__ import annotations
from pathlib import Path
import os
import duckdb

# =========================
# Utilitários de Memória
# =========================
def _total_ram_bytes() -> int:
    """Tenta obter a RAM total. Fallback seguro de 8 GiB."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb * 1024
    except Exception:
        pass
    return 8 * (1024 ** 3)  # 8 GiB

def _resolve_memlimit(mem_arg: str | int | float) -> str:
    """
    Normaliza para formato aceito pelo DuckDB:
    - "75%" -> "XXGiB"
    - "4GB", "8GiB", "2048MB" -> retorna como está
    - 4, 8, 16 (int/float) -> assume GiB
    """
    if isinstance(mem_arg, (int, float)):
        gib = max(int(mem_arg), 1)
        return f"{gib}GiB"
    s = str(mem_arg).strip()
    if s.endswith("%"):
        try:
            pct = float(s[:-1]) / 100.0
        except ValueError:
            pct = 0.75
        pct = min(max(pct, 0.05), 0.99)
        total = _total_ram_bytes()
        bytes_ = int(total * pct)
        gib = max(bytes_ // (1024 ** 3), 1)
        return f"{gib}GiB"
    return s

# =========================
# SQL Helpers
# =========================
def _escape_sql_string(path: str) -> str:
    # Escapa aspas simples para literal SQL
    return path.replace("'", "''")

def _filelist_sql(files: list[Path]) -> str:
    # Constrói a lista para read_parquet(['a','b',...])
    quoted = [f"'{_escape_sql_string(p.as_posix())}'" for p in files]
    return "[" + ", ".join(quoted) + "]"

def _build_copy_sql_from_files(files: list[Path], out_path: Path, row_group_size: int, codec: str) -> str:
    """
    COPY com:
      - mapeamento inteiro estável p/ pdv/produto (DENSE_RANK)
      - semana SMALLINT
      - quantidade BIGINT >= 0 (ROUND)
      - compressão GZIP/BROTLI
    """
    files_expr = _filelist_sql(files)
    return f"""
        COPY (
            WITH base AS (
                SELECT
                    internal_store_id,
                    internal_product_id,
                    horizon,
                    prediction
                FROM read_parquet({files_expr})
            ),
            store_map AS (
                SELECT internal_store_id,
                       DENSE_RANK() OVER (ORDER BY internal_store_id) AS pdv_code
                FROM (SELECT DISTINCT internal_store_id FROM base)
            ),
            prod_map AS (
                SELECT internal_product_id,
                       DENSE_RANK() OVER (ORDER BY internal_product_id) AS produto_code
                FROM (SELECT DISTINCT internal_product_id FROM base)
            ),
            final AS (
                SELECT
                    CAST(sm.pdv_code AS BIGINT)              AS pdv,
                    CAST(pm.produto_code AS BIGINT)          AS produto,
                    CAST(b.horizon AS SMALLINT)              AS semana,
                    CAST(GREATEST(0, ROUND(CAST(b.prediction AS DOUBLE))) AS BIGINT) AS quantidade
                FROM base b
                JOIN store_map sm ON b.internal_store_id = sm.internal_store_id
                JOIN prod_map  pm ON b.internal_product_id = pm.internal_product_id
            )
            SELECT * FROM final
        )
        TO '{_escape_sql_string(out_path.as_posix())}'
        (FORMAT PARQUET, COMPRESSION {codec}, ROW_GROUP_SIZE {int(row_group_size)});
    """

# =========================
# Merge incremental c/ limite de tamanho
# =========================
def merge_and_round_with_size_cap(
    diretorio_pasta: str,
    output_path: str | None = None,
    size_cap_mb: int = 23,           # pára quando ultrapassar este tamanho
    row_group_size: int = 32_768,     # menor => menos RAM; pode reduzir p/ 16_384 ou 8_192
    threads: int = 1,                 # 1 thread = menor RAM
    memory_limit: str | int | float = "1GiB",  # ou "2GiB" / "50%" etc.
    prefer_brotli: bool = False,      # GZIP consome menos RAM; BROTLI gera arquivo menor
    sort_files: bool = True,          # ordena os chunks para determinismo
):
    in_dir = Path(diretorio_pasta).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {in_dir}")

    chunks = list(in_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"Nenhum arquivo encontrado com padrão {in_dir/'chunk_*.parquet'}")

    if sort_files:
        chunks = sorted(chunks)

    # destinos
    final_out = Path(output_path).expanduser().resolve() if output_path else (in_dir / "merged_predictions_int.parquet").resolve()
    trial_out = in_dir / "_trial_merged.parquet"  # arquivo temporário para testar tamanho

    # ambiente DuckDB (spill em disco)
    db_path = in_dir / "merge_tmp.duckdb"
    tmp_dir = in_dir / "_duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # limpar saídas antigas
    for p in (final_out, trial_out, db_path):
        try:
            p.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass

    memlimit_str = _resolve_memlimit(memory_limit)
    size_cap_bytes = size_cap_mb * 1024 * 1024

    con = duckdb.connect(database=str(db_path))
    try:
        # PRAGMAs de baixa memória
        con.execute(f"PRAGMA threads={int(threads)};")
        con.execute(f"PRAGMA memory_limit='{memlimit_str}';")
        con.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}';")
        con.execute("PRAGMA preserve_insertion_order=false;")
        con.execute("PRAGMA enable_object_cache=false;")

        # Tenta BROTLI e cai para GZIP se necessário (somente no passo final e nos trials)
        def _try_copy(files: list[Path], out_path: Path) -> None:
            if prefer_brotli:
                try:
                    sql = _build_copy_sql_from_files(files, out_path, row_group_size, codec="BROTLI")
                    con.execute(sql)
                    return
                except duckdb.Error:
                    pass  # fallback para GZIP
            sql = _build_copy_sql_from_files(files, out_path, row_group_size, codec="GZIP")
            con.execute(sql)

        # Busca o maior prefixo de chunks cujo arquivo resultante não ultrapassa o cap
        last_ok_index = 0
        for i in range(1, len(chunks) + 1):
            files_try = chunks[:i]
            # gera trial
            if trial_out.exists():
                trial_out.unlink()
            _try_copy(files_try, trial_out)

            sz = os.path.getsize(trial_out)
            print(f"[INFO] Trial com {i} chunk(s): {sz/1024/1024:.2f} MB")
            if sz > size_cap_bytes:
                print(f"[WARN] Limite de {size_cap_mb} MB excedido ao incluir chunk #{i}.")
                break
            last_ok_index = i

        if last_ok_index == 0:
            # até mesmo o primeiro chunk excedeu — mantém o trial como final e avisa
            print("[WARN] O primeiro chunk sozinho excede o limite; mantendo mesmo assim.")
            if final_out.exists():
                final_out.unlink()
            trial_out.rename(final_out)
        else:
            # refaz a cópia final (limpa, sem arquivo trial), só com os que cabem
            if final_out.exists():
                final_out.unlink()
            files_final = chunks[:last_ok_index]
            _try_copy(files_final, final_out)
            # limpa trial
            try:
                trial_out.unlink(missing_ok=True)
            except Exception:
                pass

    finally:
        con.close()
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"Arquivo final salvo em: {final_out}")
    try:
        print(f"Tamanho final: {os.path.getsize(final_out)/1024/1024:.2f} MB")
    except Exception:
        pass

if __name__ == "__main__":
    # Exemplo de uso:
    dir = '/home/danieldcs/Save/output_forecast/predictions_tmp'
    merge_and_round_with_size_cap(
        diretorio_pasta=dir,
        size_cap_mb=23,          # limite de 23 MB
        row_group_size=32_768,   # pode reduzir p/ 16_384 se precisar
        threads=1,
        memory_limit="1GiB",
        prefer_brotli=False,     # True = arquivo menor (mais CPU/mem)
    )



