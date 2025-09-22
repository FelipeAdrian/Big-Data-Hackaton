# Pipeline de Previsões — README

Este repositório contém uma pipeline escalável para gerar previsões semanais por **loja × produto** combinando três abordagens:

1. **Zeros** para pares **descontinuados** (sem venda há X semanas);
2. **LightGBM** nos **pares com maior relevância** (top N por volume);
3. **Médias rápidas** para o **restante** (com fallback global).

O fluxo produz **arquivos parquet de checkpoint por *chunk* de lojas** e, ao final, um **parquet único**. Há também utilitários para **limpar CSVs** de checkpoints antigos e **mesclar/renomear/arredundar** as previsões finais.

---

## 📦 Estrutura dos scripts

### 1) `solver.py` — motor da pipeline (gradiente de qualidade)

**O que faz, passo a passo:**

1. **Lê** os três parquets de entrada:

   * `TRANSACTIONS_PARQUET`: transações com `transaction_date`, `quantity`, `discount`, `internal_store_id`, `internal_product_id`.
   * `PDVS_PARQUET`: metadados de lojas (usa `pdv` se existir; senão autodetecta).
   * `PRODUCTS_PARQUET`: metadados de produtos (usa `produto` se existir; senão autodetecta).
2. **Preprocessa** transações (datas, tipos numéricos, coluna `week_start` alinhada à segunda-feira).
3. **Agrega semanalmente** por `(loja, produto, week_start)` com features: `quantity`, `trans_count`, `discount_mean`, `discount_max`.
4. **Pré-calcula** a **média das últimas 4 semanas por produto (global)**: Series para **lookup O(1)**.
5. **Identifica pares extintos** (semana atual − última semana de venda > `WEEKS_THRESHOLD`) ⇒ **previsão 0** em todos os horizontes.
6. **Amostra produtos “representativos”** (por `PRODUCT_SAMPLE_RATE`), para tentar usar **média 4 semanas por loja×produto** (se existir histórico recente, senão cai no global).
7. **(Opcional, se `lightgbm` disponível)**: seleciona **TOP\_LGBM\_PAIRS** (maiores volumes), **constrói painel**, gera **lags/rolagens/agregados**, prepara **targets por horizonte** e **treina 1 modelo por horizonte** com *early stopping*.

   * Em seguida, **prediz** no **último ponto** de cada par top e **sobrescreve** a predição base (a menos que o par seja extinto).
8. **Calcula métricas baseline (WMAPE)** por horizonte (média móvel) e salva em `output_forecast/metrics/metrics_by_horizon.csv`.
9. **Gera previsões por *chunk* de lojas**:

   * Para cada loja no chunk, monta predição **vetorizada** nos 5 horizontes para **todos os produtos**:

     * **Extintos** → 0;
     * **Representativos** → média 4 semanas **loja×produto** se existir, **fallback** global por produto;
     * **Demais** → média 4 semanas **global** por produto;
     * **Top LGBM** (quando aplicável) → **override** com a saída do modelo.
   * Escreve **um parquet por chunk** em `output_forecast/predictions_tmp/chunk_XXXXX.parquet`.
10. **Concatena** todos os chunks parquet em **um arquivo final**:

    * `output_forecast/full_dataset_predictions.parquet`
    * **Obs.:** a versão atual do `solver.py` também grava `full_dataset_predictions.csv`. Se **não quiser CSV final**, remova a chamada ao `COPY ... TO ... (HEADER, DELIMITER ',')` em `finalize_concatenation`.

**Saída principal:**

* `output_forecast/predictions_tmp/chunk_*.parquet` (checkpoints por chunk);
* `output_forecast/full_dataset_predictions.parquet` (todas as lojas × todos os produtos × 5 horizontes);
* `output_forecast/metrics/metrics_by_horizon.csv` (baseline de referência).

**Campos das previsões:**

* `internal_store_id`, `internal_product_id`, `horizon` (1..5), `prediction` (float ≥ 0).

---

### 2) `rm_csvs.py` — limpeza dos CSVs de checkpoint

**O que faz:** apaga **todos os `.csv`** em `output_forecast/predictions_tmp`.

**Quando usar:**

* Se você alterou o `solver.py` para **não gravar CSV por chunk**, ou
* Se restaram CSVs legados e você quer manter apenas os parquets de checkpoint.

> ⚠️ Observação importante: no `solver.py` atual, o critério de “pular chunk já processado” checa se **parquet E csv existem**. Se você **não grava CSVs**, ou se rodar `rm_csvs.py` **antes de retomar**, o solver **não vai pular** chunks já processados (ele vai reprocessar e **sobrescrever** o mesmo parquet).
> Se deseja permitir **resume** apenas com parquet, ajuste no `solver.py` a verificação para olhar **só o parquet**.

---

### 3) `concat.py` — mescla final com arredondamento e renomeação

**O que faz:**

* Lê **todos os parquets** de `output_forecast/predictions_tmp/`;
* **Arredonda** `prediction` para inteiro (`ROUND → BIGINT`);
* **Renomeia** campos para layout final:

  * `internal_store_id → pdv`
  * `internal_product_id → produto`
  * `horizon → semana`
  * `prediction → quantidade` (inteiro)
* Escreve **um único parquet**:
  `output_forecast/predictions_tmp/merged_predictions_int.parquet`

**Quando usar:**

* Quando você precisa de um **arquivo final “bonito”**, com campo de quantidade **inteira** e nomes “pdv/produto/semana/quantidade”.
* Normalmente rodado **depois** do `solver.py` (e, se quiser, **depois** do `rm_csvs.py`).

---

## 🗂️ Estrutura de diretórios esperada

```
output_forecast/
├── predictions_tmp/            # checkpoints por chunk (parquet) e arquivos auxiliares
│   ├── chunk_00000.parquet
│   ├── chunk_00001.parquet
│   └── ...
├── metrics/
│   └── metrics_by_horizon.csv  # baseline WMAPE por horizonte
├── full_dataset_predictions.parquet   # concat final (solver)
└── full_dataset_predictions.csv       # concat final CSV (opcional, solver)
```

> Se você **não quiser CSV algum**, ajuste o `solver.py` para não chamar o `COPY ... TO ... CSV` no final **e** não gravar CSV por chunk.

---

## ⚙️ Parâmetros principais (editar em `solver.py`)

* **Caminhos de entrada**

  * `PDVS_PARQUET`, `TRANSACTIONS_PARQUET`, `PRODUCTS_PARQUET`

* **Qualidade × Desempenho**

  * `TOP_LGBM_PAIRS` (ex.: `50_000`): nº de pares (loja×produto) para LightGBM;
  * `MAX_LGBM_PANEL_ROWS` (ex.: `4_000_000`): teto de linhas do painel de treino;
  * `LGBM_MIN_TRAIN_ROWS`: mínimo de linhas por horizonte para treinar o modelo;
  * `PRODUCT_SAMPLE_RATE` (ex.: `0.20`): % de produtos “representativos” que usam média loja×produto;
  * `WEEKS_THRESHOLD` (ex.: `12`): sem venda acima desse nº de semanas ⇒ **extinto** (predição 0);
  * `STORE_CHUNK_SIZE` (ex.: `200`): nº de lojas por chunk (checkpoint); ajuste conforme RAM/IO.

* **Outros**

  * `HORIZONS = [1,2,3,4,5]`
  * `SEED = 42` (reprodutibilidade)
  * Diretórios: `OUTDIR`, `TMPDIR`, `METRICS_DIR`

---

## 🧪 Pré-requisitos e instalação

Python ≥ 3.9. Instale as dependências (LightGBM é opcional — o script funciona sem ele):

```bash
python -m pip install --upgrade pip
python -m pip install duckdb pandas numpy pyarrow tqdm
# opcional (para o bloco LightGBM):
python -m pip install lightgbm
```

> Se preferir, use também `fastparquet` como engine parquet alternativa.

---

## ▶️ Como rodar

1. **Rodar a pipeline principal (gera os chunks e o parquet final):**

```bash
python solver.py
```

2. **(Opcional) Remover CSVs de checkpoints (se houver):**

```bash
python rm_csvs.py
```

3. **Mesclar e arredondar (saída com nomes finais e quantidade inteira):**

```bash
python concat.py
```

> Se quiser **pular** a concatenação final do `solver.py` para controlar tudo por fora, comente a chamada `finalize_concatenation` no final do `solver.py` e use só o `concat.py`.

---

## 🧠 Lógica de previsão (“gradiente de qualidade”)

* **Extintos** (`weeks_since_sale > WEEKS_THRESHOLD`):
  → `prediction = 0` para todos os horizontes.

* **Top pares (LightGBM)**:

  1. Seleciona TOP por `total_sales` (com *cap* por nº de linhas de painel);
  2. Cria **lags** (`1,2,3,4,8,12,52`), **rolagens** (`qty_4w_sum`, `qty_12w_mean`) e **agregados semanais** (`store_week_total`, `product_global_week`);
  3. Prepara **targets `target_h{h}`** e treina **1 modelo por horizonte** com *early stopping*;
  4. **Override** das previsões base usando a **predição do último ponto** de cada par.

* **Demais pares**:

  * **Produtos representativos**: média 4s **loja×produto** (se houver histórico recente), **fallback** média 4s **global por produto**;
  * **Outros**: média 4s **global por produto**.

Tudo é **vetorizado** por loja, evitando varreduras por produto e permitindo ***lookup* O(1)**.

---

## 🧾 Formatos/campos de saída

### `output_forecast/full_dataset_predictions.parquet` (solver)

* `internal_store_id` (ou `pdv`)
* `internal_product_id` (ou `produto`)
* `horizon` (1..5)
* `prediction` (float)

### `output_forecast/predictions_tmp/merged_predictions_int.parquet` (concat)

* `pdv` (string/id)
* `produto` (string/id)
* `semana` (1..5)
* `quantidade` (inteiro; `ROUND(prediction)`)

---

## 🚦 Checkpoints, retomada e limpeza

* Os **chunks** são salvos em `predictions_tmp/chunk_*.parquet`.
* **Retomar**: por padrão, o `solver.py` só **pula** um chunk se **parquet e csv** existem.

  * Se não usa CSVs de chunk, **ajuste a checagem** para considerar apenas o parquet ou não rode o `rm_csvs.py` **antes** de retomar.
* `rm_csvs.py` limpa CSVs de chunk (útil quando você migrou para só-parquet).
* `concat.py` não depende do parquet final do solver; ele lê **direto** dos parquets de chunk.

---

## 🧰 Dicas de performance

* **IO**: usar disco NVMe ajuda muito na escrita dos parquets de chunk.
* **Memória**:

  * Diminua `STORE_CHUNK_SIZE` se ver *spikes* de RAM;
  * Reduza `TOP_LGBM_PAIRS` / `MAX_LGBM_PANEL_ROWS` se o painel LGBM ficar grande.
* **CPU**: DuckDB paraleliza `COPY (...)`/`read_parquet(...)`.
* **Barra de progresso**: avança **após cada loja**; se ficar “parado no 0%”, geralmente é a **primeira loja/chunk** levando mais tempo (ex.: cache frio). A versão atual evita filtros N×M e deve evoluir continuamente.

---

## 🧯 Solução de problemas

* **“Travou em 0%”**: confirme que está usando a versão com **médias pré-computadas** (lookup), não varrendo `weekly_df` por produto dentro do loop da loja.
* **Erro do DuckDB no final**: verifique espaço em disco para o parquet final; como alternativa, use o `concat.py` (que lê parquets por glob) para gerar o arquivo renomeado e com arredondamento.
* **LightGBM indisponível**: o script segue sem LGBM (apenas médias e zeros). Instale `lightgbm` para habilitar o bloco top-pairs.
* **Duplicação ao retomar**: se você **não** grava CSVs e a checagem pede `parquet AND csv`, o solver **reprocessa**. Ajuste a checagem para `if chunk_parquet.exists():` somente.

---

## 🔒 Reprodutibilidade

* `SEED=42` em amostragens/modelos;
* Pré-processamento determinístico;
* **Loga** contagens de lojas/produtos, nº de pares extintos, produtos representativos e métricas baseline;
* Versões de libs recomendadas:

  * `duckdb>=1.0.0`, `pandas>=2.2`, `numpy>=1.26`, `pyarrow>=15`, `tqdm>=4.66`, `lightgbm>=4.0` (opcional).

---

## ✅ Checklist rápido

* [ ] Ajustei os **caminhos** dos 3 parquets de entrada?
* [ ] Defini `TOP_LGBM_PAIRS`, `MAX_LGBM_PANEL_ROWS`, `PRODUCT_SAMPLE_RATE`, `STORE_CHUNK_SIZE`?
* [ ] Quero **CSV final** do solver? (se não, comente a parte do CSV em `finalize_concatenation`)
* [ ] Vou **retomar** a execução? (então garanta a checagem correta de existência dos chunks)
* [ ] Preciso dos nomes finais/inteiros? (rodar `concat.py`)

---

**Pronto!** Agora é só rodar:

```bash
python solver.py
python rm_csvs.py          # opcional
python concat.py
```

Se quiser, me peça uma versão do `solver.py` já **sem CSVs em nenhum ponto** e com o **resume** baseado apenas em parquet.



