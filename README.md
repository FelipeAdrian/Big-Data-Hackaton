# Pipeline de Forecasting de Vendas Escalável

Este projeto implementa um pipeline de ponta a ponta para gerar previsões de vendas semanais em larga escala. A solução foi projetada com foco em performance, eficiência de memória e resiliência, permitindo o processamento de milhões de combinações de loja-produto sem esgotar os recursos do sistema e com a capacidade de retomar o processo em caso de falhas.

# Metodologia do Pipeline

O pipeline é executado em uma sequência de passos lógicos.

1. Carregamento e Preparação dos Dados

    Os dados de transações, lojas e produtos são lidos a partir de arquivos Parquet.

    O script autodetecta os nomes das colunas de ID da loja e do produto a partir de uma lista de candidatos (internal_store_id, pdv, etc.), tornando-o mais flexível.

    As listas únicas de todas as lojas e produtos são extraídas para formar o "universo" de previsões a serem geradas.

2. Pré-processamento e Agregação

    Limpeza: Os dados transacionais passam por um pré-processamento que converte datas, trata valores nulos em colunas numéricas (quantity, discount) e otimiza os tipos de dados.

    Agregação Semanal: As transações são agregadas por semana, para cada par loja-produto. São calculadas métricas como quantity (soma), trans_count (contagem), e estatísticas de desconto. Esta agregação transforma os dados brutos em uma série temporal semanal, que é a base para as previsões.


3. Definição de Regras de Negócio

    Pares Extintos: Um par (loja, produto) é considerado "extinto" se não registrar vendas por um número configurável de semanas (WEEKS_THRESHOLD). As previsões para esses pares são zeradas.

    Produtos Representativos: Uma amostra de produtos (PRODUCT_SAMPLE_RATE) é selecionada com base em seu volume total de vendas. Estes produtos são considerados "representativos" e recebem um tratamento diferenciado na lógica de previsão.

4. Lógica de Previsão

A previsão para as próximas semanas (definidas em HORIZONS) é gerada com base na seguinte lógica hierárquica para cada par (loja, produto):

    - Verificação de Extinção: Se o par é "extinto", a previsão é 0.

    - Produto Representativo:

        Se há dados de vendas para o par (loja, produto) nas últimas 4 semanas, a previsão é a média de vendas dessa loja e produto.

        Caso contrário (se não houver vendas recentes nessa loja), a previsão utiliza um fallback: a média de vendas global desse produto em todas as lojas nas últimas 4 semanas.

    - Produto Não Representativo:

        A previsão é sempre a média de vendas global desse produto em todas as lojas nas últimas 4 semanas.
    Essa abordagem garante que produtos importantes (representativos) tenham previsões mais personalizadas para a realidade de cada loja, enquanto produtos de cauda longa usam uma estimativa mais geral.

5. Geração e Concatenação dos Resultados

    O script itera sobre a lista de lojas em lotes (STORE_CHUNK_SIZE).

    Para cada lote, ele gera todas as previsões e salva o resultado em um arquivo Parquet e CSV temporário no diretório predictions_tmp.

    Após processar todas as lojas, o DuckDB é usado para concatenar todos os arquivos temporários de forma eficiente, criando os arquivos de saída finais.

# Como Utilizar

## Configuração

Antes de executar, ajuste os parâmetros na seção --------------- PARÂMETROS --------------- do script chunks_do_futuro.py:

    PDVS_PARQUET: Caminho para o arquivo Parquet com os dados das lojas.

    TRANSACTIONS_PARQUET: Caminho para o arquivo Parquet com os dados de transações.

    PRODUCTS_PARQUET: Caminho para o arquivo Parquet com os dados dos produtos.

    HORIZONS: Uma lista de horizontes de previsão em semanas (ex: [1, 2, 3, 4, 5]).

    STORE_CHUNK_SIZE: Número de lojas a serem processadas por lote. Um valor menor consome menos RAM, mas pode ser mais lento.

    PRODUCT_SAMPLE_RATE: Percentual de produtos a serem considerados "representativos" (ex: 0.20 para 20%).

    WEEKS_THRESHOLD: Número de semanas sem vendas para um par ser considerado "extinto".

    OUTDIR: Diretório principal onde os resultados serão salvos.



## Execução

Para iniciar o pipeline, execute o script a partir do seu terminal:

    - python chunks_do_futuro.py

O script exibirá o progresso em tempo real, informando cada etapa e mostrando uma barra de progresso para o processamento das lojas.

## Saída

Ao final da execução, três arquivos principais serão gerados:

    1. output_forecast/full_dataset_predictions.parquet (e .csv)

        Contém a previsão final para cada combinação de loja, produto e horizonte.

        Colunas:

            internal_store_id: ID da loja.

            internal_product_id: ID do produto.

            horizon: A semana futura da previsão (ex: 1, 2, 3...).

            prediction: A quantidade de vendas previstas.

    2. output_forecast/metrics/metrics_by_horizon.csv

        Contém as métricas de desempenho do modelo de baseline (média móvel).

        Colunas:

            horizon: O horizonte de previsão avaliado.

            wmape: O Erro Percentual Absoluto Ponderado (Weighted Mean Absolute Percentage Error) para aquele horizonte.

            n: O número de observações usadas para calcular a métrica.
