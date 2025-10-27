# Tech Challenge 2: Classificação Direcional do Ibovespa (t+1)

Este projeto foi desenvolvido como resposta ao Tech Challenge 2, com o objetivo de criar um modelo de classificação para prever a direção do índice Ibovespa no próximo dia útil (t+1).

#### Integrantes do grupo: 
- Cinthia Gonçalez da Silva
- Gabriel Huzian
- Karyne Barbosa Silva
---

## Objetivo do Projeto

O desafio consiste em prever se o preço de fechamento do Ibovespa no **dia útil seguinte (t+1)** será **superior (classe 1: "sobe")** ou **igual/inferior (classe 0: "não sobe")** ao preço de fechamento do **dia atual (t)**.

### Regras Principais
1.  **Dados:** Utilizar exclusivamente o arquivo `Ibovespa.csv` fornecido.
2.  **Conjunto de Teste:** O conjunto de teste deve ser, obrigatoriamente, os **últimos 30 pregões** disponíveis na base de dados, sem embaralhamento.
3.  **Métrica de Sucesso:** Atingir uma acurácia mínima de **75%** no conjunto de teste. Em caso de empate, o critério de desempate é o maior **F1-Score**.
---
## Arquivos no Projeto

* `Tech Challenge Classificacao IBOV.ipynb`: O notebook Jupyter contendo todo o processo de análise, engenharia de features, modelagem e avaliação.
* `Ibovespa.csv`: O conjunto de dados históricos do Ibovespa utilizado (formato PT-BR).
* `Relatorio_Tech_Challenge_Ibov.pdf`: Um relatório resumido dos resultados e métricas do modelo vencedor.
* `apresentação.pdf`: Slides de apresentação do projeto e dos modelos testados.
* `POSTECH - Tech Challenge - Fase 2 (1).pdf`: Documento com a descrição original do desafio.
* `vídeo - desafio tech 2`: vídeo explicativo em que cada membro do grupo comenta o trabalho realizado.
* `README.md`: Este arquivo.
---

## O que o Script (`.ipynb`) faz?

O notebook `Tech Challenge Classificacao IBOV.ipynb` está estruturado nas seguintes etapas:

### 1. Leitura e Tratamento Inicial dos Dados
* Importação das bibliotecas necessárias (Pandas, Numpy, Scikit-learn, etc.).
* Leitura do arquivo `Ibovespa.csv`.
* Tratamento específico para o formato PT-BR:
    * Conversão de datas (DD.MM.AAAA).
    * Limpeza de valores numéricos (ponto como separador de milhar, vírgula como decimal).
    * Conversão de colunas de Volume (sufixos 'K', 'M', 'B') para valores numéricos.
    * Conversão da coluna 'Var%' (string com '%') para float.
* Ordenação dos dados pela data (do mais antigo para o mais recente).

### 2. Análise Exploratória de Dados (EDA)
* Verificação de dados faltantes (NaN).
* Análise da consistência dos dados (ex: Máxima > Mínima).
* Visualização da série histórica do preço de fechamento ('Último').
* Criação da variável alvo (`target`):
    * `1` (Sobe) se o fechamento de amanhã for maior que o de hoje.
    * `0` (Não Sobe) se o fechamento de amanhã for igual ou menor que o de hoje.

### 3. Engenharia de Features (Feature Engineering)
Para prover contexto ao modelo, diversas features baseadas em indicadores técnicos foram criadas:
* **Retornos:** Retornos diários (lag de 1, 5, 10 dias).
* **Médias Móveis:** Médias móveis simples (MMS) de 5, 10 e 20 períodos.
* **Afastamentos:** Diferença (percentual ou absoluta) do preço atual em relação às médias móveis.
* **Volatilidade:** Desvio padrão dos retornos (janelas de 5 e 20 dias).
* **RSI (Índice de Força Relativa):** Indicador de momentum.
* **Stochastic %K:** Oscilador que compara o fechamento com o range (máxima/mínima) de N períodos.
* **Variação de Volume:** Variação percentual do volume.
* **Variáveis de Calendário:** Dia da semana (encoded).

### 4. Preparação para Modelagem
* **Limpeza Final:** Remoção de valores NaN gerados pelas janelas de cálculo das features (no início da série).
* **Split Temporal (Treino/Teste):**
    * `X_teste` / `y_teste`: Separados rigorosamente como os **últimos 30 pregões**.
    * `X_treino` / `y_treino`: Todo o restante dos dados.
* **Padronização (StandardScaler):**
    * Utilização de `Pipelines` para aplicar a padronização (Z-Score) nos dados de treino e teste, evitando *data leakage*. Isso é crucial para modelos sensíveis à escala (Logística, SVM, KNN).

### 5. Modelagem, Validação Cruzada e Otimização
* **Validação Cruzada Temporal:** Uso do `TimeSeriesSplit` para validar os hiperparâmetros no conjunto de treino, respeitando a ordem cronológica dos dados.
* **GridSearch:** Otimização de hiperparâmetros (usando `GridSearchCV` com `TimeSeriesSplit`) para os seguintes modelos:
    1.  Regressão Logística
    2.  SVM Linear (LinearSVC)
    3.  SVM com Kernel RBF
    4.  KNN (K-Nearest Neighbors)
    5.  Árvore de Decisão
    6.  Random Forest

### 6. Avaliação e Seleção do Modelo
* Os modelos otimizados são avaliados no conjunto de **teste (30 dias)**.
* As métricas analisadas são: Acurácia, Precisão, Recall, F1-Score e AUC.
* O modelo vencedor é selecionado com base na meta de acurácia (>= 75%) e, secundariamente, no F1-Score.
* (Com base nos relatórios, o modelo SVM RBF ou o KNN (k=11) apresentaram os melhores resultados, atingindo a meta de acurácia).

### 7. Previsão Final
* O script utiliza o modelo vencedor (treinado com todos os dados de treino) para fazer a previsão para o dia seguinte ao último dia da base (22/10/2025), indicando a probabilidade de alta (Sobe) ou baixa/igual (Não Sobe).
---
## Como Rodar o Projeto

### 1. Pré-requisitos
* Python 3.10+
* Jupyter Notebook, Jupyter Lab ou VS Code (com a extensão Python/Jupyter)

### 2. Bibliotecas Python
Você precisará das seguintes bibliotecas. Você pode instalá-las via `pip`:

```pip install pandas numpy matplotlib seaborn scikit-learn jupyter```

## 3) Split temporal (sem embaralhar) e alvo

- **Alvo binário:** `y_t = 1` se `Close_{t+1} > Close_t`, senão `0`.
- **Split:** **treino** = todo o passado **antes** dos 30 pregões finais; **teste** = **30 pregões finais** da base (exigência do projeto).
- **Validação:** `TimeSeriesSplit` (K-fold temporal) no **treino** para pequenos grids.

## 4) Modelos — separados, simples e com justificativa

- **Regressão Logística** (baseline interpretável) — *pode falhar em não linearidades*;
- **SVM Linear** — separação aproximadamente linear;
- **SVM RBF** — capta **não linearidades** (candidato forte);
- **KNN** — padrões locais; sensível a escala/ruído;
- **Árvore de Decisão** — regras interpretáveis; risco de overfitting se profunda;
- **Random Forest** — robusta, boa de primeira; menos interpretável.

## 5) Métricas e visualizações

- **Acurácia, Precisão, Revocação, F1** (no TESTE de 30 pregões);
- **ROC/AUC** (modelos probabilísticos) e **Matriz de confusão**;
- **Baselines**: “sempre sobe” e “sinal do dia” para contextualizar.
