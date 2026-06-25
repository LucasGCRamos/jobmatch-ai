# JobMatch AI

Projeto E2E de Machine Learning para recomendação inteligente de vagas e análise de compatibilidade entre currículos e oportunidades.

## Descrição do Projeto

O **JobMatch AI** é uma aplicação de Machine Learning voltada para analisar a compatibilidade entre currículos e vagas de emprego.

A proposta do projeto é permitir que um usuário informe seu perfil profissional ou currículo e receba como resposta:

* score de aderência entre currículo e vaga;
* classificação automática como **Fit** ou **No Fit**;
* ranking com as **Top-5 vagas** mais compatíveis;
* lista de skills compatíveis;
* lista de skills faltantes;
* sugestão de desenvolvimento profissional;
* estimativa de faixa salarial, caso existam dados suficientes.

O projeto utiliza técnicas de NLP, classificação, similaridade textual, recomendação e, opcionalmente, regressão para estimativa salarial.

## Objetivo

Desenvolver uma solução E2E de Machine Learning capaz de comparar currículos e vagas de emprego, indicando o nível de compatibilidade entre eles e recomendando oportunidades mais alinhadas ao perfil do candidato.

## Motivação

Muitos candidatos se candidatam a vagas sem saber exatamente se possuem aderência aos requisitos solicitados. Ao mesmo tempo, empresas recebem muitos currículos desalinhados com as vagas disponíveis.

O **JobMatch AI** busca reduzir esse problema ao oferecer uma análise mais objetiva da compatibilidade entre candidato e oportunidade.

## Estrutura do Projeto

```text
jobmatch-ai/
│
├── app/
│   └── streamlit_app.py                     # interface web (Fit, ranking e skills)
│
├── data/
│   ├── raw/                                 # dados originais (não versionados)
│   ├── processed/                           # dados tratados (não versionados)
│   ├── sample/                              # amostras pequenas versionadas
│   └── LEIA-ME.txt
│
├── models/                                  # artefatos treinados (não versionados)
│
├── notebooks/
│   ├── 01_eda_resume_jd_match.ipynb
│   ├── 02_eda_linkedin_job_postings.ipynb
│   ├── 03_eda_job_skill_set.ipynb
│   ├── 04_composicao_datasets.ipynb
│   ├── 05_modelo_baseline_fit_no_fit.ipynb
│   ├── 05_recomendacao_e_skills.ipynb
│   └── 06_comparacao_modelos_fit_no_fit.ipynb
│
├── reports/
│   └── proposta_projeto.md
│
├── src/
│   ├── data_loader.py                       # carregamento e download dos datasets
│   ├── preprocessing.py                     # tratamento de texto e skills
│   ├── compose_datasets.py                  # composição da base de vagas
│   ├── features.py                          # engenharia de atributos Fit/No-Fit
│   ├── training.py                          # treino e comparação de modelos
│   ├── model_predict.py                     # inferência Fit/No-Fit
│   ├── recommendation.py                    # ranking Top-5 de vagas
│   └── skills.py                            # análise de skills compatíveis/faltantes
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Por que a pasta `data/` existe se os dados não sobem para o GitHub?

A pasta `data/` serve para organizar os datasets localmente durante o desenvolvimento do projeto.

Os arquivos de dados normalmente são grandes e podem ultrapassar o limite recomendado para repositórios GitHub. Por isso, eles devem ser baixados manualmente ou por comandos e armazenados apenas na máquina de quem estiver rodando o projeto.

A estrutura recomendada é:

```text
data/
├── raw/
│   ├── linkedin-job-postings/
│   ├── resume-jd-match/
│   └── job-skill-set/
│
└── processed/
    ├── vagas_tratadas.csv
    ├── pares_curriculo_vaga.csv
    └── skills_tratadas.csv
```

* `data/raw/`: armazena os dados originais, sem tratamento.
* `data/processed/`: armazena os dados tratados e prontos para modelagem.
* `data/sample/`: amostras pequenas versionadas para testes rápidos.
* `data/LEIA-ME.txt`: explica como baixar e organizar os datasets.

Os arquivos dentro de `data/raw/` e `data/processed/` não devem ser enviados para o GitHub.

## Datasets Utilizados

Este projeto prevê o uso de três bases principais:

### 1. [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

Base com vagas reais do LinkedIn, contendo informações como título da vaga, descrição, empresa, localização, salário, tipo de trabalho e skills associadas.

Uso no projeto:

* criar base de vagas;
* gerar ranking de recomendações;
* extrair descrições das vagas;
* estimar faixa salarial, caso os dados estejam disponíveis.

### 2. [Resume-JD-Match](https://huggingface.co/datasets/facehuggerapoorv/resume-jd-match)

Base com pares de currículos e descrições de vagas, rotulados como **Fit** ou **No Fit**.

Uso no projeto:

* treinar o modelo de classificação;
* avaliar se um currículo combina ou não com uma vaga;
* construir o modelo baseline de NLP.

### 3. [Job Skill Set Dataset](https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set)

Base com cargos, descrições e conjuntos de habilidades associadas.

Uso no projeto:

* identificar skills exigidas por cargo;
* comparar skills do candidato com skills da vaga;
* gerar lista de habilidades compatíveis e faltantes.

## Como baixar os dados

Os datasets não estão incluídos diretamente neste repositório. Para executar o projeto, é necessário baixá-los separadamente.

### Opção 1 — Download manual

Acesse as páginas dos datasets:

* Kaggle — LinkedIn Job Postings 2023-2024
* Hugging Face — Resume-JD-Match
* Kaggle — Job Skill Set Dataset

Depois de baixar os arquivos, organize-os assim:

```text
data/raw/
├── linkedin-job-postings/
├── resume-jd-match/
└── job-skill-set/
```

### Opção 2 — Download usando Kaggle API

Para baixar datasets do Kaggle pelo terminal, primeiro instale a biblioteca:

```bash
pip install kaggle
```

Depois, configure sua chave da API do Kaggle.

No Kaggle:

1. Acesse sua conta.
2. Vá em `Settings`.
3. Procure a seção `API`.
4. Clique em `Create New Token`.
5. O arquivo `kaggle.json` será baixado.

No Linux, coloque o arquivo em:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Depois, execute os comandos:

```bash
kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw/linkedin-job-postings --unzip
```

```bash
kaggle datasets download -d batuhanmutlu/job-skill-set -p data/raw/job-skill-set --unzip
```

### Opção 3 — Download usando Hugging Face

Para carregar o dataset Resume-JD-Match, instale a biblioteca `datasets`:

```bash
pip install datasets
```

O dataset poderá ser carregado em Python usando a biblioteca da Hugging Face e depois salvo localmente em `data/raw/resume-jd-match/`.

O download é feito automaticamente por `src/data_loader.py` na primeira
execução do treino. A organização esperada é:

```text
data/raw/resume-jd-match/
└── resume_jd_match.csv
```

## Ambiente Virtual

Crie um ambiente virtual:

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
source .venv/bin/activate
```

Atualize o `pip`:

```bash
pip install --upgrade pip
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Bibliotecas Previstas

As principais bibliotecas previstas para o projeto são:

```text
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
plotly
streamlit
joblib
jupyter
ipykernel
kaggle
datasets
pyarrow
python-dotenv
tqdm
```

## Papel de Cada Biblioteca

### Manipulação e análise de dados

* `pandas`: leitura, tratamento e manipulação dos datasets.
* `numpy`: operações numéricas e vetoriais.
* `scipy`: suporte a cálculos científicos e métricas auxiliares.

### Machine Learning

* `scikit-learn`: criação de modelos de classificação, vetorização com TF-IDF, similaridade por cosseno, divisão treino/teste e métricas.
* `joblib`: salvamento e carregamento de modelos treinados.

### Visualização de dados

* `matplotlib`: gráficos básicos para EDA.
* `seaborn`: gráficos estatísticos para análise exploratória.
* `plotly`: gráficos interativos, especialmente úteis na aplicação.

### Interface

* `streamlit`: criação da aplicação web interativa.

### Notebooks e experimentação

* `jupyter`: criação e execução de notebooks.
* `ipykernel`: integração do ambiente virtual com notebooks.

### Download e leitura de datasets

* `kaggle`: download dos datasets hospedados no Kaggle.
* `datasets`: carregamento de datasets do Hugging Face.
* `pyarrow`: leitura e escrita de arquivos `.parquet`.

### Organização e utilidades

* `python-dotenv`: gerenciamento de variáveis de ambiente.
* `tqdm`: barras de progresso em processamentos maiores.

## Pipeline Inicial do Projeto

A primeira versão do projeto será construída em etapas:

1. Baixar e organizar os datasets.
2. Realizar análise exploratória dos dados.
3. Tratar textos de currículos e vagas.
4. Criar vetores usando TF-IDF.
5. Treinar um modelo baseline de classificação Fit/No Fit.
6. Avaliar o modelo usando métricas de classificação.
7. Criar ranking de vagas por similaridade.
8. Identificar skills compatíveis e faltantes.
9. Criar interface em Streamlit.
10. Documentar resultados e conclusões.

## Modelos

### Classificação Fit / No-Fit

O baseline (Regressão Logística + TF-IDF sobre o texto concatenado vaga + currículo)
fica limitado a ~0,60 de acurácia porque trata os dois lados como um único saco de
palavras, descartando o sinal de *casamento* entre vaga e currículo.

Para superar isso, `src/features.py` separa vaga e currículo e cria atributos de
**interação** (similaridade do cosseno, termos em comum, cobertura, Jaccard e SVD
da matriz de interação). Sobre esses atributos, `src/training.py` treina e compara
os modelos previstos no projeto:

* Regressão Logística;
* SVM Linear;
* Complement Naive Bayes;
* Random Forest;
* Gradient Boosting (HistGradientBoosting).

O melhor modelo (por F1) tem o limiar de decisão calibrado por validação cruzada
no treino e é salvo em `models/modelo_fit_no_fit.joblib`, consumido por
`src/model_predict.py`.

Resultados no conjunto de teste do Resume-JD-Match (limiar padrão de 0,5):

| Modelo | Atributos | Acurácia | F1 | ROC-AUC |
|---|---|---|---|---|
| **Gradient Boosting** | match | **0,67** | **0,69** | **0,72** |
| Random Forest | match | 0,66 | 0,68 | 0,72 |
| Regressão Logística | match | 0,65 | 0,68 | 0,70 |
| Regressão Logística (baseline) | TF-IDF | 0,61 | 0,59 | 0,66 |
| Complement Naive Bayes | TF-IDF | 0,58 | 0,59 | 0,61 |
| SVM Linear | TF-IDF | 0,60 | 0,50 | 0,66 |

> Todos os modelos com features de *match* superam os do baseline: a engenharia de
> atributos pesa mais do que a escolha do algoritmo. Com o limiar calibrado, o
> Gradient Boosting chega a ~0,68 de acurácia e ~0,71 de F1 na classe Fit.

### Ranking de vagas

* TF-IDF + similaridade por cosseno (`src/recommendation.py`).

### Análise de skills

* Comparação de skills compatíveis e faltantes entre currículo e vaga
  (`src/skills.py`).

### Estimativa salarial (opcional)

Caso existam dados suficientes:

* Random Forest Regressor;
* Gradient Boosting Regressor.

## Métricas Previstas

Para classificação:

* acurácia;
* precisão;
* recall;
* F1-score;
* matriz de confusão.

Para regressão salarial:

* MAE;
* RMSE;
* R².

Para recomendação:

* análise qualitativa do ranking;
* comparação entre score de similaridade e aderência esperada;
* avaliação das Top-5 vagas retornadas.

## Como treinar e avaliar os modelos

O dataset Resume-JD-Match é baixado automaticamente do Hugging Face na primeira
execução. Para treinar, comparar os modelos e salvar o melhor:

```bash
python -m src.training
```

Isso gera, na pasta `models/`:

* `modelo_fit_no_fit.joblib`: featurizer + melhor modelo + limiar calibrado;
* `comparacao_modelos.csv` e `.md`: tabela comparativa das métricas;
* `metricas_melhor_modelo.json`: métricas do melhor modelo no teste.

O notebook `notebooks/06_comparacao_modelos_fit_no_fit.ipynb` reproduz a
comparação com tabelas e gráficos (matrizes de confusão e curvas ROC).

Para classificar um par currículo/vaga com o modelo treinado:

```python
from src.model_predict import predict_fit

predict_fit(curriculo="...", descricao_vaga="...")
# -> {'status': 'Fit', 'score_confianca': 0.85, 'probabilidade_fit': 0.85, 'modelo': '...'}
```

## Como rodar a interface

A interface web é feita em Streamlit e integra os três módulos (classificação
Fit/No-Fit, ranking Top-5 e análise de skills). Para iniciá-la:

```bash
streamlit run app/streamlit_app.py
```

O usuário cola o currículo, e a aplicação retorna o score de aderência, a
classificação Fit/No-Fit, o ranking das vagas mais compatíveis (a partir de
`data/sample/vagas_exemplo.csv`) e a análise de skills compatíveis/faltantes.

## Observação

Este projeto é construído de forma incremental e já integra ponta a ponta: EDA dos
datasets, composição da base de vagas, classificação Fit/No-Fit (baseline e
comparação de modelos), ranking de vagas, análise de skills e interface Streamlit.
A estimativa de faixa salarial permanece como evolução opcional.
