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
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│
├── notebooks/
│   ├── 01_eda_datasets.ipynb
│   └── 02_modelo_baseline.ipynb
│
├── reports/
│   └── proposta_projeto.md
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── recommendation.py
│   └── skills.py
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
* `data/README.md`: explica como baixar e organizar os datasets.

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

Exemplo de organização esperada:

```text
data/raw/resume-jd-match/
└── resume_jd_match.parquet
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

## Modelos Previstos

Para classificação Fit/No Fit:

* Logistic Regression;
* Random Forest Classifier;
* Linear SVM.

Para ranking de vagas:

* TF-IDF;
* similaridade por cosseno.

Para estimativa salarial, caso existam dados suficientes:

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

## Como rodar o projeto

Após instalar as dependências e baixar os datasets, execute a aplicação com:

```bash
streamlit run app/streamlit_app.py
```

## Observação

Este projeto está em desenvolvimento e será construído de forma incremental. A primeira versão terá foco em classificação Fit/No Fit usando NLP com TF-IDF. Em seguida, serão adicionadas as funcionalidades de ranking de vagas, análise de skills e interface final.
