# JobMatch AI 🎯

Projeto E2E de Machine Learning para recomendação inteligente de vagas e análise de compatibilidade entre currículos e oportunidades profissionais.

---

## 📌 Descrição do Projeto
O JobMatch AI é uma aplicação de Machine Learning voltada para analisar a compatibilidade entre currículos e vagas de emprego. A proposta do projeto é permitir que um usuário informe seu perfil profissional ou currículo em texto livre e receba como resposta:

* Score de aderência probabilística entre currículo e vaga;
* Classificação automática de perfil como Fit ou No Fit;
* Ranking com as Top-5 vagas mais compatíveis;
* Lista de skills compatíveis (encontradas no perfil);
* Lista de skills faltantes (exigidas pela vaga, mas ausentes no perfil);
* Sugestão de desenvolvimento profissional;
* Estimativa de faixa salarial (caso existam dados suficientes).

O projeto integra técnicas de Processamento de Linguagem Natural (NLP), classificação, similaridade textual, sistemas de recomendação e, opcionalmente, regressão para estimativa salarial.

---

## 🎯 Objetivo
Desenvolver uma solução E2E (End-to-End) de Machine Learning capaz de comparar currículos e vagas de emprego, reduzindo a assimetria de informações no mercado de trabalho ao indicar oportunidades perfeitamente alinhadas ao perfil do candidato.

---

## 💡 Motivação
Muitos candidatos se candidatam a vagas sem saber exatamente se possuem aderência aos requisitos solicitados. Ao mesmo tempo, empresas recebem muitos currículos desalinhados com as vagas disponíveis. O JobMatch AI busca resolver esse problema ao oferecer uma análise matemática e objetiva da compatibilidade entre o candidato e a oportunidade.

---

## 🧠 Arquitetura e Pipeline de Dados
Para que qualquer desenvolvedor ou avaliador entenda o fluxo de dados sob o capô, a nossa aplicação funciona na seguinte esteira de processamento:

1. **Processamento (NLP):** O currículo do usuário é recebido pela interface em Streamlit e vetorizado usando TF-IDF (transformando texto humano em representação matemática).
2. **Motor de Recomendação:** O vetor do candidato é comparado com a base de vagas utilizando Similaridade por Cosseno. O sistema devolve as 5 vagas mais aderentes.
3. **Cruzamento de Texto Completo:** O backend realiza uma busca inteligente pelo `job_id` da vaga principal na base original para resgatar o `texto_vaga_completo`.
4. **Classificação Fit/No Fit:** O texto da vaga e do currículo passam pelo nosso modelo classificador treinado, gerando o status binário e o Score de Confiança.
5. **Extração de Habilidades:** O algoritmo cruza as exigências da vaga com a experiência do candidato para mapear granularmente as compatibilidades e lacunas (`skills_compativeis` e `skills_faltantes`).

---

## 📂 Estrutura do Projeto

```text
jobmatch-ai/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
├── notebooks/
│   ├── 01_eda_datasets.ipynb
│   ├── 02_modelo_baseline.ipynb
│   ├── 03_eda_job_skill_set.ipynb
│   ├── 04_composicao_datasets.ipynb
│   ├── 05_modelo_baseline_fit_no_fit.ipynb
│   └── 05_recomendacao_e_skills.ipynb
├── reports/
│   └── proposta_projeto.md
├── src/
│   ├── compose_datasets.py
│   ├── data_loader.py
│   ├── model_predict.py
│   ├── preprocessing.py
│   ├── recommendation.py
│   ├── skills.py
│   └── training.py
├── .gitignore
├── README.md
└── requirements.txt

```

### Por que a pasta `data/` existe se os dados não sobem para o GitHub?

A pasta `data/` serve para organizar os datasets localmente durante o desenvolvimento. Os arquivos de dados normalmente são muito grandes e ultrapassam os limites recomendados para repositórios GitHub. Por isso, eles devem ser baixados manualmente e armazenados apenas na máquina de quem estiver executando o projeto.

* `data/raw/`: armazena os dados originais, sem tratamento.
* `data/processed/`: armazena os dados tratados e prontos para modelagem.

*Nota: Os arquivos dentro de `data/raw/` e `data/processed/` estão no `.gitignore` e não devem ser enviados para a nuvem.*

---

## 🗄️ Datasets Utilizados

O motor do JobMatch AI é alimentado por três bases principais:

1. **LinkedIn Job Postings 2023-2024:** Base com vagas reais, contendo título, descrição, empresa, localização e tipo de trabalho.
2. **Resume-JD-Match:** Base rotulada com pares de currículos e descrições de vagas (Fit / No Fit) para o treinamento do classificador.
3. **Job Skill Set Dataset:** Dicionário massivo com cargos, descrições e conjuntos de habilidades associadas.

### Como baixar os dados (Pré-requisito)

Os datasets não estão incluídos diretamente neste repositório. Para executar o projeto, faça o download via Kaggle API e Hugging Face:

```bash
pip install kaggle datasets
kaggle datasets download -d arshkon/linkedin-job-postings -p data/raw/linkedin-job-postings --unzip
kaggle datasets download -d batuhanmutlu/job-skill-set -p data/raw/job-skill-set --unzip

```

*O dataset `Resume-JD-Match` poderá ser carregado via script em Python usando a biblioteca da Hugging Face e salvo localmente em `data/raw/resume-jd-match/`.*

---

## 🚀 Como Configurar e Executar o Projeto Localmente

Siga o passo a passo abaixo para levantar a infraestrutura, treinar o modelo e rodar a aplicação integrada na sua máquina:

**1. Configuração do Ambiente Virtual:**
Crie e ative um ambiente isolado para não gerar conflitos de bibliotecas.

```powershell
python -m venv .venv

# No Windows (PowerShell):
Set-ExecutionPolicy Unrestricted -Scope Process
.\.venv\Scripts\Activate.ps1

# No Linux/Mac:
source .venv/bin/activate

```

**2. Instalação de Dependências:**
Com o `(.venv)` ativo, instale os pacotes necessários:

```powershell
pip install --upgrade pip
pip install -r requirements.txt

```

**3. Preparação dos Dados e Treinamento do Modelo (Crucial):**
Como os arquivos binários pesados não são versionados, **é obrigatório** compor os datasets e rodar o script de treinamento primeiro para gerar os artefatos de IA localmente. Os comandos abaixo usam a raiz do projeto como módulo:

Primeiro, unifique e prepare os datasets:
```powershell
python -m src.compose_datasets

```

Em seguida, execute o treinamento:

```powershell
python -m src.training

```

*(Aguarde o processo finalizar. Os arquivos `.pkl` ou `.joblib` aparecerão na pasta `models/`)*

**4. Executar a Aplicação:**
Com o modelo treinado, inicie o servidor da interface gráfica:

```powershell
streamlit run app/streamlit_app.py

```

O JobMatch AI abrirá automaticamente no seu navegador padrão!

---

## 🛠️ Tecnologias e Bibliotecas Empregadas

* **Manipulação e Análise de Dados:** `pandas` (leitura e tratamento), `numpy` (operações numéricas) e `scipy` (cálculos científicos).
* **Machine Learning & NLP:** `scikit-learn` (classificação, vetorização TF-IDF, similaridade por cosseno) e `joblib` (salvamento de modelos).
* **Visualização:** `matplotlib`, `seaborn` e `plotly`.
* **Interface Gráfica:** `streamlit`.
* **Controle de Versão:** Git & GitHub.

---

## 🤖 Modelos e Métricas

* **Para classificação Fit/No Fit:** Algoritmos testados incluem Logistic Regression, Random Forest Classifier e Linear SVM. Avaliação baseada em Acurácia, Precisão, Recall, F1-score e Matriz de Confusão.
* **Para ranking de vagas:** TF-IDF com Similaridade por Cosseno, avaliados mediante análise qualitativa das Top-5 recomendações.
