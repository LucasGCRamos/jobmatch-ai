# JobMatch AI 🎯

Projeto E2E (End-to-End) de Machine Learning para recomendação inteligente de vagas e análise de compatibilidade entre currículos e oportunidades profissionais.

---

## 📌 Visão Geral do Projeto

O JobMatch AI é uma aplicação integrada que utiliza técnicas de Processamento de Linguagem Natural (NLP) e Machine Learning para reduzir a assimetria de informações no mercado de trabalho. 

A plataforma recebe o perfil profissional (currículo) do candidato em texto livre e executa um pipeline de análise em tempo real, entregando:
* **Score de Confiança:** Probabilidade matemática de aderência do candidato à vaga.
* **Classificação (Fit / No Fit):** Predição binária se o perfil atende aos requisitos mínimos da oportunidade.
* **Ranking Top-5:** Recomendação das 5 vagas mais adequadas ao perfil usando métricas de similaridade.
* **Análise de Skills:** Mapeamento granular das habilidades compatíveis presentes no currículo e das habilidades faltantes exigidas pela vaga principal.

---

## 🧠 Arquitetura e Pipeline de Dados

Para que o colega apresentador ou qualquer desenvolvedor entenda o fluxo de dados, a nossa aplicação funciona na seguinte esteira de processamento:

1. **Ingestão e Processamento (NLP):** O currículo do usuário é recebido pela interface em Streamlit e vetorizado utilizando a técnica TF-IDF. Isso transforma o texto humano em uma representação matemática.
2. **Motor de Recomendação (`src/recommendation.py`):** O vetor do candidato é comparado com a nossa base de vagas pré-processada (`df_vagas`) utilizando Similaridade por Cosseno. O sistema devolve os IDs e as informações vitais das 5 vagas mais aderentes.
3. **Cruzamento de Texto Completo:** O backend realiza uma busca inteligente pelo `job_id` da vaga Top-1 na base original para resgatar o `texto_vaga_completo`.
4. **Modelo de Predição (`src/model_predict.py`):** O texto completo da vaga e do currículo são submetidos ao nosso modelo classificador treinado (ex: Logistic Regression / Random Forest), que retorna o status de *Fit* e o *Score de Confiança*.
5. **Extração de Habilidades (`src/skills.py`):** Algoritmos de busca e dicionários de termos cruzam as exigências da vaga com a experiência do candidato, separando os resultados nas chaves de `skills_compativeis` e `skills_faltantes`.

---

## 📂 Datasets Utilizados

O motor do JobMatch AI foi alimentado e treinado utilizando três bases de dados robustas (não versionadas no GitHub por limitação de tamanho):

* **LinkedIn Job Postings 2023-2024:** Base contendo vagas reais, empresas, descrições e modalidades de trabalho.
* **Resume-JD-Match:** Base rotulada de pares currículo-vaga utilizada para ensinar o modelo a classificar Fit/No Fit.
* **Job Skill Set Dataset:** Dicionário massivo de habilidades exigidas por cargo.

> **Nota para execução:** Os arquivos devem ser alocados nas pastas `data/raw/` e os resultados dos tratamentos em `data/processed/`.

---

## 🚀 Como Configurar e Executar o Projeto Localmente

Siga o passo a passo abaixo para levantar a infraestrutura, treinar o modelo e rodar a aplicação na sua máquina.

### 1. Configuração do Ambiente Virtual (Windows/PowerShell)
Crie um ambiente isolado para não gerar conflitos de bibliotecas:
```powershell
py -m venv .venv
```
Libere temporariamente a execução de scripts no PowerShell (necessário no Windows) e ative o ambiente:
```powershell
Set-ExecutionPolicy Unrestricted -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 2. Instalação de Dependências
Com o ambiente ativado `(.venv)`, instale os pacotes necessários (Pandas, Scikit-Learn, Streamlit, etc):
```powershell
pip install -r requirements.txt`
```

### 3. Treinamento do Modelo de Machine Learning
**Passo crucial:** O repositório não armazena os arquivos binários pesados do modelo (`.pkl` ou `.joblib`). É necessário gerar os artefatos de IA localmente antes de abrir a interface.

Execute o script de treino apontando a raiz do projeto como módulo:
```powershell
python -m src.training
```
*Aguarde a conclusão. Os artefatos do modelo treinado serão salvos automaticamente na pasta `models/.`*

### 4. Inicialização da Interface Integrada
Com o ambiente configurado, dependências instaladas e modelo treinado, inicie o servidor da aplicação Web:
```powershell
streamlit run app/streamlit_app.py
```
O seu navegador padrão abrirá automaticamente a interface do JobMatch AI pronta para uso!

## 🛠️ Tecnologias Empregadas
* **Linguagem:** Python
* **Interface Gráfica:** Streamlit
* **Manipulação de Dados:** Pandas, Numpy
* **Machine Learning & NLP:** Scikit-Learn (TF-IDF, Cosseno)
* **Controle de Versão:** Git & GitHub

*Desenvolvido como Projeto de Disciplina para aplicação prática de IA e Machine Learning ponta a ponta.*