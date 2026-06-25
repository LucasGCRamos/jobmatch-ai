"""
Módulo de recomendação de vagas para o JobMatch AI (Issue 3).

Implementa a recomendação Top-N de vagas a partir de um currículo em texto,
usando TF-IDF e similaridade por cosseno sobre a coluna `texto_vaga_completo`
da base de vagas enriquecidas.

A API principal é:

- `recomendar_vagas`: função direta que recebe um currículo e uma base de
  vagas e retorna as Top-N vagas mais aderentes, ordenadas por score.
- `RecomendadorVagas`: classe que ajusta o vetorizador TF-IDF uma única vez e
  permite reaproveitar a matriz de vagas em múltiplas consultas. É a forma
  recomendada para a interface Streamlit, que carrega a base apenas uma vez.

O módulo funciona com o arquivo de exemplo `data/sample/vagas_exemplo.csv` e
com a base final `data/processed/vagas_enriquecidas_lite.parquet`, pois ambos
seguem o mesmo contrato de colunas.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import remover_acentos


COLUNA_TEXTO_PADRAO = "texto_vaga_completo"

# Colunas exibidas no resultado da recomendação. São filtradas pelas que
# realmente existem na base recebida, para funcionar com qualquer versão.
COLUNAS_EXIBICAO_PADRAO = [
    "job_id",
    "title",
    "company_name",
    "location",
    "skills",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_status",
    "possui_salario",
    "salary_range_text",
    "pay_period",
    "description_preview",
    "job_posting_url",
]

# Nomes possíveis do arquivo da base final, em ordem de preferência.
_NOMES_BASE_PROCESSADA = [
    "vagas_enriquecidas_lite.parquet",
    "vagas_enriquecidas_lite.csv",
    "vagas_enriquecidas_tratada.csv",
    "vagas_enriquecidas.csv",
]


def preparar_texto_para_tfidf(texto: Any) -> str:
    """
    Normaliza um texto antes da vetorização TF-IDF.

    Remove acentos, coloca em minúsculas e mantém apenas caracteres úteis para
    comparação textual (letras, números e alguns símbolos técnicos como `+`,
    `#` e `.`, presentes em termos como `c++`, `c#` e `.net`).
    """
    texto = remover_acentos(texto).lower()
    texto = re.sub(r"[^a-z0-9+#.\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


class RecomendadorVagas:
    """
    Recomendador de vagas baseado em TF-IDF e similaridade por cosseno.

    Ao ser criado, ajusta um vetorizador TF-IDF sobre a coluna de texto das
    vagas (por padrão `texto_vaga_completo`) e guarda a matriz resultante.
    Cada chamada de `recomendar` apenas projeta o currículo no mesmo espaço
    vetorial e calcula a similaridade, o que torna consultas repetidas baratas.
    """

    def __init__(
        self,
        df_vagas: pd.DataFrame,
        coluna_texto: str = COLUNA_TEXTO_PADRAO,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = 50_000,
    ) -> None:
        if coluna_texto not in df_vagas.columns:
            raise KeyError(
                f"A coluna de texto '{coluna_texto}' não existe na base de vagas. "
                f"Colunas disponíveis: {list(df_vagas.columns)}"
            )

        if df_vagas.empty:
            raise ValueError("A base de vagas está vazia; não é possível recomendar.")

        self.coluna_texto = coluna_texto
        self.df_vagas = df_vagas.reset_index(drop=True).copy()

        self.vectorizer = TfidfVectorizer(
            preprocessor=preparar_texto_para_tfidf,
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=True,
        )

        textos = self.df_vagas[coluna_texto].fillna("").astype(str)
        self.matriz_tfidf = self.vectorizer.fit_transform(textos)

    def calcular_similaridades(self, curriculo: str) -> np.ndarray:
        """Retorna o vetor de similaridade do currículo com todas as vagas."""
        if not isinstance(curriculo, str) or not curriculo.strip():
            raise ValueError("O currículo informado está vazio.")

        vetor_curriculo = self.vectorizer.transform([curriculo])
        similaridades = cosine_similarity(vetor_curriculo, self.matriz_tfidf).ravel()
        return similaridades

    def recomendar(
        self,
        curriculo: str,
        top_n: int = 5,
        colunas_retorno: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Recomenda as Top-N vagas mais aderentes ao currículo.

        Retorna um DataFrame ordenado por score (maior primeiro), contendo:
        - `posicao`: posição no ranking (1 = mais aderente);
        - `score_aderencia`: similaridade por cosseno (0 a 1);
        - `aderencia_pct`: o mesmo score em escala de 0 a 100, para exibição;
        - as colunas de exibição da vaga (título, empresa, salário, link etc.).
        """
        similaridades = self.calcular_similaridades(curriculo)

        quantidade = max(1, min(int(top_n), len(similaridades)))

        # argpartition seleciona os maiores sem ordenar tudo; depois ordenamos
        # apenas os candidatos selecionados (eficiente para bases grandes).
        candidatos = np.argpartition(similaridades, -quantidade)[-quantidade:]
        indices_ordenados = candidatos[np.argsort(similaridades[candidatos])[::-1]]

        if colunas_retorno is None:
            colunas_retorno = [
                coluna
                for coluna in COLUNAS_EXIBICAO_PADRAO
                if coluna in self.df_vagas.columns
            ]
        else:
            colunas_retorno = [
                coluna for coluna in colunas_retorno if coluna in self.df_vagas.columns
            ]

        resultado = self.df_vagas.iloc[indices_ordenados][colunas_retorno].copy()

        scores = similaridades[indices_ordenados]
        resultado.insert(0, "aderencia_pct", np.round(scores * 100, 1))
        resultado.insert(0, "score_aderencia", np.round(scores, 4))
        resultado.insert(0, "posicao", range(1, len(resultado) + 1))

        return resultado.reset_index(drop=True)


def recomendar_vagas(
    curriculo: str,
    df_vagas: pd.DataFrame,
    top_n: int = 5,
    coluna_texto: str = COLUNA_TEXTO_PADRAO,
    colunas_retorno: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Recomenda as Top-N vagas mais aderentes a um currículo.

    Esta é a função direta exigida pela Issue 3. Internamente ela cria um
    `RecomendadorVagas`, ajusta o TF-IDF sobre `df_vagas[coluna_texto]` e
    calcula a similaridade por cosseno com o currículo.

    Parameters
    ----------
    curriculo:
        Texto livre do currículo / perfil do candidato.
    df_vagas:
        Base de vagas (arquivo de exemplo ou base final), com a coluna de texto.
    top_n:
        Quantidade de vagas a retornar (padrão 5).
    coluna_texto:
        Coluna usada na comparação textual (padrão `texto_vaga_completo`).
    colunas_retorno:
        Colunas da vaga a incluir no resultado. Se None, usa um conjunto padrão.

    Returns
    -------
    pandas.DataFrame
        Ranking ordenado por `score_aderencia` (maior primeiro).
    """
    recomendador = RecomendadorVagas(
        df_vagas,
        coluna_texto=coluna_texto,
        **kwargs,
    )
    return recomendador.recomendar(
        curriculo,
        top_n=top_n,
        colunas_retorno=colunas_retorno,
    )


def carregar_base_vagas(
    caminho: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
    usar_amostra_se_faltar: bool = True,
) -> pd.DataFrame:
    """
    Carrega a base de vagas para a recomendação.

    Ordem de busca:
    1. `caminho`, se informado (aceita `.parquet` ou `.csv`);
    2. a primeira base encontrada em `data/processed/` (preferindo a lite
       em Parquet, que é a versão recomendada);
    3. o arquivo de exemplo `data/sample/vagas_exemplo.csv`, caso
       `usar_amostra_se_faltar=True`.

    Assim o módulo pode ser testado com o arquivo de exemplo e, quando a base
    final estiver disponível, passa a usá-la automaticamente.
    """
    if caminho is not None:
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo de vagas não encontrado: {caminho}")
        return _ler_base(caminho)

    raiz = _resolver_raiz_projeto(project_root)
    pasta_processed = raiz / "data" / "processed"

    for nome in _NOMES_BASE_PROCESSADA:
        candidato = pasta_processed / nome
        if candidato.exists():
            return _ler_base(candidato)

    if usar_amostra_se_faltar:
        caminho_amostra = raiz / "data" / "sample" / "vagas_exemplo.csv"
        if caminho_amostra.exists():
            return _ler_base(caminho_amostra)

    raise FileNotFoundError(
        "Nenhuma base de vagas encontrada. Gere a base final (Issue 1) ou "
        "garanta que data/sample/vagas_exemplo.csv exista."
    )


def _ler_base(caminho: Path) -> pd.DataFrame:
    """Lê uma base de vagas a partir de CSV ou Parquet."""
    if caminho.suffix.lower() == ".parquet":
        return pd.read_parquet(caminho)
    return pd.read_csv(caminho, low_memory=False)


def _resolver_raiz_projeto(project_root: Optional[Path | str]) -> Path:
    """Resolve a raiz do projeto, subindo a partir de `notebooks/` se preciso."""
    if project_root is not None:
        return Path(project_root)

    atual = Path.cwd()
    if atual.name == "notebooks":
        return atual.parent
    return atual
