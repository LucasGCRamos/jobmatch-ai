"""
Engenharia de atributos para o classificador Fit / No-Fit do JobMatch AI.

O dataset Resume-JD-Match traz cada par currículo/vaga em um único texto no
formato:

    "For the given job description << <descrição da vaga> >> <currículo>"

A abordagem baseline trata esse texto concatenado como um único "saco de
palavras" (TF-IDF). Isso descarta justamente o sinal mais importante do
problema: o quanto o currículo *casa* com a vaga. Termos que aparecem nos dois
lados (skills em comum) ficam misturados com termos que aparecem em apenas um.

Este módulo separa as duas partes e constrói atributos de *interação* que
capturam esse casamento:

* similaridade do cosseno entre vaga e currículo;
* número de termos em comum (sobreposição) e índice de Jaccard;
* cobertura: fração dos termos da vaga que aparecem no currículo;
* representações latentes (TruncatedSVD) do texto completo e da matriz de
  interação termo-a-termo (J ⊙ C).

A classe :class:`JobMatchFeaturizer` segue a convenção ``fit``/``transform`` do
scikit-learn e é serializável com ``joblib``, sendo reaproveitada tanto no
treino (:mod:`src.training`) quanto na inferência (:mod:`src.model_predict`).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Template usado no dataset Resume-JD-Match para juntar vaga + currículo.
TEMPLATE_TEXTO = "For the given job description <<{vaga}>> {curriculo}"

# Captura o conteúdo entre "<<" e ">>" (vaga) e o restante (currículo).
_PADRAO_SPLIT = re.compile(r"<<(.*?)>>(.*)", re.DOTALL)

# Nomes das features escalares, na mesma ordem em que são geradas.
NOMES_FEATURES_ESCALARES = [
    "cos_vaga_cv",        # similaridade do cosseno vaga x currículo
    "termos_em_comum",    # nº de termos compartilhados
    "jaccard",            # interseção / união dos termos
    "cobertura_vaga",     # termos_em_comum / nº termos da vaga
    "log_tamanho_vaga",   # log(1 + comprimento da vaga em caracteres)
    "log_tamanho_cv",     # log(1 + comprimento do currículo em caracteres)
    "n_termos_vaga",      # nº de termos distintos na vaga
    "n_termos_cv",        # nº de termos distintos no currículo
]


def montar_texto_par(curriculo: str, descricao_vaga: str) -> str:
    """Monta o texto no mesmo formato visto no treino (vaga + currículo)."""
    return TEMPLATE_TEXTO.format(
        vaga=(descricao_vaga or "").strip(),
        curriculo=(curriculo or "").strip(),
    )


def separar_vaga_curriculo(texto: str) -> tuple[str, str]:
    """Separa um texto do dataset em ``(descrição da vaga, currículo)``."""
    correspondencia = _PADRAO_SPLIT.search(texto or "")
    if correspondencia:
        return correspondencia.group(1).strip(), correspondencia.group(2).strip()
    # Sem o delimitador esperado: trata tudo como currículo.
    return "", (texto or "").strip()


class JobMatchFeaturizer:
    """Transforma textos vaga+currículo em uma matriz densa de atributos.

    Parameters
    ----------
    max_features, ngram_range, min_df, max_df:
        Hiperparâmetros do :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
        compartilhado entre vaga e currículo.
    n_svd_texto:
        Componentes do SVD sobre o TF-IDF do texto completo.
    n_svd_interacao:
        Componentes do SVD sobre a matriz de interação termo-a-termo.
    random_state:
        Semente para reprodutibilidade dos SVDs.
    """

    def __init__(
        self,
        max_features: int = 30_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 3,
        max_df: float = 0.9,
        n_svd_texto: int = 100,
        n_svd_interacao: int = 60,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.n_svd_texto = n_svd_texto
        self.n_svd_interacao = n_svd_interacao
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Helpers internos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _separar(textos: Sequence[str]) -> tuple[list[str], list[str]]:
        vagas, curriculos = [], []
        for texto in textos:
            vaga, curriculo = separar_vaga_curriculo(texto)
            vagas.append(vaga)
            curriculos.append(curriculo)
        return vagas, curriculos

    def _vetores_normalizados(self, textos: Sequence[str]) -> spmatrix:
        return normalize(self.vectorizer_.transform(textos))

    def _features_escalares(
        self,
        textos: Sequence[str],
        vagas: Sequence[str],
        curriculos: Sequence[str],
        matriz_vaga: spmatrix,
        matriz_cv: spmatrix,
    ) -> np.ndarray:
        interacao = matriz_vaga.multiply(matriz_cv)
        cosseno = np.asarray(interacao.sum(axis=1)).ravel()
        termos_comuns = np.asarray((interacao > 0).sum(axis=1)).ravel()
        n_vaga = np.asarray((matriz_vaga > 0).sum(axis=1)).ravel()
        n_cv = np.asarray((matriz_cv > 0).sum(axis=1)).ravel()
        jaccard = termos_comuns / (n_vaga + n_cv - termos_comuns + 1e-9)
        cobertura = termos_comuns / (n_vaga + 1e-9)
        log_tam_vaga = np.log1p([len(v) for v in vagas])
        log_tam_cv = np.log1p([len(c) for c in curriculos])
        return np.column_stack(
            [
                cosseno,
                termos_comuns,
                jaccard,
                cobertura,
                log_tam_vaga,
                log_tam_cv,
                n_vaga,
                n_cv,
            ]
        )

    def _ajustar_svd(self, n_componentes: int, matriz: spmatrix) -> TruncatedSVD:
        # n_components precisa ser < nº de features e < nº de amostras.
        limite = max(1, min(n_componentes, matriz.shape[1] - 1, matriz.shape[0] - 1))
        svd = TruncatedSVD(n_components=limite, random_state=self.random_state)
        svd.fit(matriz)
        return svd

    # ------------------------------------------------------------------ #
    # API pública (estilo scikit-learn)
    # ------------------------------------------------------------------ #
    def fit(self, textos: Iterable[str], y=None) -> "JobMatchFeaturizer":
        textos = list(textos)
        vagas, curriculos = self._separar(textos)

        # Vocabulário compartilhado: a mesma palavra precisa ter o mesmo índice
        # na vaga e no currículo para que a interação faça sentido.
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            stop_words="english",
        )
        self.vectorizer_.fit(vagas + curriculos)

        matriz_texto = self.vectorizer_.transform(textos)
        self.svd_texto_ = self._ajustar_svd(self.n_svd_texto, matriz_texto)

        matriz_vaga = self._vetores_normalizados(vagas)
        matriz_cv = self._vetores_normalizados(curriculos)
        self.svd_interacao_ = self._ajustar_svd(
            self.n_svd_interacao, matriz_vaga.multiply(matriz_cv)
        )

        self.feature_names_ = (
            list(NOMES_FEATURES_ESCALARES)
            + [f"svd_texto_{i}" for i in range(self.svd_texto_.n_components)]
            + [f"svd_interacao_{i}" for i in range(self.svd_interacao_.n_components)]
        )
        return self

    def transform(self, textos: Iterable[str]) -> np.ndarray:
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("Chame fit() antes de transform().")
        textos = list(textos)
        vagas, curriculos = self._separar(textos)

        matriz_vaga = self._vetores_normalizados(vagas)
        matriz_cv = self._vetores_normalizados(curriculos)
        interacao = matriz_vaga.multiply(matriz_cv)

        escalares = self._features_escalares(
            textos, vagas, curriculos, matriz_vaga, matriz_cv
        )
        svd_texto = self.svd_texto_.transform(self.vectorizer_.transform(textos))
        svd_interacao = self.svd_interacao_.transform(interacao)
        return np.hstack([escalares, svd_texto, svd_interacao]).astype(np.float32)

    def fit_transform(self, textos: Iterable[str], y=None) -> np.ndarray:
        return self.fit(textos, y).transform(textos)

    def transform_par(self, curriculo: str, descricao_vaga: str) -> np.ndarray:
        """Atalho para gerar features de um único par (currículo, vaga)."""
        return self.transform([montar_texto_par(curriculo, descricao_vaga)])
