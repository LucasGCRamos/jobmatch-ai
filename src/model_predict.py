"""
Inferência Fit / No-Fit a partir de um currículo e uma descrição de vaga.

Consome o *bundle* salvo por :mod:`src.training`
(``models/modelo_fit_no_fit.joblib``), que contém o featurizer, o melhor modelo
e o limiar de decisão calibrado. Caso o bundle ainda não exista, faz *fallback*
para os artefatos do baseline antigo (``logistic_regression_baseline.pkl`` +
``tfidf_vectorizer.pkl``), mantendo a compatibilidade com o pipeline anterior.
"""

from __future__ import annotations

import math

import joblib
import numpy as np

from src.data_loader import garantir_pastas_projeto
from src.features import JobMatchFeaturizer, montar_texto_par

ARQUIVO_MODELO = "modelo_fit_no_fit.joblib"


def _score_para_confianca(score: float, usa_proba: bool) -> float:
    """Normaliza o score para [0, 1] (sigmoide quando não é probabilidade)."""
    if usa_proba:
        return float(score)
    return float(1.0 / (1.0 + math.exp(-score)))


def _prever_com_bundle(bundle: dict, curriculo: str, descricao_vaga: str) -> dict:
    modelo = bundle["modelo"]
    featurizer = bundle["featurizer"]
    limiar = bundle["limiar"]
    usa_proba = bundle["usa_proba"]

    if bundle["tipo_features"] == "match" and isinstance(featurizer, JobMatchFeaturizer):
        X = featurizer.transform_par(curriculo, descricao_vaga)
    else:  # baseline: TfidfVectorizer sobre o texto concatenado
        X = featurizer.transform([montar_texto_par(curriculo, descricao_vaga)])

    score = float(_scores_positivos(modelo, X)[0])
    eh_fit = score >= limiar
    prob_fit = _score_para_confianca(score, usa_proba)
    return {
        "status": "Fit" if eh_fit else "No Fit",
        "score_confianca": round(prob_fit if eh_fit else 1.0 - prob_fit, 4),
        "probabilidade_fit": round(prob_fit, 4),
        "modelo": bundle.get("nome_modelo", "desconhecido"),
    }


def _scores_positivos(modelo, X) -> np.ndarray:
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def _prever_legado(pastas, curriculo: str, descricao_vaga: str) -> dict:
    """Fallback para os artefatos do baseline antigo."""
    model_path = pastas["models"] / "logistic_regression_baseline.pkl"
    vectorizer_path = pastas["models"] / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Nenhum modelo encontrado em {pastas['models']}. "
            "Execute o treino primeiro: python -m src.training"
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform([montar_texto_par(curriculo, descricao_vaga)])
    predicao = int(model.predict(X)[0])
    probabilidades = model.predict_proba(X)[0]
    prob_fit = float(probabilidades[1])
    return {
        "status": "Fit" if predicao == 1 else "No Fit",
        "score_confianca": round(prob_fit if predicao == 1 else 1.0 - prob_fit, 4),
        "probabilidade_fit": round(prob_fit, 4),
        "modelo": "LogReg (TF-IDF baseline)",
    }


def predict_fit(curriculo: str, descricao_vaga: str) -> dict:
    """
    Classifica a aderência entre um currículo e uma descrição de vaga.

    Returns
    -------
    dict
        ``status`` ("Fit"/"No Fit"), ``score_confianca`` (confiança na decisão),
        ``probabilidade_fit`` (probabilidade da classe Fit) e ``modelo`` usado.
    """
    pastas = garantir_pastas_projeto()
    caminho_bundle = pastas["models"] / ARQUIVO_MODELO

    if caminho_bundle.exists():
        bundle = joblib.load(caminho_bundle)
        return _prever_com_bundle(bundle, curriculo, descricao_vaga)

    return _prever_legado(pastas, curriculo, descricao_vaga)


if __name__ == "__main__":
    # Teste rápido de fumaça (Sanity Check)
    vaga_teste = "Looking for a Data Scientist skilled in Python, SQL and Machine Learning."
    curriculo_teste = "Experienced Data Scientist with background in Python development and Scikit-Learn."

    print("\n=== TESTANDO INFERÊNCIA LOCAL ===")
    try:
        resultado = predict_fit(curriculo_teste, vaga_teste)
        print("Resultado do Teste:", resultado)
    except Exception as e:
        print("Erro ao rodar o teste:", e)
