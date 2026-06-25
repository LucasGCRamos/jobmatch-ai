"""
Pipeline de treino e **comparação de modelos** para o classificador Fit / No-Fit.

Evolui o baseline original (Regressão Logística + TF-IDF sobre o texto
concatenado) em duas frentes:

1. **Engenharia de atributos** (:mod:`src.features`): separa vaga e currículo e
   constrói features de casamento (similaridade, sobreposição de termos, SVD da
   interação). Isso ataca a principal limitação do baseline, que misturava os
   dois lados em um único saco de palavras.
2. **Comparação de modelos**: treina e avalia, lado a lado, os modelos previstos
   no projeto — Regressão Logística, SVM Linear, Random Forest, Complement Naive
   Bayes e Gradient Boosting — em métricas de classificação (acurácia, precisão,
   recall, F1 e ROC-AUC).

O melhor modelo (por F1 no conjunto de teste) tem o limiar de decisão calibrado
por validação cruzada **no treino** (sem vazamento do teste) e é serializado num
único *bundle* (``models/modelo_fit_no_fit.joblib``) consumido por
:mod:`src.model_predict`.

Uso::

    python -m src.training          # treina, compara e salva o melhor modelo
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data_loader import carregar_resume_jd_match, garantir_pastas_projeto
from src.features import JobMatchFeaturizer

# No Fit = 0 ; Potential/Good Fit = 1 (binarização exigida nos critérios de aceite).
MAPA_LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 1}
SEMENTE = 42

ARQUIVO_MODELO = "modelo_fit_no_fit.joblib"
ARQUIVO_COMPARACAO_CSV = "comparacao_modelos.csv"
ARQUIVO_COMPARACAO_MD = "comparacao_modelos.md"


# --------------------------------------------------------------------------- #
# Registro de modelos
# --------------------------------------------------------------------------- #
def _modelos_baseline() -> dict:
    """Modelos sobre o TF-IDF do texto concatenado (abordagem saco de palavras)."""
    return {
        "LogReg (TF-IDF baseline)": LogisticRegression(max_iter=1000, random_state=SEMENTE),
        "LinearSVM (TF-IDF)": LinearSVC(C=1.0, random_state=SEMENTE),
        "ComplementNB (TF-IDF)": ComplementNB(),
    }


def _modelos_match() -> dict:
    """Modelos sobre as features de casamento vaga↔currículo (:mod:`src.features`)."""
    return {
        # StandardScaler ajuda os modelos lineares; árvores ignoram a escala.
        "LogReg (features match)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=SEMENTE)
        ),
        "RandomForest (features match)": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEMENTE
        ),
        "HistGradientBoosting (features match)": HistGradientBoostingClassifier(
            max_iter=800,
            learning_rate=0.04,
            l2_regularization=2.0,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEMENTE,
        ),
    }


def _todos_estimadores() -> dict:
    return {**_modelos_baseline(), **_modelos_match()}


# --------------------------------------------------------------------------- #
# Helpers de avaliação
# --------------------------------------------------------------------------- #
def _scores_positivos(modelo, X) -> np.ndarray:
    """Score da classe positiva: probabilidade quando disponível, senão margem."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def _metricas(y_true, y_pred, scores) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def _melhor_limiar(y_true, scores) -> float:
    """Limiar que maximiza o F1, estimado sobre os scores fornecidos."""
    precisao, recall, limiares = precision_recall_curve(y_true, scores)
    f1 = 2 * precisao * recall / (precisao + recall + 1e-9)
    indice = int(np.argmax(f1))
    # precision_recall_curve devolve um limiar a menos que pontos de P/R.
    return float(limiares[min(indice, len(limiares) - 1)])


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def treinar_e_comparar(project_root=None, verbose: bool = True) -> dict:
    """Treina, compara e persiste o melhor modelo. Retorna um dicionário-resumo."""

    def log(*args):
        if verbose:
            print(*args)

    log("--- CARREGANDO DADOS (Resume-JD-Match) ---")
    df = carregar_resume_jd_match(project_root)
    df["y"] = df["label"].map(MAPA_LABELS)
    if df["y"].isna().any():
        raise ValueError("Há labels fora de {No Fit, Potential Fit, Good Fit}.")

    treino = df[df["split"] == "train"]
    teste = df[df["split"] == "test"]
    y_tr, y_te = treino["y"].to_numpy(), teste["y"].to_numpy()
    log(f"[INFO] Treino: {len(treino)} | Teste: {len(teste)} | "
        f"Fit no treino: {y_tr.mean():.1%}")

    # ----- Representações de atributos ------------------------------------ #
    log("\n--- EXTRAINDO ATRIBUTOS ---")
    vec_baseline = TfidfVectorizer(max_features=5000, stop_words="english")
    Xb_tr = vec_baseline.fit_transform(treino["text"])
    Xb_te = vec_baseline.transform(teste["text"])
    log(f"[INFO] TF-IDF baseline: {Xb_tr.shape[1]} features")

    featurizer = JobMatchFeaturizer(random_state=SEMENTE)
    Xm_tr = featurizer.fit_transform(treino["text"])
    Xm_te = featurizer.transform(teste["text"])
    log(f"[INFO] Features de match: {Xm_tr.shape[1]} features")

    grupos = {
        "baseline": (_modelos_baseline(), Xb_tr, Xb_te),
        "match": (_modelos_match(), Xm_tr, Xm_te),
    }

    # ----- Treino + avaliação de cada modelo ------------------------------ #
    log("\n--- TREINANDO E AVALIANDO MODELOS ---")
    linhas, ajustados = [], {}
    for grupo, (modelos, X_tr, X_te) in grupos.items():
        for nome, estimador in modelos.items():
            inicio = time.time()
            modelo = clone(estimador).fit(X_tr, y_tr)
            scores = _scores_positivos(modelo, X_te)
            y_pred = modelo.predict(X_te)
            m = _metricas(y_te, y_pred, scores)
            m.update(modelo=nome, grupo=grupo, tipo_features=grupo,
                     tem_proba=hasattr(modelo, "predict_proba"),
                     tempo_treino_s=round(time.time() - inicio, 1))
            linhas.append(m)
            ajustados[nome] = modelo
            log(f"  {nome:38s} acc={m['acuracia']:.4f} f1={m['f1']:.4f} "
                f"auc={m['roc_auc']:.4f} ({m['tempo_treino_s']}s)")

    comparacao = (
        pd.DataFrame(linhas)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
        [["modelo", "grupo", "acuracia", "precisao", "recall", "f1", "roc_auc",
          "tem_proba", "tempo_treino_s", "tipo_features"]]
    )

    # ----- Seleção + calibração de limiar do melhor ----------------------- #
    melhor = comparacao.iloc[0]
    nome_melhor = melhor["modelo"]
    tipo_melhor = melhor["tipo_features"]
    modelo_melhor = ajustados[nome_melhor]
    X_tr_best = Xb_tr if tipo_melhor == "baseline" else Xm_tr
    X_te_best = Xb_te if tipo_melhor == "baseline" else Xm_te

    log(f"\n--- MELHOR MODELO: {nome_melhor} (F1={melhor['f1']:.4f}) ---")
    log("[INFO] Calibrando limiar por validação cruzada no treino...")
    estimador_base = clone(_todos_estimadores()[nome_melhor])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    metodo = "predict_proba" if hasattr(estimador_base, "predict_proba") else "decision_function"
    oof = cross_val_predict(estimador_base, X_tr_best, y_tr, cv=cv, method=metodo, n_jobs=-1)
    scores_oof = oof[:, 1] if metodo == "predict_proba" else oof
    limiar = _melhor_limiar(y_tr, scores_oof)
    auc_cv = roc_auc_score(y_tr, scores_oof)
    log(f"[INFO] Limiar calibrado = {limiar:.3f} | ROC-AUC (CV treino) = {auc_cv:.4f}")

    scores_te = _scores_positivos(modelo_melhor, X_te_best)
    limiar_05 = 0.5 if melhor["tem_proba"] else 0.0
    y_pred_05 = (scores_te >= limiar_05).astype(int)
    y_pred_cal = (scores_te >= limiar).astype(int)
    log("\n=== MELHOR MODELO NO TESTE (limiar calibrado) ===")
    log("Acurácia:", round(accuracy_score(y_te, y_pred_cal), 4))
    log("\n" + classification_report(y_te, y_pred_cal, target_names=["No Fit", "Fit"]))
    log("Matriz de confusão:\n", confusion_matrix(y_te, y_pred_cal))

    # ----- Persistência --------------------------------------------------- #
    pastas = garantir_pastas_projeto(project_root)
    featurizer_para_salvar = vec_baseline if tipo_melhor == "baseline" else featurizer
    bundle = {
        "tipo_features": tipo_melhor,
        "featurizer": featurizer_para_salvar,
        "modelo": modelo_melhor,
        "nome_modelo": nome_melhor,
        "limiar": limiar,
        "usa_proba": bool(melhor["tem_proba"]),
        "metricas_teste": {
            "limiar_0.5": _metricas(y_te, y_pred_05, scores_te),
            "limiar_calibrado": _metricas(y_te, y_pred_cal, scores_te),
            "roc_auc_cv_treino": auc_cv,
        },
    }
    caminho_modelo = pastas["models"] / ARQUIVO_MODELO
    joblib.dump(bundle, caminho_modelo)

    comparacao.to_csv(pastas["models"] / ARQUIVO_COMPARACAO_CSV, index=False)
    (pastas["models"] / ARQUIVO_COMPARACAO_MD).write_text(
        _tabela_markdown(comparacao, nome_melhor), encoding="utf-8"
    )
    (pastas["models"] / "metricas_melhor_modelo.json").write_text(
        json.dumps(bundle["metricas_teste"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log(f"\n[SUCESSO] Melhor modelo salvo em: {caminho_modelo}")
    log(f"[SUCESSO] Tabela de comparação em: {pastas['models'] / ARQUIVO_COMPARACAO_MD}")
    log("\n" + _tabela_markdown(comparacao, nome_melhor))

    return {"comparacao": comparacao, "bundle": bundle, "caminho_modelo": caminho_modelo}


def _tabela_markdown(comparacao: pd.DataFrame, nome_melhor: str) -> str:
    cabecalho = "| Modelo | Grupo | Acurácia | Precisão | Recall | F1 | ROC-AUC |\n"
    cabecalho += "|---|---|---|---|---|---|---|\n"
    linhas = []
    for _, r in comparacao.iterrows():
        destaque = " **(melhor)**" if r["modelo"] == nome_melhor else ""
        linhas.append(
            f"| {r['modelo']}{destaque} | {r['grupo']} | {r['acuracia']:.4f} | "
            f"{r['precisao']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"
        )
    return cabecalho + "\n".join(linhas) + "\n"


# Compatibilidade retroativa: o nome antigo continua funcionando.
def train_baseline_model(project_root=None):
    return treinar_e_comparar(project_root)


if __name__ == "__main__":
    treinar_e_comparar()
