"""
Módulo de análise de skills para o JobMatch AI (Issue 3).

Compara as habilidades presentes em um currículo com as habilidades exigidas
por uma vaga e retorna:

- as skills compatíveis (que o candidato já demonstra);
- as skills faltantes (exigidas pela vaga e ausentes no currículo);
- o percentual de aderência de skills;
- sugestões de desenvolvimento profissional.

A detecção de skills no currículo é feita por correspondência textual
normalizada (sem acento, em minúsculas e respeitando limites de palavra), o
que mantém o resultado simples e explicável, alinhado à proposta do projeto.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import pandas as pd

from src.preprocessing import normalizar_lista_skills, remover_acentos


def _normalizar_para_busca(texto: Any) -> str:
    """Normaliza um texto para a busca de skills (sem acento, minúsculo)."""
    texto = remover_acentos(texto).lower()
    texto = re.sub(r"[^a-z0-9+#.\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def skill_presente_no_texto(skill: Any, texto_normalizado: str) -> bool:
    """
    Indica se uma skill aparece em um texto já normalizado.

    A busca usa limites de caractere alfanumérico (em vez de `\\b`) para lidar
    melhor com símbolos técnicos como `c++` ou `c#`. Skills com mais de uma
    palavra são buscadas como expressão completa (ex.: "information technology").
    """
    skill_norm = _normalizar_para_busca(skill)
    if not skill_norm:
        return False

    padrao = r"(?<![a-z0-9+#.])" + re.escape(skill_norm) + r"(?![a-z0-9+#])"
    return re.search(padrao, texto_normalizado) is not None


def comparar_skills(curriculo: Any, skills_vaga: Any) -> dict[str, Any]:
    """
    Compara as skills do currículo com as skills exigidas pela vaga.

    Parameters
    ----------
    curriculo:
        Texto livre do currículo / perfil do candidato.
    skills_vaga:
        Skills da vaga. Aceita lista (`["python", "sql"]`) ou string no formato
        usado na base (`"python; sql; etl"`).

    Returns
    -------
    dict
        Dicionário com:
        - `skills_vaga`: skills da vaga normalizadas e sem duplicatas;
        - `skills_compativeis`: skills da vaga encontradas no currículo;
        - `skills_faltantes`: skills da vaga ausentes no currículo;
        - `qtd_skills_vaga`, `qtd_compativeis`, `qtd_faltantes`;
        - `percentual_aderencia`: percentual de skills compatíveis (0 a 100).
    """
    skills_vaga_norm = normalizar_lista_skills(skills_vaga)
    texto_norm = _normalizar_para_busca(curriculo)

    compativeis: list[str] = []
    faltantes: list[str] = []

    for skill in skills_vaga_norm:
        if texto_norm and skill_presente_no_texto(skill, texto_norm):
            compativeis.append(skill)
        else:
            faltantes.append(skill)

    total = len(skills_vaga_norm)
    percentual = round(len(compativeis) / total * 100, 1) if total else 0.0

    return {
        "skills_vaga": skills_vaga_norm,
        "skills_compativeis": compativeis,
        "skills_faltantes": faltantes,
        "qtd_skills_vaga": total,
        "qtd_compativeis": len(compativeis),
        "qtd_faltantes": len(faltantes),
        "percentual_aderencia": percentual,
    }


def sugerir_desenvolvimento(
    skills_faltantes: Sequence[str],
    max_sugestoes: int = 5,
) -> list[str]:
    """
    Gera sugestões de desenvolvimento a partir das skills faltantes.

    Retorna uma lista de frases curtas, prontas para exibição na interface.
    """
    skills_faltantes = list(skills_faltantes)

    if not skills_faltantes:
        return ["O currículo já cobre as principais skills exigidas pela vaga."]

    sugestoes = [
        f"Desenvolver ou evidenciar a competência: {skill}."
        for skill in skills_faltantes[:max_sugestoes]
    ]

    restantes = len(skills_faltantes) - max_sugestoes
    if restantes > 0:
        sugestoes.append(
            f"Há mais {restantes} skill(s) da vaga para desenvolver no futuro."
        )

    return sugestoes


def extrair_vocabulario_skills(
    df_vagas: pd.DataFrame,
    coluna_skills: str = "skills",
) -> list[str]:
    """
    Constrói o vocabulário de skills conhecidas a partir da base de vagas.

    Útil para identificar quais skills o candidato possui mesmo sem uma vaga
    específica, comparando o currículo com todas as skills observadas na base.
    """
    if coluna_skills not in df_vagas.columns:
        raise KeyError(f"A coluna de skills '{coluna_skills}' não existe na base.")

    vocabulario: set[str] = set()
    for valor in df_vagas[coluna_skills]:
        vocabulario.update(normalizar_lista_skills(valor))

    return sorted(vocabulario)


def extrair_skills_do_curriculo(
    curriculo: Any,
    vocabulario_skills: Sequence[str],
) -> list[str]:
    """
    Extrai do currículo as skills presentes em um vocabulário conhecido.

    Percorre o vocabulário (por exemplo, todas as skills da base) e retorna
    aquelas que aparecem no texto do currículo.
    """
    texto_norm = _normalizar_para_busca(curriculo)
    if not texto_norm:
        return []

    encontradas = [
        skill
        for skill in vocabulario_skills
        if skill_presente_no_texto(skill, texto_norm)
    ]
    return sorted(set(encontradas))


def analisar_skills_recomendacoes(
    curriculo: Any,
    df_recomendacoes: pd.DataFrame,
    coluna_skills: str = "skills",
) -> pd.DataFrame:
    """
    Enriquece um DataFrame de recomendações com a análise de skills por vaga.

    Para cada vaga recomendada, compara as skills do currículo com as skills da
    vaga e adiciona as colunas:
    - `skills_compativeis`;
    - `skills_faltantes`;
    - `percentual_aderencia_skills`.

    Esta função conecta a recomendação (Top-5) à análise de skills, formato
    consumido depois pela interface Streamlit.
    """
    df = df_recomendacoes.copy()

    if coluna_skills not in df.columns:
        df["skills_compativeis"] = [[] for _ in range(len(df))]
        df["skills_faltantes"] = [[] for _ in range(len(df))]
        df["percentual_aderencia_skills"] = 0.0
        return df

    compativeis_col: list[list[str]] = []
    faltantes_col: list[list[str]] = []
    percentuais: list[float] = []

    for skills_vaga in df[coluna_skills]:
        analise = comparar_skills(curriculo, skills_vaga)
        compativeis_col.append(analise["skills_compativeis"])
        faltantes_col.append(analise["skills_faltantes"])
        percentuais.append(analise["percentual_aderencia"])

    df["skills_compativeis"] = compativeis_col
    df["skills_faltantes"] = faltantes_col
    df["percentual_aderencia_skills"] = percentuais

    return df
