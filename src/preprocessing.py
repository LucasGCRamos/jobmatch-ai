"""
Funções de pré-processamento textual e tratamento de skills para o JobMatch AI.

Este arquivo concentra utilidades reutilizáveis pelos módulos de composição,
recomendação, análise de skills e modelagem.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from typing import Iterable, Any

import pandas as pd


def normalizar_texto_basico(texto: Any) -> str:
    """Remove excesso de espaços e converte valores nulos para string vazia."""
    if pd.isna(texto):
        return ""

    texto = str(texto).replace("\n", " ").replace("\r", " ").strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def remover_acentos(texto: Any) -> str:
    """Remove acentos de uma string."""
    texto = normalizar_texto_basico(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    return texto


def normalizar_cargo(cargo: Any) -> str:
    """Normaliza título de vaga para comparações auxiliares."""
    cargo = remover_acentos(cargo).lower()
    cargo = re.sub(r"[^a-z0-9\s]", " ", cargo)
    cargo = re.sub(r"\s+", " ", cargo).strip()
    return cargo


def converter_para_lista(valor: Any) -> list[str]:
    """
    Converte valores de skills para lista.

    Aceita:
    - listas reais;
    - tuplas ou conjuntos;
    - strings no formato de lista Python, exemplo: "['Python', 'SQL']";
    - strings separadas por vírgula, ponto e vírgula ou pipe.
    """
    if valor is None:
        return []

    if isinstance(valor, (list, tuple, set)):
        return [
            str(item).strip()
            for item in valor
            if item is not None
            and str(item).strip()
            and str(item).strip().lower() not in {"nan", "none", "null"}
        ]

    try:
        if pd.isna(valor):
            return []
    except Exception:
        pass

    texto = str(valor).strip()

    if texto == "" or texto.lower() in {"nan", "none", "null"}:
        return []

    if texto.startswith("[") and texto.endswith("]"):
        try:
            resultado = ast.literal_eval(texto)
            if isinstance(resultado, list):
                return [
                    str(item).strip()
                    for item in resultado
                    if item is not None
                    and str(item).strip()
                    and str(item).strip().lower() not in {"nan", "none", "null"}
                ]
        except Exception:
            pass

    separador = None
    for candidato in [";", "|", ","]:
        if candidato in texto:
            separador = candidato
            break

    if separador is None:
        return [texto]

    return [item.strip() for item in texto.split(separador) if item.strip()]


def normalizar_skill(skill: Any) -> str:
    """Normaliza uma skill para comparação."""
    skill = normalizar_texto_basico(skill).lower()
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill


def normalizar_lista_skills(skills: Any) -> list[str]:
    """Converte e normaliza uma coleção de skills, removendo duplicatas."""
    lista = converter_para_lista(skills)
    normalizadas = [normalizar_skill(skill) for skill in lista]
    normalizadas = [skill for skill in normalizadas if skill]
    return sorted(set(normalizadas))


def unir_listas_skills(*listas: Any) -> list[str]:
    """Une várias listas/strings de skills em uma única lista normalizada e sem duplicatas."""
    resultado: list[str] = []

    for lista in listas:
        resultado.extend(normalizar_lista_skills(lista))

    return sorted(set(resultado))


def lista_para_string(lista: Any, separador: str = "; ") -> str:
    """Transforma uma lista de skills em string padronizada."""
    return separador.join(normalizar_lista_skills(lista))


def combinar_textos(*partes: Any) -> str:
    """Combina vários campos textuais em uma única string limpa."""
    textos = []

    for parte in partes:
        texto = normalizar_texto_basico(parte)
        if texto:
            textos.append(texto)

    return " ".join(textos)
