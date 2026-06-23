"""
Funções de carregamento de dados para o JobMatch AI.

Este módulo tenta carregar os datasets a partir de `data/raw/`. Caso eles não
estejam disponíveis localmente e `usar_kagglehub=True`, faz o download via
KaggleHub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def encontrar_raiz_projeto(caminho_inicial: Optional[Path | str] = None) -> Path:
    """Encontra a raiz do projeto procurando por `.git`, `README.md` ou `src/`."""
    caminho = Path(caminho_inicial or Path.cwd()).resolve()

    for atual in [caminho, *caminho.parents]:
        if (atual / ".git").exists():
            return atual
        if (atual / "README.md").exists() and (atual / "src").exists():
            return atual

    if caminho.name == "notebooks":
        return caminho.parent

    return caminho


def garantir_pastas_projeto(project_root: Optional[Path | str] = None) -> dict[str, Path]:
    """Cria e retorna as principais pastas usadas no projeto."""
    raiz = encontrar_raiz_projeto(project_root)

    pastas = {
        "root": raiz,
        "data": raiz / "data",
        "raw": raiz / "data" / "raw",
        "processed": raiz / "data" / "processed",
        "sample": raiz / "data" / "sample",
        "models": raiz / "models",
        "notebooks": raiz / "notebooks",
        "src": raiz / "src",
    }

    for chave in ["data", "raw", "processed", "sample", "models"]:
        pastas[chave].mkdir(parents=True, exist_ok=True)

    return pastas


def baixar_dataset_kagglehub(dataset_slug: str) -> Path:
    """Baixa um dataset pelo KaggleHub e retorna o caminho local do cache."""
    try:
        import kagglehub
    except ImportError as erro:
        raise ImportError(
            "A biblioteca kagglehub não está instalada. "
            "Instale com: pip install kagglehub"
        ) from erro

    caminho = kagglehub.dataset_download(dataset_slug)
    return Path(caminho)


def localizar_arquivo(pasta_base: Path, nome_arquivo: str) -> Optional[Path]:
    """Procura um arquivo pelo nome dentro de uma pasta, incluindo subpastas."""
    if not pasta_base.exists():
        return None

    encontrados = sorted(pasta_base.rglob(nome_arquivo))
    return encontrados[0] if encontrados else None


def resolver_pasta_dataset(
    project_root: Path | str,
    pasta_relativa_raw: str,
    kaggle_slug: Optional[str] = None,
    usar_kagglehub: bool = True,
) -> Path:
    """
    Resolve o caminho de um dataset.

    Primeiro tenta `data/raw/<pasta_relativa_raw>`. Se não encontrar arquivos
    e `usar_kagglehub=True`, baixa pelo KaggleHub.
    """
    pastas = garantir_pastas_projeto(project_root)
    pasta_local = pastas["raw"] / pasta_relativa_raw

    if pasta_local.exists() and any(pasta_local.rglob("*")):
        return pasta_local

    if kaggle_slug and usar_kagglehub:
        return baixar_dataset_kagglehub(kaggle_slug)

    return pasta_local


def carregar_csv(caminho: Path | str, **kwargs) -> pd.DataFrame:
    """Carrega um CSV com configuração padrão segura para bases grandes."""
    return pd.read_csv(caminho, low_memory=False, **kwargs)
