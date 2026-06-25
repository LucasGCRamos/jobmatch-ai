"""
Composição dos datasets de vagas para o JobMatch AI.

Este módulo cria a base final de vagas enriquecidas a partir de:
- LinkedIn Job Postings 2023-2024;
- Job Skill Set Dataset.

A base Resume-JD-Match não entra na composição de vagas porque seu papel é
treinar o modelo supervisionado Fit / No Fit. A base final de vagas vem dos
datasets de vagas reais e skills.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from src.data_loader import (
    carregar_csv,
    garantir_pastas_projeto,
    localizar_arquivo,
    resolver_pasta_dataset,
)
from src.preprocessing import (
    combinar_textos,
    lista_para_string,
    normalizar_cargo,
    normalizar_lista_skills,
    unir_listas_skills,
)


COLUNAS_BASE_FINAL = [
    "job_id",
    "title",
    "description",
    "company_name",
    "location",
    "skills",
    "texto_vaga_completo",
    "min_salary",
    "med_salary",
    "max_salary",
    "pay_period",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_allowed",
    "job_posting_url",
]


COLUNAS_BASE_TRATADA = [
    "job_id",
    "title",
    "description",
    "company_name",
    "location",
    "skills",
    "texto_vaga_completo",
    "min_salary",
    "med_salary",
    "max_salary",
    "pay_period",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_allowed",
    "remote_status",
    "remote_allowed_flag",
    "possui_salario",
    "salary_min_display",
    "salary_med_display",
    "salary_max_display",
    "salary_range_text",
    "job_posting_url",
]


COLUNAS_BASE_LITE = [
    "job_id",
    "title",
    "company_name",
    "location",
    "skills",
    "texto_vaga_completo",
    "description_preview",
    "formatted_work_type",
    "formatted_experience_level",
    "remote_status",
    "remote_allowed_flag",
    "possui_salario",
    "salary_range_text",
    "pay_period",
    "job_posting_url",
]


def padronizar_job_id(valor: Any) -> str:
    """Padroniza job_id como string, evitando diferenças como 123.0 vs 123."""
    if valor is None or pd.isna(valor):
        return ""

    texto = str(valor).strip()

    if texto == "" or texto.lower() in {"nan", "none", "null"}:
        return ""

    try:
        numero = float(texto)
        if numero.is_integer():
            return str(int(numero))
    except Exception:
        pass

    if texto.endswith(".0") and texto[:-2].isdigit():
        return texto[:-2]

    return texto


def _primeiro_valor_nao_vazio(serie: pd.Series) -> Any:
    """Retorna o primeiro valor não vazio de uma série."""
    for valor in serie:
        if pd.notna(valor) and str(valor).strip() != "":
            return valor
    return np.nan


def _unir_skills_serie(serie: pd.Series) -> list[str]:
    """Une skills vindas de várias linhas agrupadas."""
    skills: list[str] = []

    for valor in serie:
        skills = unir_listas_skills(skills, valor)

    return skills


def _formatar_numero_salario(valor: Any) -> Optional[float]:
    """Converte salário para float, retornando None quando não for possível."""
    if valor is None or pd.isna(valor):
        return None

    try:
        return float(valor)
    except Exception:
        return None


def preparar_linkedin_jobs(
    project_root: Optional[Path | str] = None,
    usar_kagglehub: bool = True,
) -> pd.DataFrame:
    """
    Carrega e padroniza o LinkedIn Job Postings.

    Retorna uma base com uma linha por vaga e skills mapeadas, quando os
    arquivos `job_skills.csv` e `skills.csv` estiverem disponíveis.
    """
    pastas = garantir_pastas_projeto(project_root)

    pasta_linkedin = resolver_pasta_dataset(
        project_root=pastas["root"],
        pasta_relativa_raw="linkedin-job-postings",
        kaggle_slug="arshkon/linkedin-job-postings",
        usar_kagglehub=usar_kagglehub,
    )

    caminho_postings = localizar_arquivo(pasta_linkedin, "postings.csv")

    if caminho_postings is None:
        raise FileNotFoundError(
            "Arquivo postings.csv não encontrado. "
            "Baixe o dataset LinkedIn Job Postings ou habilite usar_kagglehub=True."
        )

    df = carregar_csv(caminho_postings)
    df["job_id"] = df["job_id"].apply(padronizar_job_id)

    colunas_necessarias = [
        "job_id",
        "title",
        "description",
        "company_name",
        "location",
        "company_id",
        "min_salary",
        "med_salary",
        "max_salary",
        "pay_period",
        "formatted_work_type",
        "formatted_experience_level",
        "remote_allowed",
        "job_posting_url",
    ]

    for coluna in colunas_necessarias:
        if coluna not in df.columns:
            df[coluna] = np.nan

    caminho_companies = localizar_arquivo(pasta_linkedin, "companies.csv")

    if caminho_companies is not None and "company_id" in df.columns:
        df_companies = carregar_csv(caminho_companies)

        if {"company_id", "name"}.issubset(df_companies.columns):
            df_companies = df_companies[["company_id", "name"]].drop_duplicates("company_id")

            df = df.merge(
                df_companies,
                on="company_id",
                how="left",
            )

            df["company_name"] = df["company_name"].fillna(df["name"])
            df = df.drop(columns=["name"])

    caminho_job_skills = localizar_arquivo(pasta_linkedin, "job_skills.csv")
    caminho_skills = localizar_arquivo(pasta_linkedin, "skills.csv")

    if caminho_job_skills is not None:
        df_job_skills = carregar_csv(caminho_job_skills)
        df_job_skills["job_id"] = df_job_skills["job_id"].apply(padronizar_job_id)

        if caminho_skills is not None:
            df_skills_map = carregar_csv(caminho_skills)

            if (
                {"skill_abr", "skill_name"}.issubset(df_skills_map.columns)
                and "skill_abr" in df_job_skills.columns
            ):
                df_job_skills = df_job_skills.merge(
                    df_skills_map,
                    on="skill_abr",
                    how="left",
                )
                coluna_skill = "skill_name"

            elif "skill_abr" in df_job_skills.columns:
                coluna_skill = "skill_abr"

            else:
                coluna_skill = None

        else:
            coluna_skill = "skill_abr" if "skill_abr" in df_job_skills.columns else None

        if coluna_skill is not None:
            df_job_skills["skill_normalizada"] = df_job_skills[coluna_skill].apply(
                lambda valor: lista_para_string([valor])
            )

            skills_por_vaga = (
                df_job_skills.groupby("job_id")["skill_normalizada"]
                .apply(lambda valores: sorted(set(v for v in valores if str(v).strip())))
                .reset_index(name="skills_linkedin_lista")
            )

            df = df.merge(
                skills_por_vaga,
                on="job_id",
                how="left",
            )

        else:
            df["skills_linkedin_lista"] = [[] for _ in range(len(df))]

    else:
        df["skills_linkedin_lista"] = [[] for _ in range(len(df))]

    df["skills_linkedin_lista"] = df["skills_linkedin_lista"].apply(
        lambda valor: valor if isinstance(valor, list) else []
    )

    df["skills_linkedin"] = df["skills_linkedin_lista"].apply(lista_para_string)
    df["title_normalized"] = df["title"].apply(normalizar_cargo)

    colunas_saida = [
        "job_id",
        "title",
        "title_normalized",
        "description",
        "company_name",
        "location",
        "min_salary",
        "med_salary",
        "max_salary",
        "pay_period",
        "formatted_work_type",
        "formatted_experience_level",
        "remote_allowed",
        "job_posting_url",
        "skills_linkedin",
        "skills_linkedin_lista",
    ]

    df = df[colunas_saida].copy()
    df = df[df["job_id"].str.strip() != ""].drop_duplicates("job_id")

    return df


def preparar_job_skill_set(
    project_root: Optional[Path | str] = None,
    usar_kagglehub: bool = True,
) -> pd.DataFrame:
    """Carrega e padroniza o Job Skill Set Dataset."""
    pastas = garantir_pastas_projeto(project_root)

    pasta_job_skill = resolver_pasta_dataset(
        project_root=pastas["root"],
        pasta_relativa_raw="job-skill-set",
        kaggle_slug="batuhanmutlu/job-skill-set",
        usar_kagglehub=usar_kagglehub,
    )

    caminho_csv = localizar_arquivo(pasta_job_skill, "all_job_post.csv")

    if caminho_csv is None:
        raise FileNotFoundError(
            "Arquivo all_job_post.csv não encontrado. "
            "Baixe o dataset Job Skill Set ou habilite usar_kagglehub=True."
        )

    df = carregar_csv(caminho_csv)

    colunas_necessarias = [
        "job_id",
        "category",
        "job_title",
        "job_description",
        "job_skill_set",
    ]

    for coluna in colunas_necessarias:
        if coluna not in df.columns:
            df[coluna] = np.nan

    df["job_id"] = df["job_id"].apply(padronizar_job_id)
    df["title_normalized"] = df["job_title"].apply(normalizar_cargo)
    df["skills_job_skill_set_lista"] = df["job_skill_set"].apply(normalizar_lista_skills)
    df["skills_job_skill_set"] = df["skills_job_skill_set_lista"].apply(lista_para_string)

    colunas_saida = [
        "job_id",
        "category",
        "job_title",
        "title_normalized",
        "job_description",
        "skills_job_skill_set",
        "skills_job_skill_set_lista",
    ]

    df = df[colunas_saida].copy()
    df = df[df["job_id"].str.strip() != ""]

    df = (
        df.groupby("job_id", as_index=False)
        .agg(
            {
                "category": _primeiro_valor_nao_vazio,
                "job_title": _primeiro_valor_nao_vazio,
                "title_normalized": _primeiro_valor_nao_vazio,
                "job_description": _primeiro_valor_nao_vazio,
                "skills_job_skill_set": _primeiro_valor_nao_vazio,
                "skills_job_skill_set_lista": _unir_skills_serie,
            }
        )
    )

    df["skills_job_skill_set"] = df["skills_job_skill_set_lista"].apply(lista_para_string)

    return df


def verificar_intersecao_job_id(
    df_linkedin: pd.DataFrame,
    df_job_skill: pd.DataFrame,
) -> dict[str, float | int]:
    """Calcula a interseção de job_id entre LinkedIn e Job Skill Set."""
    ids_linkedin = set(df_linkedin["job_id"].dropna().astype(str))
    ids_job_skill = set(df_job_skill["job_id"].dropna().astype(str))
    ids_comum = ids_linkedin.intersection(ids_job_skill)

    return {
        "qtd_vagas_linkedin": len(ids_linkedin),
        "qtd_vagas_job_skill_set": len(ids_job_skill),
        "qtd_job_ids_em_comum": len(ids_comum),
        "percentual_linkedin_com_match": round(len(ids_comum) / len(ids_linkedin) * 100, 2)
        if ids_linkedin
        else 0,
        "percentual_job_skill_com_match": round(len(ids_comum) / len(ids_job_skill) * 100, 2)
        if ids_job_skill
        else 0,
    }


def compor_datasets_vagas(
    df_linkedin: pd.DataFrame,
    df_job_skill: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """
    Compõe LinkedIn Job Postings com Job Skill Set.

    Estratégia:
    - LinkedIn é a base principal de vagas reais.
    - Job Skill Set entra como enriquecimento de skills por `job_id`.
    - É usado left join para preservar as vagas do LinkedIn.
    - Antes do merge, os dois lados são deduplicados por `job_id`.
    """
    linkedin = df_linkedin.copy()
    job_skill = df_job_skill.copy()

    linkedin["job_id"] = linkedin["job_id"].apply(padronizar_job_id)
    job_skill["job_id"] = job_skill["job_id"].apply(padronizar_job_id)

    linkedin = linkedin[linkedin["job_id"].str.strip() != ""].drop_duplicates("job_id")
    job_skill = job_skill[job_skill["job_id"].str.strip() != ""].drop_duplicates("job_id")

    relatorio = verificar_intersecao_job_id(linkedin, job_skill)

    df = linkedin.merge(
        job_skill[
            [
                "job_id",
                "category",
                "job_title",
                "job_description",
                "skills_job_skill_set",
                "skills_job_skill_set_lista",
            ]
        ],
        on="job_id",
        how="left",
        validate="one_to_one",
    )

    df["title"] = df["title"].fillna(df["job_title"])
    df["description"] = df["description"].fillna(df["job_description"])

    df["skills_lista"] = df.apply(
        lambda linha: unir_listas_skills(
            linha.get("skills_linkedin_lista", []),
            linha.get("skills_job_skill_set_lista", []),
        ),
        axis=1,
    )

    df["skills"] = df["skills_lista"].apply(lista_para_string)

    df["texto_vaga_completo"] = df.apply(
        lambda linha: combinar_textos(
            linha.get("title", ""),
            linha.get("description", ""),
            linha.get("skills", ""),
        ),
        axis=1,
    )

    for coluna in COLUNAS_BASE_FINAL:
        if coluna not in df.columns:
            df[coluna] = np.nan

    df_final = df[COLUNAS_BASE_FINAL].copy()
    df_final = df_final.drop_duplicates("job_id")

    relatorio.update(
        {
            "qtd_linhas_base_final": len(df_final),
            "qtd_job_id_duplicados_final": int(df_final["job_id"].duplicated().sum()),
            "qtd_vagas_com_skills": int(
                df_final["skills"].fillna("").astype(str).str.strip().ne("").sum()
            ),
            "qtd_vagas_com_texto_completo": int(
                df_final["texto_vaga_completo"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),
        }
    )

    return df_final, relatorio


def validar_base_final(df: pd.DataFrame) -> dict[str, Any]:
    """Valida se a base final atende ao contrato da Issue 1."""
    colunas_faltantes = [coluna for coluna in COLUNAS_BASE_FINAL if coluna not in df.columns]

    return {
        "possui_colunas_obrigatorias": len(colunas_faltantes) == 0,
        "colunas_faltantes": colunas_faltantes,
        "qtd_linhas": len(df),
        "qtd_job_id_duplicados": int(df["job_id"].duplicated().sum())
        if "job_id" in df.columns
        else None,
        "qtd_textos_vazios": int(
            df["texto_vaga_completo"].fillna("").astype(str).str.strip().eq("").sum()
        )
        if "texto_vaga_completo" in df.columns
        else None,
        "qtd_skills_vazias": int(
            df["skills"].fillna("").astype(str).str.strip().eq("").sum()
        )
        if "skills" in df.columns
        else None,
    }


def tratar_base_enriquecida(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata a base enriquecida sem inventar valores artificiais.

    Salários originais são preservados com nulos, pois preencher com média ou
    mediana criaria informações que não existem na vaga.

    São criadas colunas auxiliares para facilitar uso na aplicação:
    - remote_status;
    - remote_allowed_flag;
    - possui_salario;
    - salary_min_display;
    - salary_med_display;
    - salary_max_display;
    - salary_range_text.
    """
    df_tratado = df.copy()

    for coluna in COLUNAS_BASE_FINAL:
        if coluna not in df_tratado.columns:
            df_tratado[coluna] = np.nan

    df_tratado["job_id"] = df_tratado["job_id"].apply(padronizar_job_id)
    df_tratado["title"] = df_tratado["title"].fillna("").astype(str)
    df_tratado["description"] = df_tratado["description"].fillna("").astype(str)
    df_tratado["company_name"] = df_tratado["company_name"].fillna("NOT_INFORMED").astype(str)
    df_tratado["location"] = df_tratado["location"].fillna("NOT_INFORMED").astype(str)
    df_tratado["skills"] = df_tratado["skills"].fillna("").astype(str)
    df_tratado["texto_vaga_completo"] = df_tratado["texto_vaga_completo"].fillna("").astype(str)

    df_tratado["formatted_work_type"] = (
        df_tratado["formatted_work_type"]
        .fillna("NOT_INFORMED")
        .astype(str)
    )

    df_tratado["formatted_experience_level"] = (
        df_tratado["formatted_experience_level"]
        .fillna("NOT_INFORMED")
        .astype(str)
    )

    df_tratado["pay_period"] = (
        df_tratado["pay_period"]
        .fillna("NOT_INFORMED")
        .astype(str)
    )

    df_tratado["job_posting_url"] = (
        df_tratado["job_posting_url"]
        .fillna("")
        .astype(str)
    )

    remote_num = pd.to_numeric(df_tratado["remote_allowed"], errors="coerce")

    df_tratado["remote_status"] = np.where(
        remote_num == 1,
        "REMOTE_ALLOWED",
        "NOT_INFORMED",
    )

    df_tratado["remote_allowed_flag"] = np.where(
        remote_num == 1,
        1,
        0,
    )

    for coluna in ["min_salary", "med_salary", "max_salary"]:
        df_tratado[coluna] = pd.to_numeric(df_tratado[coluna], errors="coerce")

    df_tratado["possui_salario"] = (
        df_tratado["min_salary"].notna()
        | df_tratado["med_salary"].notna()
        | df_tratado["max_salary"].notna()
    )

    df_tratado["salary_min_display"] = df_tratado["min_salary"]
    df_tratado["salary_med_display"] = df_tratado["med_salary"]
    df_tratado["salary_max_display"] = df_tratado["max_salary"]

    df_tratado["salary_min_display"] = df_tratado["salary_min_display"].fillna(
        df_tratado["med_salary"]
    )

    df_tratado["salary_max_display"] = df_tratado["salary_max_display"].fillna(
        df_tratado["med_salary"]
    )

    def criar_faixa_salarial_texto(linha: pd.Series) -> str:
        if not bool(linha.get("possui_salario", False)):
            return "Não informado"

        min_salary = _formatar_numero_salario(linha.get("salary_min_display"))
        med_salary = _formatar_numero_salario(linha.get("salary_med_display"))
        max_salary = _formatar_numero_salario(linha.get("salary_max_display"))
        pay_period = linha.get("pay_period", "NOT_INFORMED")

        if min_salary is not None and max_salary is not None:
            return f"{min_salary:.2f} - {max_salary:.2f} ({pay_period})"

        if med_salary is not None:
            return f"{med_salary:.2f} ({pay_period})"

        return "Não informado"

    df_tratado["salary_range_text"] = df_tratado.apply(
        criar_faixa_salarial_texto,
        axis=1,
    )

    for coluna in COLUNAS_BASE_TRATADA:
        if coluna not in df_tratado.columns:
            df_tratado[coluna] = np.nan

    df_tratado = df_tratado[COLUNAS_BASE_TRATADA].copy()

    return df_tratado


def criar_base_lite(df_tratado: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma versão otimizada da base para uso nos próximos módulos.

    A coluna `description` completa é removida porque seu conteúdo já está
    representado em `texto_vaga_completo`, usado na recomendação.

    Para a interface, é criada a coluna `description_preview`, com uma prévia
    da descrição original.
    """
    df_lite = df_tratado.copy()

    if "description" not in df_lite.columns:
        df_lite["description"] = ""

    df_lite["description_preview"] = (
        df_lite["description"]
        .fillna("")
        .astype(str)
        .str.slice(0, 800)
    )

    for coluna in COLUNAS_BASE_LITE:
        if coluna not in df_lite.columns:
            df_lite[coluna] = ""

    df_lite = df_lite[COLUNAS_BASE_LITE].copy()

    return df_lite


def validar_base_lite(df: pd.DataFrame) -> dict[str, Any]:
    """Valida se a base lite atende ao contrato dos próximos módulos."""
    colunas_faltantes = [coluna for coluna in COLUNAS_BASE_LITE if coluna not in df.columns]

    return {
        "possui_colunas_obrigatorias": len(colunas_faltantes) == 0,
        "colunas_faltantes": colunas_faltantes,
        "qtd_linhas": len(df),
        "qtd_job_id_duplicados": int(df["job_id"].duplicated().sum())
        if "job_id" in df.columns
        else None,
        "qtd_textos_vazios": int(
            df["texto_vaga_completo"].fillna("").astype(str).str.strip().eq("").sum()
        )
        if "texto_vaga_completo" in df.columns
        else None,
        "qtd_skills_vazias": int(
            df["skills"].fillna("").astype(str).str.strip().eq("").sum()
        )
        if "skills" in df.columns
        else None,
    }


def salvar_base_final(
    df: pd.DataFrame,
    project_root: Optional[Path | str] = None,
    nome_arquivo: str = "vagas_enriquecidas.csv",
) -> Path:
    """Salva uma base em CSV dentro de data/processed/."""
    pastas = garantir_pastas_projeto(project_root)
    caminho_saida = pastas["processed"] / nome_arquivo

    df.to_csv(caminho_saida, index=False)

    return caminho_saida


def salvar_base_lite(
    df_lite: pd.DataFrame,
    project_root: Optional[Path | str] = None,
    nome_csv: str = "vagas_enriquecidas_lite.csv",
    nome_parquet: str = "vagas_enriquecidas_lite.parquet",
) -> dict[str, Path]:
    """Salva a base lite em CSV e Parquet dentro de data/processed/."""
    pastas = garantir_pastas_projeto(project_root)

    caminho_csv = pastas["processed"] / nome_csv
    caminho_parquet = pastas["processed"] / nome_parquet

    df_lite.to_csv(caminho_csv, index=False)
    df_lite.to_parquet(caminho_parquet, index=False)

    return {
        "csv": caminho_csv,
        "parquet": caminho_parquet,
    }


def criar_amostra_versionavel(
    df: pd.DataFrame,
    project_root: Optional[Path | str] = None,
    nome_arquivo: str = "vagas_exemplo.csv",
    n: int = 10,
) -> Path:
    """
    Cria um arquivo pequeno versionável em data/sample/.

    A amostra preserva as colunas do DataFrame recebido. Portanto, se a função
    for chamada com a base lite, o arquivo `vagas_exemplo.csv` terá as colunas
    otimizadas usadas pelos próximos desenvolvedores.
    """
    pastas = garantir_pastas_projeto(project_root)
    caminho_saida = pastas["sample"] / nome_arquivo

    if df.empty:
        raise ValueError("Não é possível criar amostra versionável a partir de uma base vazia.")

    if "skills" in df.columns:
        filtro_skills = df["skills"].fillna("").astype(str).str.strip().ne("")
        amostra = df[filtro_skills].head(n).copy()
    else:
        amostra = df.head(n).copy()

    if len(amostra) < n:
        amostra = df.head(n).copy()

    amostra.to_csv(caminho_saida, index=False)

    return caminho_saida


def gerar_base_vagas_enriquecidas(
    project_root: Optional[Path | str] = None,
    usar_kagglehub: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Executa o pipeline completo da Issue 1.

    Saídas criadas:
    - data/processed/vagas_enriquecidas.csv
    - data/processed/vagas_enriquecidas_tratada.csv
    - data/processed/vagas_enriquecidas_lite.csv
    - data/processed/vagas_enriquecidas_lite.parquet
    - data/sample/vagas_exemplo.csv

    Retorna a base lite, pois ela é a versão recomendada para uso nos módulos
    de recomendação, análise de skills e interface.
    """
    df_linkedin = preparar_linkedin_jobs(
        project_root=project_root,
        usar_kagglehub=usar_kagglehub,
    )

    df_job_skill = preparar_job_skill_set(
        project_root=project_root,
        usar_kagglehub=usar_kagglehub,
    )

    df_final, relatorio = compor_datasets_vagas(
        df_linkedin=df_linkedin,
        df_job_skill=df_job_skill,
    )

    df_tratado = tratar_base_enriquecida(df_final)
    df_lite = criar_base_lite(df_tratado)

    caminho_base_final = salvar_base_final(
        df_final,
        project_root=project_root,
        nome_arquivo="vagas_enriquecidas.csv",
    )

    caminho_base_tratada = salvar_base_final(
        df_tratado,
        project_root=project_root,
        nome_arquivo="vagas_enriquecidas_tratada.csv",
    )

    caminhos_lite = salvar_base_lite(
        df_lite,
        project_root=project_root,
        nome_csv="vagas_enriquecidas_lite.csv",
        nome_parquet="vagas_enriquecidas_lite.parquet",
    )

    caminho_amostra = criar_amostra_versionavel(
        df_lite,
        project_root=project_root,
        nome_arquivo="vagas_exemplo.csv",
        n=10,
    )

    relatorio["caminho_base_final"] = str(caminho_base_final)
    relatorio["caminho_base_tratada"] = str(caminho_base_tratada)
    relatorio["caminho_base_lite_csv"] = str(caminhos_lite["csv"])
    relatorio["caminho_base_lite_parquet"] = str(caminhos_lite["parquet"])
    relatorio["caminho_amostra"] = str(caminho_amostra)

    relatorio["validacao_base_final"] = validar_base_final(df_final)
    relatorio["validacao_base_tratada"] = {
        "qtd_linhas": len(df_tratado),
        "qtd_colunas": len(df_tratado.columns),
        "colunas": df_tratado.columns.tolist(),
    }
    relatorio["validacao_base_lite"] = validar_base_lite(df_lite)

    return df_lite, relatorio

if __name__ == "__main__":
    print("--- COMPONDO A BASE DE VAGAS ---")
    df_lite, relatorio = gerar_base_vagas_enriquecidas()

    print("\n=== BASE DE VAGAS GERADA ===")
    print(f"[INFO] Linhas na base lite: {len(df_lite)}")
    for chave in (
        "caminho_base_final",
        "caminho_base_tratada",
        "caminho_base_lite_csv",
        "caminho_base_lite_parquet",
        "caminho_amostra",
    ):
        if chave in relatorio:
            print(f"  {chave}: {relatorio[chave]}")
