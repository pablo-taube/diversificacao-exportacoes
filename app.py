# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS
==============================================================================
Arquivo único (Streamlit) que combina:

  1) Dados de importação mundial e do Brasil (UN Comtrade, nível SH6)
     -> usados para calcular RCA (Vantagem Comparativa Revelada), CAGR
        mundial e participação de mercado.
  2) Dados de exportação do Brasil por País, SH6, CGCE, CUCI Grupo,
     ISIC Divisão, ISIC Seção (Comexstat).
  3) (Opcional) Dados de exportação do Brasil por Estado (Comexstat/UF),
     usados na 4ª página, cruzados com o RCA/CAGR calculados a partir
     do Comtrade.

Páginas (navegação pela barra lateral, tudo em um único arquivo .py):
  - Dashboard Resumo
  - Tabela Completa (SH6)
  - CUCI Grupo (detalhamento)
  - Análise por Estado

Formatos aceitos:
  - Comtrade: csv, xlsx/xls, parquet, json
  - Comexstat (nacional e por UF): csv, xlsx/xls, parquet

Como executar:
  streamlit run app.py
==============================================================================
"""
from __future__ import annotations

import io
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 0. CONFIGURAÇÃO DA PÁGINA (deve ser a primeira chamada st.* do script)
# ==============================================================================
st.set_page_config(
    page_title="Diversificação das Exportações Brasileiras",
    page_icon="🌎",
    layout="wide",
)

# ==============================================================================
# 1. UTILITÁRIOS GERAIS
# ==============================================================================


def normalize_text(s) -> str:
    """Remove acentos, baixa a caixa e troca não-alfanuméricos por underscore."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@st.cache_data(show_spinner=False)
def read_any_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Lê csv, xlsx/xls, parquet ou json a partir dos bytes de um arquivo enviado."""
    name = file_name.lower()
    raw = file_bytes

    if name.endswith(".csv") or name.endswith(".txt"):
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8", low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding="latin-1", low_memory=False)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))

    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))

    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            for key in ["data", "dataset", "results"]:
                if key in data and isinstance(data[key], list):
                    return pd.json_normalize(data[key])
            return pd.json_normalize(data)
        return pd.json_normalize(data)

    raise ValueError(f"Formato de arquivo não suportado: {file_name}")


def find_column(df: pd.DataFrame, keywords: list, exclude: list | None = None):
    """Encontra a primeira coluna cujo nome normalizado contenha algum dos keywords."""
    exclude = exclude or []
    norm_map = {c: normalize_text(c) for c in df.columns}
    for col, norm in norm_map.items():
        if any(ex in norm for ex in exclude):
            continue
        for kw in keywords:
            if kw in norm:
                return col
    return None


def format_pct(x, decimals=1):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:,.{decimals}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def format_usd(x):
    if x is None or pd.isna(x):
        return "—"
    absx = abs(x)
    sign = "-" if x < 0 else ""
    if absx >= 1e9:
        return f"{sign}US$ {absx/1e9:,.2f} bi"
    if absx >= 1e6:
        return f"{sign}US$ {absx/1e6:,.2f} mi"
    if absx >= 1e3:
        return f"{sign}US$ {absx/1e3:,.1f} mil"
    return f"{sign}US$ {absx:,.0f}"


def format_num(x, decimals=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


# ==============================================================================
# 2. PADRONIZAÇÃO COMEXSTAT (exportações do Brasil — nacional e por UF)
# ==============================================================================

COMEXSTAT_FIELD_KEYWORDS = {
    "ano": ["ano", "year"],
    "pais": ["pais", "country", "parceiro"],
    "sh6_cod": ["codigo_sh6", "cod_sh6", "sh6_cod", "codigosh6"],
    "sh6_desc": ["descricao_sh6", "desc_sh6", "sh6_desc"],
    "cgce2_cod": ["codigo_cgce_nivel_2", "cgce_nivel_2_cod", "cgce2_cod"],
    "cgce2_desc": ["descricao_cgce_nivel_2", "cgce_nivel_2_desc", "cgce2_desc"],
    "cgce1_cod": ["codigo_cgce_nivel_1", "cgce_nivel_1_cod", "cgce1_cod"],
    "cgce1_desc": ["descricao_cgce_nivel_1", "cgce_nivel_1_desc", "cgce1_desc"],
    "cuci_cod": ["codigo_cuci_grupo", "cuci_grupo_cod", "cuci_cod"],
    "cuci_desc": ["descricao_cuci_grupo", "cuci_grupo_desc", "cuci_desc"],
    "isic_div_cod": ["codigo_isic_divisao", "isic_divisao_cod"],
    "isic_div_desc": ["descricao_isic_divisao", "isic_divisao_desc"],
    "isic_sec_cod": ["codigo_isic_secao", "isic_secao_cod"],
    "isic_sec_desc": ["descricao_isic_secao", "isic_secao_desc"],
    "valor_fob": ["valor_us_fob", "valor_fob", "fob", "valor_us"],
    "uf": ["sigla_uf", "uf", "estado"],
}

# Ordem em que os campos devem ser procurados: campos mais específicos
# (ex.: "cgce_nivel_2") precisam ser resolvidos antes de campos mais genéricos
# para não haver conflito de keywords entre colunas parecidas.
_COMEXSTAT_FIELD_ORDER = [
    "ano", "pais", "sh6_cod", "sh6_desc",
    "cgce2_cod", "cgce2_desc", "cgce1_cod", "cgce1_desc",
    "cuci_cod", "cuci_desc",
    "isic_div_cod", "isic_div_desc", "isic_sec_cod", "isic_sec_desc",
    "valor_fob", "uf",
]


def standardize_comexstat(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia as colunas do arquivo Comexstat para nomes padronizados internos."""
    df = df.copy()
    rename = {}
    used_cols = set()
    for std_name in _COMEXSTAT_FIELD_ORDER:
        keywords = COMEXSTAT_FIELD_KEYWORDS[std_name]
        col = None
        for c in df.columns:
            if c in used_cols:
                continue
            norm = normalize_text(c)
            if any(kw in norm for kw in keywords):
                col = c
                break
        if col:
            rename[col] = std_name
            used_cols.add(col)
    df = df.rename(columns=rename)

    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")

    if "valor_fob" in df.columns:
        if df["valor_fob"].dtype == object:
            cleaned = (
                df["valor_fob"].astype(str)
                .str.replace(r"[^\d,.\-]", "", regex=True)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df["valor_fob"] = pd.to_numeric(cleaned, errors="coerce")
        else:
            df["valor_fob"] = pd.to_numeric(df["valor_fob"], errors="coerce")

    if "sh6_cod" in df.columns:
        extracted = df["sh6_cod"].astype(str).str.extract(r"(\d+)")[0]
        df["sh6_cod"] = extracted.fillna(df["sh6_cod"].astype(str)).astype(str).str.zfill(6)

    if "uf" in df.columns:
        df["uf"] = df["uf"].astype(str).str.strip().str.upper()

    return df


def available_columns_report(df: pd.DataFrame):
    expected = list(COMEXSTAT_FIELD_KEYWORDS.keys())
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    return present, missing


# ==============================================================================
# 3. PADRONIZAÇÃO COMTRADE (importações mundiais e do Brasil, nível SH6)
# ==============================================================================


def guess_comtrade_columns(df: pd.DataFrame) -> dict:
    """Tenta adivinhar as colunas relevantes de um extrato do UN Comtrade."""
    return {
        "year": find_column(df, ["refyear", "ano", "year", "period"]),
        "partner": find_column(df, ["partnerdesc", "partner_desc", "partner", "parceiro", "reporterdesc"]),
        "sh6": find_column(df, ["cmdcode", "sh6_cod", "sh6", "codigo_sh6", "commoditycode"]),
        "sh6_desc": find_column(df, ["cmddesc", "sh6_desc", "descricao_sh6", "commoditydesc", "description"]),
        "value": find_column(df, ["primaryvalue", "tradevalue", "value", "valor", "fobvalue"]),
    }


# Colunas padrão de um extrato oficial do UN Comtrade (bulk download / API).
# Quando o arquivo enviado contém esse layout, a configuração é feita
# automaticamente, sem necessidade de mapeamento manual de colunas.
COMTRADE_STANDARD_COLUMNS = [
    "typeCode", "freqCode", "refPeriodId", "refYear", "refMonth", "period",
    "reporterCode", "reporterISO", "reporterDesc", "flowCode", "flowDesc",
    "partnerCode", "partnerISO", "partnerDesc", "partner2Code", "partner2ISO", "partner2Desc",
    "classificationCode", "classificationSearchCode", "isOriginalClassification",
    "cmdCode", "cmdDesc", "aggrLevel", "isLeaf", "customsCode", "customsDesc",
    "mosCode", "motCode", "motDesc", "qtyUnitCode", "qtyUnitAbbr", "qty",
    "isQtyEstimated", "altQtyUnitCode", "altQtyUnitAbbr", "altQty", "isAltQtyEstimated",
    "netWgt", "isNetWgtEstimated", "grossWgt", "isGrossWgtEstimated",
    "cifvalue", "fobvalue", "primaryValue", "legacyEstimationFlag", "isReported", "isAggregate",
]

# Colunas mínimas exigidas para considerar o arquivo como "layout padrão Comtrade"
_COMTRADE_REQUIRED_STD_COLS = {"refYear", "cmdCode", "partnerCode", "partnerDesc", "flowCode"}
_COMTRADE_VALUE_STD_COLS = {"primaryValue", "fobvalue", "cifvalue"}


def is_standard_comtrade_format(df: pd.DataFrame) -> bool:
    """Verifica se o arquivo segue o layout padrão de colunas do UN Comtrade."""
    cols = set(df.columns)
    return _COMTRADE_REQUIRED_STD_COLS.issubset(cols) and bool(_COMTRADE_VALUE_STD_COLS.intersection(cols))


def _norm_code_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().str.lstrip("0").replace({"": "0"})


def filter_comtrade_totals(df: pd.DataFrame):
    """
    Reduz um extrato padrão do Comtrade às linhas de TOTAL, evitando dupla
    contagem quando o arquivo já vem detalhado por fluxo, classificação,
    segundo parceiro, modo de transporte/fornecimento ou procedimento
    aduaneiro. Retorna (dataframe_filtrado, lista_de_mensagens_explicativas).
    """
    d = df.copy()
    msgs = []

    # Fluxo = Importação (flowCode 'M')
    if "flowCode" in d.columns:
        before = len(d)
        mask = d["flowCode"].astype(str).str.upper().isin(["M", "MIP"])
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `flowCode = M` (Importação): {before:,} → {len(d):,} linhas.")
    elif "flowDesc" in d.columns:
        before = len(d)
        mask = d["flowDesc"].astype(str).str.contains("import", case=False, na=False)
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `flowDesc` contendo 'Import': {before:,} → {len(d):,} linhas.")

    # Nível SH6 (aggrLevel == 6; fallback: cmdCode com 6 dígitos)
    if "aggrLevel" in d.columns:
        before = len(d)
        mask = pd.to_numeric(d["aggrLevel"], errors="coerce") == 6
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `aggrLevel = 6` (nível SH6): {before:,} → {len(d):,} linhas.")
    elif "cmdCode" in d.columns:
        before = len(d)
        mask = d["cmdCode"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.len() == 6
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `cmdCode` com 6 dígitos (nível SH6): {before:,} → {len(d):,} linhas.")

    # Classificação: mantém apenas a mais frequente (evita duplicidade entre vintages, ex. H4/H5/H6)
    if "classificationCode" in d.columns and d["classificationCode"].nunique() > 1:
        top_class = d["classificationCode"].value_counts().idxmax()
        before = len(d)
        d = d[d["classificationCode"] == top_class]
        msgs.append(f"✓ Múltiplas classificações encontradas — mantida `{top_class}` (mais frequente): {before:,} → {len(d):,} linhas.")

    # Segundo parceiro (reexportação): mantém apenas partner2Code == 0 (não aplicável)
    if "partner2Code" in d.columns and d["partner2Code"].nunique() > 1:
        before = len(d)
        mask = _norm_code_series(d["partner2Code"]).isin(["0"])
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `partner2Code = 0` (sem segundo parceiro): {before:,} → {len(d):,} linhas.")

    # Modo de transporte / modo de fornecimento: mantém o TOTAL (código 0), se existir
    for col, label in [("motCode", "modo de transporte"), ("mosCode", "modo de fornecimento")]:
        if col in d.columns and d[col].nunique() > 1:
            before = len(d)
            mask = _norm_code_series(d[col]).isin(["0"])
            if mask.any():
                d = d[mask]
                msgs.append(f"✓ Filtrado `{col} = 0` (total de {label}): {before:,} → {len(d):,} linhas.")

    # Procedimento aduaneiro: mantém o total (customsCode 'C00'), se existir
    if "customsCode" in d.columns and d["customsCode"].nunique() > 1:
        before = len(d)
        mask = d["customsCode"].astype(str).str.upper().isin(["C00"])
        if mask.any():
            d = d[mask]
            msgs.append(f"✓ Filtrado `customsCode = C00` (total de procedimentos aduaneiros): {before:,} → {len(d):,} linhas.")

    return d, msgs


def auto_configure_comtrade(df: pd.DataFrame) -> dict:
    """
    Detecta e configura automaticamente um extrato padrão do UN Comtrade
    (colunas oficiais da API/bulk download), identificando o total mundial
    e o total do Brasil a partir dos códigos oficiais de parceiro
    (partnerCode/partnerISO/partnerDesc) — sem necessidade de mapeamento
    manual de colunas.
    """
    result = {
        "ok": False,
        "messages": [],
        "year_col": None, "sh6_col": None, "sh6desc_col": None, "value_col": None,
        "partner_col": None, "world_values": [], "brazil_values": [],
        "filtered_df": None,
    }

    if not is_standard_comtrade_format(df):
        result["messages"].append(
            "O arquivo não segue o layout padrão de colunas do UN Comtrade "
            "(refYear, cmdCode, partnerCode, partnerDesc, flowCode, primaryValue/fobvalue)."
        )
        return result

    filtered, filter_msgs = filter_comtrade_totals(df)
    result["messages"].extend(filter_msgs)

    result["year_col"] = "refYear"
    result["sh6_col"] = "cmdCode"
    result["sh6desc_col"] = "cmdDesc" if "cmdDesc" in filtered.columns else None
    result["value_col"] = (
        "primaryValue" if "primaryValue" in filtered.columns
        else "fobvalue" if "fobvalue" in filtered.columns
        else "cifvalue"
    )
    result["partner_col"] = "partnerDesc"

    # Identificação do Mundo e do Brasil pelos códigos oficiais do Comtrade
    # (partnerCode 0 = World / 76 = Brazil; ISO 'WLD' / 'BRA'; descrição 'World' / 'Brazil')
    idx = filtered.index
    world_mask = pd.Series(False, index=idx)
    brazil_mask = pd.Series(False, index=idx)

    if "partnerCode" in filtered.columns:
        pcode = _norm_code_series(filtered["partnerCode"])
        world_mask |= pcode.isin(["0"])
        brazil_mask |= pcode.isin(["76"])
    if "partnerISO" in filtered.columns:
        piso = filtered["partnerISO"].astype(str).str.strip().str.lower()
        world_mask |= piso.isin(["wld", "w00"])
        brazil_mask |= piso.isin(["bra"])
    if "partnerDesc" in filtered.columns:
        pdesc = filtered["partnerDesc"].astype(str).str.strip().str.lower()
        world_mask |= pdesc.isin(["world"])
        brazil_mask |= pdesc.isin(["brazil", "brasil"])

    world_values = sorted(filtered.loc[world_mask, "partnerDesc"].dropna().astype(str).unique().tolist())
    brazil_values = sorted(filtered.loc[brazil_mask, "partnerDesc"].dropna().astype(str).unique().tolist())

    result["world_values"] = world_values
    result["brazil_values"] = brazil_values
    result["filtered_df"] = filtered

    if not world_values:
        result["messages"].append("⚠️ Não foi possível identificar linhas de **total mundial** (partnerCode = 0 / 'World').")
    if not brazil_values:
        result["messages"].append("⚠️ Não foi possível identificar linhas do **Brasil** (partnerCode = 76 / 'Brazil').")

    result["ok"] = bool(world_values) and bool(brazil_values)
    return result


def standardize_comtrade(
    df: pd.DataFrame,
    year_col: str,
    partner_col: str,
    sh6_col: str,
    sh6desc_col,
    value_col: str,
    world_values: list,
    brazil_values: list,
) -> pd.DataFrame:
    """
    Constrói uma tabela "tidy" (longa) com colunas: ano, sh6, sh6_desc,
    fluxo ('Mundo'/'Brasil'), valor — representando as IMPORTAÇÕES de cada
    produto SH6 pelo Mundo e pelo Brasil (usadas para estimar o mercado
    mundial e a posição competitiva do Brasil nesse mercado).
    """
    d = df.copy()
    cols = {year_col: "ano", partner_col: "partner_raw", sh6_col: "sh6", value_col: "valor"}
    if sh6desc_col:
        cols[sh6desc_col] = "sh6_desc"
    d = d.rename(columns=cols)

    d["ano"] = pd.to_numeric(d["ano"], errors="coerce").astype("Int64")
    extracted = d["sh6"].astype(str).str.extract(r"(\d+)")[0]
    d["sh6"] = extracted.fillna(d["sh6"].astype(str)).astype(str).str.zfill(6)
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce")

    d["fluxo"] = np.where(
        d["partner_raw"].astype(str).isin([str(v) for v in world_values]), "Mundo",
        np.where(d["partner_raw"].astype(str).isin([str(v) for v in brazil_values]), "Brasil", None),
    )
    d = d[d["fluxo"].notna()].copy()

    if "sh6_desc" not in d.columns:
        d["sh6_desc"] = d["sh6"]

    agg = d.groupby(["ano", "sh6", "sh6_desc", "fluxo"], as_index=False)["valor"].sum()
    return agg


# ==============================================================================
# 4. CÁLCULOS: CAGR, RCA, PARTICIPAÇÃO DE MERCADO E QUADRANTES
# ==============================================================================


def cagr(v_start, v_end, n_years: int) -> float:
    if v_start is None or v_end is None or n_years is None or n_years <= 0:
        return np.nan
    if pd.isna(v_start) or pd.isna(v_end):
        return np.nan
    if v_start <= 0 or v_end <= 0:
        return np.nan
    return (v_end / v_start) ** (1 / n_years) - 1


QUADRANT_ORDER = [
    "Estrela (ganho de espaço, mercado cresce)",
    "Oportunidade perdida (perda de espaço, mercado cresce)",
    "Resistência (ganho de espaço, mercado cai)",
    "Ameaça (perda de espaço, mercado cai)",
]

# Categorias usadas quando o Comtrade traz apenas 1 ano: não é possível calcular
# variação de RCA nem CAGR mundial (exigem 2 pontos no tempo), mas o RCA em
# nível (Balassa) para o ano disponível continua sendo calculado normalmente.
SINGLE_YEAR_QUADRANTS = [
    "Vantagem Comparativa (RCA ≥ 1)",
    "Sem Vantagem Comparativa (RCA < 1)",
]

QUADRANT_SHORT = {
    "Estrela (ganho de espaço, mercado cresce)": "Estrela",
    "Oportunidade perdida (perda de espaço, mercado cresce)": "Oportunidade perdida",
    "Resistência (ganho de espaço, mercado cai)": "Resistência",
    "Ameaça (perda de espaço, mercado cai)": "Ameaça",
    "Vantagem Comparativa (RCA ≥ 1)": "Vantagem Comparativa",
    "Sem Vantagem Comparativa (RCA < 1)": "Sem Vantagem Comparativa",
    "Sem dados suficientes": "Sem dados",
}

QUADRANT_COLORS = {
    "Estrela (ganho de espaço, mercado cresce)": "#1a9850",
    "Oportunidade perdida (perda de espaço, mercado cresce)": "#f5a623",
    "Resistência (ganho de espaço, mercado cai)": "#4a90d9",
    "Ameaça (perda de espaço, mercado cai)": "#d73027",
    "Vantagem Comparativa (RCA ≥ 1)": "#1a9850",
    "Sem Vantagem Comparativa (RCA < 1)": "#d73027",
    "Sem dados suficientes": "#bdbdbd",
}

# Descrição curta usada nos cartões de quadrante do dashboard (estilo "CNI")
QUADRANT_DESCRIPTIONS = {
    "Estrela (ganho de espaço, mercado cresce)": [
        "O Brasil ganha espaço competitivo (↑ RCA) no produto",
        "O mercado mundial do produto está em expansão (CAGR mundial > 0)",
    ],
    "Oportunidade perdida (perda de espaço, mercado cresce)": [
        "O Brasil perde espaço competitivo (↓ RCA) no produto",
        "O mercado mundial do produto está em expansão (CAGR mundial > 0)",
    ],
    "Resistência (ganho de espaço, mercado cai)": [
        "O Brasil ganha espaço competitivo (↑ RCA) no produto",
        "O mercado mundial do produto está em retração (CAGR mundial < 0)",
    ],
    "Ameaça (perda de espaço, mercado cai)": [
        "O Brasil perde espaço competitivo (↓ RCA) no produto",
        "O mercado mundial do produto está em retração (CAGR mundial < 0)",
    ],
    "Vantagem Comparativa (RCA ≥ 1)": [
        "O Brasil é mais especializado neste produto do que na pauta média mundial (RCA ≥ 1)",
        "Apenas 1 ano de Comtrade disponível — sem tendência (CAGR/variação de RCA)",
    ],
    "Sem Vantagem Comparativa (RCA < 1)": [
        "O Brasil é menos especializado neste produto do que na pauta média mundial (RCA < 1)",
        "Apenas 1 ano de Comtrade disponível — sem tendência (CAGR/variação de RCA)",
    ],
}


def classify_quadrant(delta_rca: float, cagr_world: float) -> str:
    if pd.isna(delta_rca) or pd.isna(cagr_world):
        return "Sem dados suficientes"
    ganho_espaco = delta_rca > 0
    mercado_cresce = cagr_world > 0
    if ganho_espaco and mercado_cresce:
        return QUADRANT_ORDER[0]
    if (not ganho_espaco) and mercado_cresce:
        return QUADRANT_ORDER[1]
    if ganho_espaco and (not mercado_cresce):
        return QUADRANT_ORDER[2]
    return QUADRANT_ORDER[3]


def classify_quadrant_single_year(rca: float) -> str:
    """Classificação simplificada usada quando só há 1 ano de dados Comtrade:
    não há como medir tendência (variação de RCA / CAGR mundial), então o
    produto é classificado apenas pelo nível do RCA (Balassa) naquele ano."""
    if pd.isna(rca):
        return "Sem dados suficientes"
    return SINGLE_YEAR_QUADRANTS[0] if rca >= 1 else SINGLE_YEAR_QUADRANTS[1]


def build_world_brazil_wide(comtrade_tidy: pd.DataFrame) -> pd.DataFrame:
    """Pivota a base tidy do Comtrade em colunas Mundo/Brasil por ano e sh6."""
    wide = comtrade_tidy.pivot_table(
        index=["sh6", "sh6_desc", "ano"], columns="fluxo", values="valor", aggfunc="sum"
    ).reset_index()
    for c in ["Mundo", "Brasil"]:
        if c not in wide.columns:
            wide[c] = np.nan
    wide["participacao_brasil"] = wide["Brasil"] / wide["Mundo"]
    return wide


@st.cache_data(show_spinner=False)
def compute_product_metrics(comtrade_tidy: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Calcula, para cada SH6, as métricas de competitividade a partir do Comtrade.

    - Quando `start_year != end_year` (2 anos disponíveis): valores em início/fim
      de período, CAGR mundial e do Brasil, participação de mercado, RCA em
      início/fim, variação de RCA/participação e o quadrante de competitividade
      (matriz RCA × crescimento do mercado mundial).
    - Quando `start_year == end_year` (apenas 1 ano de Comtrade disponível): o
      RCA (Balassa) é calculado normalmente para aquele único ano — o que NÃO
      depende de série temporal —, mas CAGR mundial/Brasil e a variação de RCA
      ficam indisponíveis (NaN) e o produto é classificado apenas pelo nível
      do RCA (Vantagem Comparativa / Sem Vantagem Comparativa).
    """
    single_year = start_year == end_year
    years_needed = [start_year] if single_year else [start_year, end_year]

    wide = build_world_brazil_wide(comtrade_tidy)
    wide = wide[wide["ano"].isin(years_needed)]
    n_years = end_year - start_year

    totals = wide.groupby("ano")[["Mundo", "Brasil"]].sum()

    def _safe_rca(braz, braz_tot, world, world_tot):
        vals = [braz, braz_tot, world, world_tot]
        if any(pd.isna(v) for v in vals) or any(v == 0 for v in vals):
            return np.nan
        return (braz / braz_tot) / (world / world_tot)

    rows = []
    for sh6, g in wide.groupby("sh6"):
        sh6_desc = g["sh6_desc"].iloc[0]
        g = g.set_index("ano")

        world_start = g["Mundo"].get(start_year, np.nan)
        braz_start = g["Brasil"].get(start_year, np.nan)
        world_tot_start = totals["Mundo"].get(start_year, np.nan)
        braz_tot_start = totals["Brasil"].get(start_year, np.nan)

        share_start = (braz_start / world_start) if (pd.notna(world_start) and world_start > 0) else np.nan
        rca_start = _safe_rca(braz_start, braz_tot_start, world_start, world_tot_start)

        if single_year:
            # Sem segundo ponto no tempo: RCA de nível apenas, sem variação/CAGR.
            world_end = world_start
            braz_end = braz_start
            share_end = share_start
            rca_end = rca_start
            cagr_world = np.nan
            cagr_brasil = np.nan
            delta_rca = np.nan
            delta_share = np.nan
            quadrant = classify_quadrant_single_year(rca_start)
        else:
            world_end = g["Mundo"].get(end_year, np.nan)
            braz_end = g["Brasil"].get(end_year, np.nan)
            world_tot_end = totals["Mundo"].get(end_year, np.nan)
            braz_tot_end = totals["Brasil"].get(end_year, np.nan)

            share_end = (braz_end / world_end) if (pd.notna(world_end) and world_end > 0) else np.nan
            rca_end = _safe_rca(braz_end, braz_tot_end, world_end, world_tot_end)

            cagr_world = cagr(world_start, world_end, n_years)
            cagr_brasil = cagr(braz_start, braz_end, n_years)

            delta_rca = (rca_end - rca_start) if (pd.notna(rca_end) and pd.notna(rca_start)) else np.nan
            delta_share = (share_end - share_start) if (pd.notna(share_end) and pd.notna(share_start)) else np.nan

            quadrant = classify_quadrant(delta_rca, cagr_world)

        rows.append({
            "sh6": sh6,
            "sh6_desc": sh6_desc,
            "mundo_inicio": world_start,
            "mundo_fim": world_end,
            "brasil_inicio": braz_start,
            "brasil_fim": braz_end,
            "participacao_inicio": share_start,
            "participacao_fim": share_end,
            "var_participacao": delta_share,
            "rca_inicio": rca_start,
            "rca_fim": rca_end,
            "var_rca": delta_rca,
            "cagr_mundo": cagr_world,
            "cagr_brasil": cagr_brasil,
            "quadrante": quadrant,
        })

    result = pd.DataFrame(rows)
    result.attrs["single_year_mode"] = single_year
    return result


def compute_summary_kpis(product_metrics: pd.DataFrame, comtrade_tidy: pd.DataFrame,
                          start_year: int, end_year: int) -> dict:
    single_year_mode = start_year == end_year
    n_years = end_year - start_year
    totals = comtrade_tidy.groupby(["ano", "fluxo"])["valor"].sum().unstack("fluxo")

    world_start = totals.loc[start_year, "Mundo"] if start_year in totals.index and "Mundo" in totals.columns else np.nan
    world_end = totals.loc[end_year, "Mundo"] if end_year in totals.index and "Mundo" in totals.columns else np.nan
    braz_start = totals.loc[start_year, "Brasil"] if start_year in totals.index and "Brasil" in totals.columns else np.nan
    braz_end = totals.loc[end_year, "Brasil"] if end_year in totals.index and "Brasil" in totals.columns else np.nan

    cagr_world = cagr(world_start, world_end, n_years)
    cagr_brasil = cagr(braz_start, braz_end, n_years)

    brasil_cresceu_mais = None
    if pd.notna(cagr_world) and pd.notna(cagr_brasil):
        brasil_cresceu_mais = bool(cagr_brasil > cagr_world)

    quad_counts = product_metrics["quadrante"].value_counts().to_dict()

    braz_shares_end = product_metrics["brasil_fim"].dropna()
    if braz_shares_end.sum() > 0:
        shares = braz_shares_end / braz_shares_end.sum()
        hhi = float((shares ** 2).sum() * 10000)
    else:
        hhi = np.nan

    n_produtos_exportados_fim = int((product_metrics["brasil_fim"].fillna(0) > 0).sum())
    n_produtos_exportados_inicio = int((product_metrics["brasil_inicio"].fillna(0) > 0).sum())
    novos_produtos = int(((product_metrics["brasil_inicio"].fillna(0) == 0) &
                           (product_metrics["brasil_fim"].fillna(0) > 0)).sum())
    produtos_perdidos = int(((product_metrics["brasil_inicio"].fillna(0) > 0) &
                              (product_metrics["brasil_fim"].fillna(0) == 0)).sum())

    rca_col = "rca_fim" if "rca_fim" in product_metrics.columns else "rca_inicio"
    rca_valid = product_metrics[rca_col].dropna()
    rca_medio = float(rca_valid.mean()) if not rca_valid.empty else np.nan
    n_com_vantagem = int((rca_valid >= 1).sum())

    return {
        "single_year_mode": single_year_mode,
        "cagr_mundo": cagr_world,
        "cagr_brasil": cagr_brasil,
        "brasil_cresceu_mais_que_mundo": brasil_cresceu_mais,
        "quadrant_counts": quad_counts,
        "hhi": hhi,
        "n_produtos_inicio": n_produtos_exportados_inicio,
        "n_produtos_fim": n_produtos_exportados_fim,
        "produtos_novos": novos_produtos,
        "produtos_perdidos": produtos_perdidos,
        "world_start": world_start, "world_end": world_end,
        "braz_start": braz_start, "braz_end": braz_end,
        "rca_medio": rca_medio,
        "n_produtos_com_vantagem_comparativa": n_com_vantagem,
        "n_produtos_com_rca": int(rca_valid.shape[0]),
    }


def compute_regional_rca(comexstat_uf: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    RCA Regional (dentro do Brasil): mede a especialização de um Estado em um
    produto SH6 frente ao padrão nacional de exportação, sem depender de
    dados mundiais:

        RCA_regional = (Exp_estado,produto / Exp_estado,total)
                      / (Exp_brasil,produto / Exp_brasil,total)

    Valores > 1 indicam que o Estado é relativamente mais especializado
    naquele produto do que o Brasil como um todo.
    """
    base = comexstat_uf[comexstat_uf["ano"] == year]
    if base.empty or "uf" not in base.columns:
        return pd.DataFrame()

    exp_estado_prod = base.groupby(["uf", "sh6_cod"])["valor_fob"].sum()
    exp_estado_tot = base.groupby("uf")["valor_fob"].sum()
    exp_brasil_prod = base.groupby("sh6_cod")["valor_fob"].sum()
    exp_brasil_tot = base["valor_fob"].sum()

    df = exp_estado_prod.reset_index().rename(columns={"valor_fob": "exp_estado_produto"})
    df["exp_estado_total"] = df["uf"].map(exp_estado_tot)
    df["exp_brasil_produto"] = df["sh6_cod"].map(exp_brasil_prod)
    df["exp_brasil_total"] = exp_brasil_tot

    df["rca_regional"] = (df["exp_estado_produto"] / df["exp_estado_total"]) / (
        df["exp_brasil_produto"] / df["exp_brasil_total"]
    )
    return df


# ==============================================================================
# 5. COMPONENTES VISUAIS (estilo "Radar CNI" — cartões, estatísticas, badges)
# ==============================================================================


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        /* ---- Big stats row (estilo "1.646 produtos monitorados") ---- */
        .stat-row { display:flex; gap:40px; flex-wrap:wrap; margin: 4px 0 22px 0; }
        .stat-item .stat-value { font-size:2.1rem; font-weight:800; line-height:1.15; color:#111827; }
        .stat-item .stat-label { font-size:0.82rem; color:#6b7280; margin-top:2px; }

        /* ---- Cartões de quadrante ---- */
        .quadrant-grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin: 6px 0 18px 0; }
        @media (max-width: 900px) { .quadrant-grid { grid-template-columns: 1fr; } }
        .quadrant-card { border-radius:12px; padding:20px 24px; }
        .quadrant-card .qc-title { font-weight:800; font-size:1.05rem; display:flex; align-items:center; gap:8px; color:#111827;}
        .quadrant-card .qc-dot { width:10px; height:10px; border-radius:50%; display:inline-block; flex-shrink:0; }
        .quadrant-card .qc-desc { font-size:0.80rem; color:#4b5563; margin:10px 0 2px 0; line-height:1.5; }
        .quadrant-card .qc-bottom { display:flex; justify-content:space-between; align-items:flex-end; margin-top:20px; }
        .quadrant-card .qc-count { font-size:2.0rem; font-weight:800; color:#111827; }
        .quadrant-card .qc-count-label { font-size:0.76rem; color:#6b7280; }
        .quadrant-card .qc-value { font-size:1.35rem; font-weight:800; color:#111827; text-align:right; }
        .quadrant-card .qc-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.70rem;
            font-weight:700; margin-top:6px; }

        /* ---- Barra de distribuição por quadrante ---- */
        .dist-bar-label { font-size:0.70rem; letter-spacing:0.06em; color:#9ca3af; font-weight:700;
            text-transform:uppercase; margin-bottom:6px; }
        .dist-bar { display:flex; height:9px; border-radius:6px; overflow:hidden; margin: 0 0 22px 0; background:#eee; }

        /* ---- Badge de quadrante (usado em tabelas/listas) ---- */
        .quad-badge { display:inline-flex; align-items:center; gap:6px; font-size:0.78rem; font-weight:700;
            white-space:nowrap; }
        .quad-badge .qc-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

        /* ---- Tabela estilo "Visão Estratégica" ---- */
        .cni-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
        .cni-table th { text-align:left; font-size:0.72rem; color:#6b7280; text-transform:uppercase;
            letter-spacing:0.03em; padding:8px 10px; border-bottom:2px solid #e5e7eb; }
        .cni-table td { padding:9px 10px; border-bottom:1px solid #f0f0f0; vertical-align:middle; }
        .cni-table tr:hover td { background:#fafafa; }
        .cni-table .sector-name { font-weight:700; color:#111827; }
        .cni-table .sector-sub { font-size:0.72rem; color:#9ca3af; }
        .cni-table .pct-cell { font-weight:700; }

        /* ---- Painel "produtos mais impactados" ---- */
        .impact-card { border-bottom:1px solid #f0f0f0; padding:12px 4px; }
        .impact-card .impact-name { font-weight:700; font-size:0.86rem; color:#111827; line-height:1.3; }
        .impact-card .impact-meta { font-size:0.72rem; color:#9ca3af; margin-top:2px; }
        .impact-card .impact-value { font-weight:800; font-size:0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stat_row(items: list[tuple[str, str]]):
    """items: lista de (valor_grande, rótulo)."""
    html = '<div class="stat-row">'
    for value, label in items:
        html += (
            '<div class="stat-item">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_quad_badge(quadrant: str) -> str:
    color = QUADRANT_COLORS.get(quadrant, "#bdbdbd")
    label = QUADRANT_SHORT.get(quadrant, quadrant)
    return (
        f'<span class="quad-badge" style="color:{color};">'
        f'<span class="qc-dot" style="background:{color};"></span>{label}</span>'
    )


def render_quadrant_cards(quadrant_list: list, counts: dict, values: dict, total_value: float,
                           descriptions: dict):
    """Grade 2x2 (ou Nx1) de cartões de quadrante, no estilo do Radar CNI."""
    html = '<div class="quadrant-grid">'
    for q in quadrant_list:
        color = QUADRANT_COLORS.get(q, "#bdbdbd")
        bg = _hex_to_rgba(color, 0.08)
        pill_bg = _hex_to_rgba(color, 0.16)
        n = int(counts.get(q, 0))
        val = values.get(q, 0.0)
        pct = (val / total_value) if total_value else np.nan
        desc_lines = descriptions.get(q, [])
        desc_html = "".join(f"<div>{d}</div>" for d in desc_lines)
        html += (
            f'<div class="quadrant-card" style="background:{bg};">'
            f'<div class="qc-title"><span class="qc-dot" style="background:{color};"></span>{QUADRANT_SHORT.get(q, q)}</div>'
            f'<div class="qc-desc">{desc_html}</div>'
            '<div class="qc-bottom">'
            f'<div><div class="qc-count">{n}</div><div class="qc-count-label">produtos</div></div>'
            '<div>'
            f'<div class="qc-value">{format_usd(val)}</div>'
            f'<div class="qc-pill" style="background:{pill_bg}; color:{color};">{format_pct(pct)} do mercado</div>'
            "</div>"
            "</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_distribution_bar(quadrant_list: list, values: dict, total_value: float, label="DISTRIBUIÇÃO DO MERCADO POR QUADRANTE"):
    if not total_value:
        return
    html = f'<div class="dist-bar-label">{label}</div><div class="dist-bar">'
    for q in quadrant_list:
        color = QUADRANT_COLORS.get(q, "#bdbdbd")
        val = values.get(q, 0.0)
        pct = (val / total_value * 100) if total_value else 0
        if pct <= 0:
            continue
        html += f'<div style="width:{pct:.3f}%; background:{color};" title="{QUADRANT_SHORT.get(q, q)}: {pct:.1f}%"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_nav_cards(cards: list[dict], nav_key: str):
    """cards: [{'icon':'📋','title':'...', 'target': '📋 Tabela Completa (SH6)'}]"""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{card['icon']} {card['title']}**")
                if c2.button("→", key=f"navcard_{card['target']}", use_container_width=True):
                    st.session_state[nav_key] = card["target"]
                    st.rerun()


def build_isic_quadrant_table(comexstat: pd.DataFrame, product_metrics: pd.DataFrame,
                               cod_col: str, desc_col: str, year, quadrant_list: list) -> pd.DataFrame:
    """Monta a tabela 'Setor × % de mercado por quadrante' no estilo Visão Estratégica."""
    base = comexstat[comexstat["ano"] == year] if year is not None else comexstat
    merged = base.merge(product_metrics[["sh6", "quadrante"]], left_on="sh6_cod", right_on="sh6", how="left")
    merged["quadrante"] = merged["quadrante"].fillna("Sem dados suficientes")

    rows = []
    for (cod, desc), g in merged.groupby([cod_col, desc_col]):
        total = g["valor_fob"].sum()
        row = {"cod": cod, "desc": desc, "mercado_total": total, "n_produtos": g["sh6_cod"].nunique()}
        for q in quadrant_list:
            val = g.loc[g["quadrante"] == q, "valor_fob"].sum()
            row[q] = (val / total) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mercado_total", ascending=False)


def render_cni_style_table(df: pd.DataFrame, quadrant_list: list, height_rows: int = 12):
    """Renderiza a tabela setor × quadrante como HTML, com pontos coloridos e %."""
    if df.empty:
        st.info("Sem dados para exibir.")
        return
    html = '<table class="cni-table"><thead><tr>'
    html += '<th>Setor</th>'
    for q in quadrant_list:
        html += f'<th>{QUADRANT_SHORT.get(q, q)}</th>'
    html += '<th>Mercado total</th></tr></thead><tbody>'
    for _, r in df.head(height_rows).iterrows():
        html += "<tr>"
        html += (
            f'<td><div class="sector-name">{r["cod"]} — {r["desc"]}</div>'
            f'<div class="sector-sub">{int(r["n_produtos"])} produtos</div></td>'
        )
        for q in quadrant_list:
            color = QUADRANT_COLORS.get(q, "#bdbdbd")
            pct = r.get(q, np.nan)
            pct_str = format_pct(pct) if pd.notna(pct) else "—"
            html += (
                f'<td class="pct-cell" style="color:{color};">'
                f'<span class="qc-dot" style="background:{color}; width:8px;height:8px;'
                f'border-radius:50%; display:inline-block; margin-right:6px;"></span>{pct_str}</td>'
            )
        html += f'<td>{format_usd(r["mercado_total"])}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def render_impact_products(products_df: pd.DataFrame, value_col: str, value_label: str,
                            market_col: str, code_col: str = "sh6", desc_col: str = "sh6_desc",
                            n: int = 10, value_is_currency: bool = True):
    """Painel 'Produtos mais impactados' (lista de cartões), estilo CNI."""
    if products_df.empty:
        st.info("Sem produtos para exibir neste recorte.")
        return
    for _, r in products_df.head(n).iterrows():
        badge = render_quad_badge(r.get("quadrante", "Sem dados suficientes"))
        val = r[value_col]
        val_color = "#d73027" if pd.notna(val) and val < 0 else "#1a9850"
        val_str = format_usd(val) if value_is_currency else format_num(val, 2)
        st.markdown(
            f'<div class="impact-card">'
            f'<div class="impact-name">{r[desc_col]}</div>'
            f'<div class="impact-meta">SH6 {r[code_col]} · {badge}</div>'
            f'<div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:6px;">'
            f'<span class="impact-value" style="color:{val_color};">{value_label}: {val_str}</span>'
            f'<span class="impact-meta">Mercado: {format_usd(r[market_col])}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ==============================================================================
# 6. SIDEBAR — UPLOAD DE ARQUIVOS E NAVEGAÇÃO (comum a todas as páginas)
# ==============================================================================

inject_custom_css()

st.sidebar.title("🌎 Navegação")
NAV_KEY = "nav_page_radio"
_PAGE_OPTIONS = ["📊 Dashboard Resumo", "📋 Tabela Completa (SH6)", "📦 CUCI Grupo", "🗺️ Análise por Estado"]
if NAV_KEY not in st.session_state:
    st.session_state[NAV_KEY] = _PAGE_OPTIONS[0]
PAGE = st.sidebar.radio(
    "Selecione a página",
    _PAGE_OPTIONS,
    key=NAV_KEY,
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.header("📁 Dados de entrada")

st.sidebar.subheader("1. Comexstat — Exportações do Brasil")
st.sidebar.caption("Ano, País, SH6, CGCE, CUCI Grupo, ISIC Divisão/Seção, Valor US$ FOB")
comexstat_file = st.sidebar.file_uploader(
    "Arquivo Comexstat (csv, xlsx, parquet)", type=["csv", "xlsx", "xls", "parquet"], key="comexstat_upl"
)

st.sidebar.subheader("2. UN Comtrade — Importações (Mundo e Brasil)")
st.sidebar.caption("Necessário para calcular RCA, participação de mercado e quadrantes.")
comtrade_file = st.sidebar.file_uploader(
    "Arquivo Comtrade (csv, xlsx, parquet, json)",
    type=["csv", "xlsx", "xls", "parquet", "json"], key="comtrade_upl"
)

st.sidebar.subheader("3. Comexstat por Estado (opcional)")
st.sidebar.caption("Usado na página 'Análise por Estado'. Mesma estrutura + coluna de UF/Estado.")
comexstat_uf_file = st.sidebar.file_uploader(
    "Arquivo Comexstat por UF (csv, xlsx, parquet)",
    type=["csv", "xlsx", "xls", "parquet"], key="comexstat_uf_upl"
)

# ------------------------------------------------------------------------------
# Processamento Comexstat nacional
# ------------------------------------------------------------------------------
if comexstat_file is not None:
    if st.session_state.get("_comexstat_name") != comexstat_file.name:
        raw = read_any_file(comexstat_file.getvalue(), comexstat_file.name)
        st.session_state["comexstat"] = standardize_comexstat(raw)
        st.session_state["_comexstat_name"] = comexstat_file.name

# ------------------------------------------------------------------------------
# Processamento Comexstat por UF
# ------------------------------------------------------------------------------
if comexstat_uf_file is not None:
    if st.session_state.get("_comexstat_uf_name") != comexstat_uf_file.name:
        raw_uf = read_any_file(comexstat_uf_file.getvalue(), comexstat_uf_file.name)
        df_uf = standardize_comexstat(raw_uf)
        if "uf" not in df_uf.columns:
            st.sidebar.error("Não foi possível identificar a coluna de Estado/UF no arquivo enviado.")
        else:
            st.session_state["comexstat_uf"] = df_uf
            st.session_state["_comexstat_uf_name"] = comexstat_uf_file.name

# ------------------------------------------------------------------------------
# Processamento Comtrade — mapeamento de colunas assistido
# ------------------------------------------------------------------------------
if comtrade_file is not None:
    if st.session_state.get("_comtrade_raw_name") != comtrade_file.name:
        st.session_state["comtrade_raw"] = read_any_file(comtrade_file.getvalue(), comtrade_file.name)
        st.session_state["_comtrade_raw_name"] = comtrade_file.name
        st.session_state.pop("comtrade_tidy", None)
        st.session_state.pop("product_metrics", None)
        st.session_state.pop("kpis", None)
        st.session_state.pop("_comtrade_autocfg", None)
        st.session_state.pop("_comtrade_autocfg_for", None)
        st.session_state["_comtrade_manual_override"] = False

if "comtrade_raw" in st.session_state:
    raw = st.session_state["comtrade_raw"]

    # ---------------------------------------------------------------------
    # Tentativa de auto-configuração (layout padrão de colunas do UN Comtrade)
    # ---------------------------------------------------------------------
    if st.session_state.get("_comtrade_autocfg_for") != st.session_state.get("_comtrade_raw_name"):
        st.session_state["_comtrade_autocfg"] = auto_configure_comtrade(raw)
        st.session_state["_comtrade_autocfg_for"] = st.session_state.get("_comtrade_raw_name")

    autocfg = st.session_state["_comtrade_autocfg"]
    manual_override = st.session_state.get("_comtrade_manual_override", False)

    if autocfg["ok"] and not manual_override and "comtrade_tidy" not in st.session_state:
        # Processa automaticamente assim que o arquivo padrão é detectado — sem cliques.
        tidy = standardize_comtrade(
            autocfg["filtered_df"], autocfg["year_col"], autocfg["partner_col"],
            autocfg["sh6_col"], autocfg["sh6desc_col"], autocfg["value_col"],
            autocfg["world_values"], autocfg["brazil_values"],
        )
        if not tidy.empty:
            st.session_state["comtrade_tidy"] = tidy
            st.session_state.pop("product_metrics", None)
            st.session_state.pop("kpis", None)

    if autocfg["ok"] and not manual_override:
        with st.sidebar.expander("✅ Comtrade — configuração automática", expanded=False):
            st.caption("Layout padrão do UN Comtrade detectado. Configuração aplicada automaticamente:")
            st.markdown(
                f"- **Ano:** `{autocfg['year_col']}`\n"
                f"- **Produto SH6:** `{autocfg['sh6_col']}`"
                + (f" (`{autocfg['sh6desc_col']}` como descrição)" if autocfg["sh6desc_col"] else "")
                + f"\n- **Valor:** `{autocfg['value_col']}`\n"
                f"- **Fluxo:** Importação (`flowCode = M`)\n"
                f"- **Total Mundial identificado em `partnerDesc`:** {', '.join(autocfg['world_values']) or '—'}\n"
                f"- **Brasil identificado em `partnerDesc`:** {', '.join(autocfg['brazil_values']) or '—'}"
            )
            if autocfg["messages"]:
                st.caption("Filtros de totalização aplicados automaticamente:")
                for m in autocfg["messages"]:
                    st.caption(m)
            if "comtrade_tidy" in st.session_state:
                tidy = st.session_state["comtrade_tidy"]
                st.success(
                    f"Base processada: {tidy['sh6'].nunique()} produtos SH6, "
                    f"anos {int(tidy['ano'].min())}–{int(tidy['ano'].max())}."
                )
            if st.button("🔧 Ajustar manualmente"):
                st.session_state["_comtrade_manual_override"] = True
                st.rerun()
    else:
        # -------------------------------------------------------------
        # Fallback: mapeamento manual (layout não padrão, ou ajuste solicitado)
        # -------------------------------------------------------------
        with st.sidebar.expander("⚙️ Mapeamento de colunas (Comtrade)", expanded=True):
            if not autocfg["ok"]:
                for m in autocfg["messages"]:
                    st.caption(("⚠️ " if not m.startswith("⚠️") else "") + m)
                st.caption("Configure manualmente as colunas abaixo.")
            elif manual_override:
                st.caption("Configuração automática disponível — ajuste os campos abaixo se necessário.")
                if st.button("↩️ Voltar para configuração automática"):
                    st.session_state["_comtrade_manual_override"] = False
                    st.rerun()

            guesses = guess_comtrade_columns(raw)
            cols = list(raw.columns)

            def idx_or_0(val):
                return cols.index(val) if val in cols else 0

            default_year = autocfg["year_col"] or guesses["year"]
            default_sh6 = autocfg["sh6_col"] or guesses["sh6"]
            default_partner = autocfg["partner_col"] or guesses["partner"]
            default_sh6desc = autocfg["sh6desc_col"] or guesses["sh6_desc"]
            default_value = autocfg["value_col"] or guesses["value"]

            year_col = st.selectbox("Coluna de Ano", cols, index=idx_or_0(default_year))
            sh6_col = st.selectbox("Coluna de código SH6", cols, index=idx_or_0(default_sh6))
            partner_col = st.selectbox("Coluna de Parceiro/Reporter", cols, index=idx_or_0(default_partner))
            sh6desc_col = st.selectbox(
                "Coluna de descrição SH6 (opcional)", ["(nenhuma)"] + cols,
                index=(cols.index(default_sh6desc) + 1) if default_sh6desc in cols else 0,
            )
            value_col = st.selectbox("Coluna de Valor", cols, index=idx_or_0(default_value))

            unique_partners = sorted(raw[partner_col].dropna().astype(str).unique().tolist())
            world_values = st.multiselect(
                "Valor(es) = TOTAL MUNDIAL", unique_partners,
                default=autocfg["world_values"] or [p for p in unique_partners if "world" in p.lower() or "mundo" in p.lower()],
            )
            brazil_values = st.multiselect(
                "Valor(es) = BRASIL", unique_partners,
                default=autocfg["brazil_values"] or [p for p in unique_partners if "brazil" in p.lower() or "brasil" in p.lower()],
            )

            if st.button("Processar dados do Comtrade", type="primary"):
                if not world_values or not brazil_values:
                    st.error("Selecione ao menos um valor para Mundo e um para Brasil.")
                else:
                    tidy = standardize_comtrade(
                        raw, year_col, partner_col, sh6_col,
                        None if sh6desc_col == "(nenhuma)" else sh6desc_col,
                        value_col, world_values, brazil_values,
                    )
                    if tidy.empty:
                        st.error("Nenhum registro casou com os valores de Mundo/Brasil selecionados.")
                    else:
                        st.session_state["comtrade_tidy"] = tidy
                        st.session_state.pop("product_metrics", None)
                        st.session_state.pop("kpis", None)
                        st.success(
                            f"Base processada: {tidy['sh6'].nunique()} produtos SH6, "
                            f"anos {int(tidy['ano'].min())}–{int(tidy['ano'].max())}."
                        )

has_comtrade = "comtrade_tidy" in st.session_state
has_comexstat = "comexstat" in st.session_state
has_comexstat_uf = "comexstat_uf" in st.session_state

# ------------------------------------------------------------------------------
# Seleção de período e cálculo de métricas (comum às páginas)
# ------------------------------------------------------------------------------
if has_comtrade:
    tidy = st.session_state["comtrade_tidy"]
    anos_disponiveis = sorted(tidy["ano"].dropna().unique().astype(int).tolist())
    st.sidebar.divider()
    st.sidebar.subheader("📅 Período de análise")

    if len(anos_disponiveis) >= 2:
        start_year = st.sidebar.selectbox("Ano inicial", anos_disponiveis, index=0)
        end_years = [a for a in anos_disponiveis if a > start_year]
        end_year = st.sidebar.selectbox("Ano final", end_years, index=len(end_years) - 1) if end_years else None
    elif len(anos_disponiveis) == 1:
        # Apenas 1 ano no Comtrade: ainda assim calculamos o RCA (Balassa) de
        # nível para esse ano — apenas CAGR mundial e variação de RCA (que
        # exigem 2 pontos no tempo) ficam indisponíveis.
        start_year = end_year = anos_disponiveis[0]
        st.sidebar.info(
            f"Apenas o ano **{start_year}** disponível no Comtrade. O RCA será "
            "calculado normalmente para esse ano; CAGR mundial e variação de "
            "RCA (tendência) exigem pelo menos 2 anos."
        )
    else:
        start_year = end_year = None
        st.sidebar.warning("Nenhum ano válido encontrado na base Comtrade.")

    if start_year is not None and end_year is not None:
        cache_key = (start_year, end_year, id(tidy))
        if st.session_state.get("_metrics_cache_key") != cache_key:
            product_metrics = compute_product_metrics(tidy, start_year, end_year)
            kpis = compute_summary_kpis(product_metrics, tidy, start_year, end_year)
            st.session_state["product_metrics"] = product_metrics
            st.session_state["kpis"] = kpis
            st.session_state["_metrics_cache_key"] = cache_key
        st.session_state["start_year"] = start_year
        st.session_state["end_year"] = end_year
        st.session_state["single_year_mode"] = (start_year == end_year)

product_metrics = st.session_state.get("product_metrics")
kpis = st.session_state.get("kpis")
start_year = st.session_state.get("start_year")
end_year = st.session_state.get("end_year")


# ==============================================================================
# 6. PÁGINA 1 — DASHBOARD RESUMO
# ==============================================================================


def page_dashboard():
    st.title("🌎 Radar da Diversificação das Exportações Brasileiras")
    st.caption(
        "Combine dados de importação mundial/Brasil (UN Comtrade, nível SH6) com dados de "
        "exportação do Brasil (Comexstat) para identificar oportunidades e ameaças competitivas."
    )

    if not has_comexstat and not has_comtrade:
        st.info(
            "⬅️ Envie ao menos o arquivo **Comexstat** na barra lateral para começar. "
            "Envie também o **Comtrade** para desbloquear RCA, quadrantes e participação de mercado."
        )
        return

    single_year_mode = bool(kpis and kpis.get("single_year_mode"))
    active_quadrants = SINGLE_YEAR_QUADRANTS if single_year_mode else QUADRANT_ORDER

    # ---------------- Faixa de referência do período ----------------
    if start_year is not None and end_year is not None:
        periodo_label = f"Ano de referência: **{start_year}**" if single_year_mode else f"Período: **{start_year} → {end_year}**"
        st.caption(f"📅 {periodo_label} (Comtrade)")

    # ---------------- Stat row (grandes números, estilo CNI) ----------------
    n_sh6_monitorados = int(product_metrics["sh6"].nunique()) if product_metrics is not None else (
        int(st.session_state["comexstat"]["sh6_cod"].nunique()) if has_comexstat else 0
    )
    n_sh6_exportados = int(kpis["n_produtos_fim"]) if kpis else 0
    mercado_total = float(product_metrics["mundo_fim"].fillna(0).sum()) if product_metrics is not None else np.nan

    stat_items = [(f"{n_sh6_monitorados:,}".replace(",", "."), "produtos SH6 monitorados (Comtrade)")]
    if kpis:
        stat_items.append((f"{n_sh6_exportados:,}".replace(",", "."), "SH6 exportados pelo Brasil"))
    if pd.notna(mercado_total) and mercado_total > 0:
        stat_items.append((format_usd(mercado_total), "mercado mundial monitorado"))
    render_stat_row(stat_items)

    # ---------------- KPIs de crescimento / competitividade ----------------
    if kpis:
        if single_year_mode:
            st.info(
                "ℹ️ Apenas **1 ano** de dados Comtrade disponível. O **RCA (Balassa) é calculado normalmente** "
                "para esse ano, mas CAGR mundial, CAGR do Brasil por produto e a variação de RCA (tendência) "
                "exigem pelo menos 2 anos e por isso não estão disponíveis."
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ano de referência (Comtrade)", str(start_year))
            k2.metric("RCA médio dos produtos", format_num(kpis["rca_medio"], 2))
            k3.metric(
                "Produtos com Vantagem Comparativa",
                f"{kpis['n_produtos_com_vantagem_comparativa']} / {kpis['n_produtos_com_rca']}",
                help="Produtos com RCA ≥ 1 sobre o total de produtos com RCA calculável.",
            )
            k4.metric(
                "Índice HHI (concentração)",
                f"{kpis['hhi']:.0f}" if pd.notna(kpis["hhi"]) else "—",
                help="Herfindahl-Hirschman das exportações brasileiras por SH6 (0–10.000). "
                     "Quanto maior, mais concentrada/menos diversificada a pauta.",
            )
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CAGR Mundial (período)", format_pct(kpis["cagr_mundo"]))
            delta_kpi = None
            if pd.notna(kpis["cagr_brasil"]) and pd.notna(kpis["cagr_mundo"]):
                delta_kpi = format_pct(kpis["cagr_brasil"] - kpis["cagr_mundo"])
            k2.metric("CAGR do Brasil (período)", format_pct(kpis["cagr_brasil"]), delta=delta_kpi)

            if kpis["brasil_cresceu_mais_que_mundo"] is True:
                k3.metric("Brasil vs. Mundo", "✅ Cresceu mais")
            elif kpis["brasil_cresceu_mais_que_mundo"] is False:
                k3.metric("Brasil vs. Mundo", "⚠️ Cresceu menos")
            else:
                k3.metric("Brasil vs. Mundo", "—")

            k4.metric(
                "Índice HHI (concentração)",
                f"{kpis['hhi']:.0f}" if pd.notna(kpis["hhi"]) else "—",
                help="Herfindahl-Hirschman das exportações brasileiras por SH6 no ano final (0–10.000). "
                     "Quanto maior, mais concentrada/menos diversificada a pauta.",
            )

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Produtos SH6 exportados (início)", kpis["n_produtos_inicio"])
            k6.metric("Produtos SH6 exportados (fim)", kpis["n_produtos_fim"])
            k7.metric("Novos produtos no período", kpis["produtos_novos"])
            k8.metric("Produtos perdidos no período", kpis["produtos_perdidos"])
    else:
        st.warning("Envie e processe o arquivo Comtrade (barra lateral) para ver RCA, CAGR e quadrantes.")

    st.divider()

    # ---------------- Cartões de quadrante + barra de distribuição (estilo CNI) ----------------
    if product_metrics is not None and not product_metrics.empty:
        if single_year_mode:
            st.subheader("🧭 Produtos do Brasil por nível de RCA")
            st.caption("Classificação baseada apenas no nível do RCA (Balassa) no ano disponível — sem tendência.")
        else:
            st.subheader("🧭 Oportunidades do Brasil por quadrante (RCA × Crescimento do mercado mundial)")
            st.caption(
                "Cada cartão mostra quantos produtos SH6 estão no quadrante e quanto isso representa do "
                "valor total exportado pelo Brasil no ano final do período."
            )

        counts = product_metrics["quadrante"].value_counts().to_dict()
        values = {
            q: float(product_metrics.loc[product_metrics["quadrante"] == q, "brasil_fim"].fillna(0).sum())
            for q in active_quadrants
        }
        total_value = float(product_metrics["brasil_fim"].fillna(0).sum())

        render_quadrant_cards(active_quadrants, counts, values, total_value, QUADRANT_DESCRIPTIONS)
        render_distribution_bar(active_quadrants, values, total_value)

        # ---- Navegação rápida (estilo cartões "Explorar setores" / "Visão Estratégica") ----
        render_nav_cards(
            [
                {"icon": "📋", "title": "Explorar produtos (Tabela Completa)", "target": "📋 Tabela Completa (SH6)"},
                {"icon": "📦", "title": "Explorar por CUCI Grupo", "target": "📦 CUCI Grupo"},
                {"icon": "🗺️", "title": "Ver exportações por Estado", "target": "🗺️ Análise por Estado"},
            ],
            NAV_KEY,
        )

        st.divider()

        # ---- Detalhamento por produto (dispersão RCA x CAGR mundial) ----
        if not single_year_mode:
            with st.expander("🔬 Detalhamento por produto (dispersão RCA × CAGR mundial)", expanded=False):
                plot_df = product_metrics.dropna(subset=["cagr_mundo", "var_rca"]).copy()
                if plot_df.empty:
                    st.info("Não há produtos com CAGR mundial e variação de RCA calculáveis no período selecionado.")
                else:
                    plot_df["valor_bolha"] = plot_df["brasil_fim"].fillna(0).clip(lower=0)
                    fig = px.scatter(
                        plot_df, x="cagr_mundo", y="var_rca", color="quadrante",
                        size="valor_bolha", size_max=45, hover_name="sh6_desc",
                        hover_data={"sh6": True, "cagr_mundo": ":.1%", "var_rca": ":.2f", "valor_bolha": ":,.0f"},
                        color_discrete_map=QUADRANT_COLORS,
                        labels={"cagr_mundo": "CAGR do mercado mundial", "var_rca": "Variação do RCA"},
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig.update_layout(xaxis_tickformat=".0%", height=480, legend_title="Quadrante")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            with st.expander("🔬 Detalhamento por produto (ranking de RCA)", expanded=False):
                rank_rca = product_metrics.dropna(subset=["rca_fim"]).sort_values("rca_fim", ascending=False).head(25)
                if rank_rca.empty:
                    st.info("Não há RCA calculável para os produtos no ano selecionado.")
                else:
                    fig_rca = px.bar(
                        rank_rca, x="rca_fim", y="sh6_desc", orientation="h", color="quadrante",
                        color_discrete_map=QUADRANT_COLORS, hover_data={"sh6": True, "rca_fim": ":.2f"},
                    )
                    fig_rca.add_vline(x=1, line_dash="dash", line_color="gray")
                    fig_rca.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="",
                                           xaxis_title="RCA (Balassa)", height=600, legend_title="Quadrante")
                    st.plotly_chart(fig_rca, use_container_width=True)

    st.divider()

    # ---------------- ISIC Divisão / Seção — Visão Estratégica (estilo CNI) ----------------
    if has_comexstat and product_metrics is not None:
        st.subheader("🏭 Visão Estratégica da Indústria — participação de mercado por quadrante (ISIC)")
        st.caption(
            "% por quadrante = participação no valor exportado do setor pelo Brasil (não na quantidade de produtos)."
        )

        comexstat = st.session_state["comexstat"]
        base_isic = comexstat[comexstat["ano"] == end_year] if end_year else comexstat

        tab1, tab2 = st.tabs(["Por ISIC Divisão", "Por ISIC Seção"])

        for tab, cod_col, desc_col, label in [
            (tab1, "isic_div_cod", "isic_div_desc", "ISIC Divisão"),
            (tab2, "isic_sec_cod", "isic_sec_desc", "ISIC Seção"),
        ]:
            with tab:
                if desc_col not in base_isic.columns:
                    st.info(f"Coluna de {label} não encontrada no arquivo Comexstat.")
                    continue

                isic_table = build_isic_quadrant_table(base_isic, product_metrics, cod_col, desc_col, None, active_quadrants)
                if isic_table.empty:
                    st.info("Sem dados suficientes para este recorte.")
                    continue

                col_table, col_impact = st.columns([2, 1])
                with col_table:
                    render_cni_style_table(isic_table, active_quadrants, height_rows=15)

                with col_impact:
                    sector_options = [f"{r['cod']} — {r['desc']}" for _, r in isic_table.iterrows()]
                    sel = st.selectbox(f"Setor ({label})", sector_options, key=f"sel_{label}")
                    sel_desc = sel.split(" — ", 1)[1] if " — " in sel else sel

                    merged_products = base_isic[base_isic[desc_col] == sel_desc].merge(
                        product_metrics, left_on="sh6_cod", right_on="sh6", how="left", suffixes=("", "_pm")
                    )
                    merged_products = merged_products.drop_duplicates(subset=["sh6_cod"])

                    if single_year_mode:
                        st.markdown(f"**Produtos com menor RCA em '{sel_desc}'**")
                        worst = merged_products.dropna(subset=["rca_fim"]).sort_values("rca_fim").head(10)
                        render_impact_products(worst, value_col="rca_fim", value_label="RCA", market_col="mundo_fim",
                                                code_col="sh6_cod", desc_col="sh6_desc", value_is_currency=False)
                    else:
                        st.markdown(f"**Produtos mais impactados em '{sel_desc}'**")
                        merged_products["perda"] = merged_products["brasil_fim"] - merged_products["brasil_inicio"]
                        impacted = merged_products.dropna(subset=["perda"]).sort_values("perda").head(10)
                        render_impact_products(impacted, value_col="perda", value_label="Perda estimada", market_col="mundo_fim",
                                                code_col="sh6_cod", desc_col="sh6_desc")

    st.divider()

    # ---------------- Outros dados relevantes ----------------
    st.subheader("📌 Outros indicadores relevantes")

    colA, colB = st.columns(2)
    if product_metrics is not None:
        if single_year_mode:
            with colA:
                st.markdown("**Top 10 produtos com maior RCA**")
                top_gain = product_metrics.dropna(subset=["rca_fim"]).sort_values("rca_fim", ascending=False).head(10)
                st.dataframe(
                    top_gain[["sh6", "sh6_desc", "rca_fim", "mundo_fim", "brasil_fim"]]
                    .rename(columns={"sh6_desc": "Produto (SH6)"}),
                    use_container_width=True, hide_index=True,
                )
            with colB:
                st.markdown("**Top 10 produtos com menor RCA**")
                top_loss = product_metrics.dropna(subset=["rca_fim"]).sort_values("rca_fim", ascending=True).head(10)
                st.dataframe(
                    top_loss[["sh6", "sh6_desc", "rca_fim", "mundo_fim", "brasil_fim"]]
                    .rename(columns={"sh6_desc": "Produto (SH6)"}),
                    use_container_width=True, hide_index=True,
                )
        else:
            with colA:
                st.markdown("**Top 10 produtos com maior ganho de RCA**")
                top_gain = product_metrics.dropna(subset=["var_rca"]).sort_values("var_rca", ascending=False).head(10)
                st.dataframe(
                    top_gain[["sh6", "sh6_desc", "rca_inicio", "rca_fim", "var_rca", "cagr_mundo"]]
                    .rename(columns={"sh6_desc": "Produto (SH6)"}),
                    use_container_width=True, hide_index=True,
                )
            with colB:
                st.markdown("**Top 10 produtos com maior perda de RCA**")
                top_loss = product_metrics.dropna(subset=["var_rca"]).sort_values("var_rca", ascending=True).head(10)
                st.dataframe(
                    top_loss[["sh6", "sh6_desc", "rca_inicio", "rca_fim", "var_rca", "cagr_mundo"]]
                    .rename(columns={"sh6_desc": "Produto (SH6)"}),
                    use_container_width=True, hide_index=True,
                )

    if has_comexstat:
        comexstat = st.session_state["comexstat"]
        st.markdown("**Evolução das exportações totais do Brasil (Comexstat)**")
        evol = comexstat.groupby("ano")["valor_fob"].sum().reset_index()
        fig_evol = px.line(evol, x="ano", y="valor_fob", markers=True)
        fig_evol.update_layout(yaxis_title="US$ FOB", xaxis_title="Ano", height=350)
        st.plotly_chart(fig_evol, use_container_width=True)

        if "pais" in comexstat.columns:
            st.markdown("**Top 10 países de destino (ano mais recente disponível)**")
            last_year_c = comexstat["ano"].max()
            top_paises = (
                comexstat[comexstat["ano"] == last_year_c].groupby("pais")["valor_fob"]
                .sum().sort_values(ascending=False).head(10).reset_index()
            )
            fig_paises = px.bar(top_paises, x="valor_fob", y="pais", orientation="h")
            fig_paises.update_layout(yaxis_title="", xaxis_title="US$ FOB", height=400,
                                      yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_paises, use_container_width=True)

    st.caption("➡️ Use o menu lateral (ou os cartões acima) para navegar entre as páginas do sistema.")


# ==============================================================================
# 7. PÁGINA 2 — TABELA COMPLETA POR SH6
# ==============================================================================


def page_tabela_completa():
    st.title("📋 Tabela Completa por Produto (SH6)")

    if not has_comexstat:
        st.warning("Envie o arquivo Comexstat na barra lateral primeiro.")
        return

    comexstat = st.session_state["comexstat"]

    st.subheader("Filtros")
    f1, f2, f3 = st.columns(3)
    with f1:
        cuci_opts = sorted(comexstat["cuci_desc"].dropna().unique().tolist()) if "cuci_desc" in comexstat.columns else []
        cuci_sel = st.multiselect("CUCI Grupo", cuci_opts)
    with f2:
        secao_opts = sorted(comexstat["isic_sec_desc"].dropna().unique().tolist()) if "isic_sec_desc" in comexstat.columns else []
        secao_sel = st.multiselect("ISIC Seção", secao_opts)
    with f3:
        div_opts = sorted(comexstat["isic_div_desc"].dropna().unique().tolist()) if "isic_div_desc" in comexstat.columns else []
        div_sel = st.multiselect("ISIC Divisão", div_opts)

    search = st.text_input("🔎 Buscar por código ou descrição do SH6")

    df = comexstat.copy()
    if cuci_sel and "cuci_desc" in df.columns:
        df = df[df["cuci_desc"].isin(cuci_sel)]
    if secao_sel and "isic_sec_desc" in df.columns:
        df = df[df["isic_sec_desc"].isin(secao_sel)]
    if div_sel and "isic_div_desc" in df.columns:
        df = df[df["isic_div_desc"].isin(div_sel)]
    if search:
        mask = df["sh6_cod"].astype(str).str.contains(search, case=False, na=False)
        if "sh6_desc" in df.columns:
            mask = mask | df["sh6_desc"].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

    st.divider()

    st.subheader("Valores anuais de exportação do Brasil por SH6 (Comexstat)")
    if df.empty:
        st.info("Nenhum registro para os filtros selecionados.")
    else:
        pivot_annual = df.pivot_table(
            index=["sh6_cod", "sh6_desc"], columns="ano", values="valor_fob", aggfunc="sum", fill_value=0
        ).reset_index()
        pivot_annual.columns = [str(c) for c in pivot_annual.columns]

        display_pivot = pivot_annual.copy()
        year_cols = [c for c in display_pivot.columns if c.isdigit()]
        for yc in year_cols:
            display_pivot[yc] = display_pivot[yc].apply(format_usd)

        st.dataframe(
            display_pivot.rename(columns={"sh6_cod": "SH6", "sh6_desc": "Descrição"}),
            use_container_width=True, hide_index=True, height=380,
        )

        st.download_button(
            "⬇️ Baixar tabela (CSV)",
            pivot_annual.to_csv(index=False).encode("utf-8"),
            file_name="exportacoes_por_sh6_anual.csv",
            mime="text/csv",
        )

    st.divider()

    st.subheader("Indicadores de competitividade por SH6")
    if product_metrics is None:
        st.info(
            "Envie e processe o arquivo Comtrade na barra lateral para ver crescimento de participação, "
            "RCA, CAGR do produto e variação de share mundial/Brasil."
        )
    else:
        sh6_in_filter = df["sh6_cod"].unique().tolist() if not df.empty else []
        analytic = (
            product_metrics[product_metrics["sh6"].isin(sh6_in_filter)].copy()
            if sh6_in_filter else product_metrics.iloc[0:0].copy()
        )

        if analytic.empty:
            st.info("Nenhum produto do filtro atual possui dados do Comtrade correspondentes.")
        else:
            analytic_disp = analytic[[
                "sh6", "sh6_desc", "mundo_inicio", "mundo_fim", "brasil_inicio", "brasil_fim",
                "participacao_inicio", "participacao_fim", "var_participacao",
                "cagr_mundo", "cagr_brasil", "rca_inicio", "rca_fim", "var_rca", "quadrante",
            ]].rename(columns={
                "sh6": "SH6", "sh6_desc": "Descrição",
                "mundo_inicio": f"Mundo {start_year}", "mundo_fim": f"Mundo {end_year}",
                "brasil_inicio": f"Brasil {start_year}", "brasil_fim": f"Brasil {end_year}",
                "participacao_inicio": f"Part. Mundial {start_year}", "participacao_fim": f"Part. Mundial {end_year}",
                "var_participacao": "Variação da Participação (Mundo)",
                "cagr_mundo": "CAGR Mundo", "cagr_brasil": "CAGR Brasil (produto)",
                "rca_inicio": f"RCA {start_year}", "rca_fim": f"RCA {end_year}",
                "var_rca": "Variação do RCA (espaço)", "quadrante": "Quadrante",
            })

            fmt_cols_usd = [f"Mundo {start_year}", f"Mundo {end_year}", f"Brasil {start_year}", f"Brasil {end_year}"]
            fmt_cols_pct = [f"Part. Mundial {start_year}", f"Part. Mundial {end_year}",
                            "Variação da Participação (Mundo)", "CAGR Mundo", "CAGR Brasil (produto)"]

            show = analytic_disp.copy()
            for c in fmt_cols_usd:
                show[c] = show[c].apply(format_usd)
            for c in fmt_cols_pct:
                show[c] = show[c].apply(format_pct)
            for c in [f"RCA {start_year}", f"RCA {end_year}", "Variação do RCA (espaço)"]:
                show[c] = show[c].apply(lambda x: format_num(x, 2))

            st.dataframe(show, use_container_width=True, hide_index=True, height=450)

            st.download_button(
                "⬇️ Baixar indicadores completos (CSV)",
                analytic_disp.to_csv(index=False).encode("utf-8"),
                file_name="indicadores_competitividade_sh6.csv",
                mime="text/csv",
            )


# ==============================================================================
# 8. PÁGINA 3 — CUCI GRUPO
# ==============================================================================


def page_cuci_grupo():
    st.title("📦 Análise por CUCI Grupo")

    if not has_comexstat:
        st.warning("Envie o arquivo Comexstat na barra lateral primeiro.")
        return

    comexstat = st.session_state["comexstat"]
    if "cuci_desc" not in comexstat.columns:
        st.error("Coluna de CUCI Grupo não encontrada no arquivo Comexstat.")
        return

    last_year = end_year if end_year else comexstat["ano"].max()
    base = comexstat[comexstat["ano"] == last_year]

    rank = base.groupby(["cuci_cod", "cuci_desc"])["valor_fob"].sum().sort_values(ascending=False).reset_index()
    st.subheader(f"Ranking de CUCI Grupos por valor exportado em {last_year}")
    fig_rank = px.bar(rank.head(20), x="valor_fob", y="cuci_desc", orientation="h")
    fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="", xaxis_title="US$ FOB", height=550)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.divider()

    st.subheader("Detalhamento de um CUCI Grupo")
    cuci_options = sorted(comexstat["cuci_desc"].dropna().unique().tolist())
    if not cuci_options:
        st.info("Nenhum CUCI Grupo disponível.")
        return
    selected_cuci = st.selectbox("Selecione o CUCI Grupo", cuci_options)

    grupo_df = comexstat[comexstat["cuci_desc"] == selected_cuci]

    c1, c2, c3 = st.columns(3)
    c1.metric("Valor exportado (último ano)", format_usd(grupo_df[grupo_df["ano"] == last_year]["valor_fob"].sum()))
    evol_grupo = grupo_df.groupby("ano")["valor_fob"].sum()
    if len(evol_grupo) >= 2:
        v0, v1 = evol_grupo.iloc[0], evol_grupo.iloc[-1]
        growth = (v1 / v0 - 1) if v0 else None
        c2.metric("Crescimento no período (nominal)", format_pct(growth) if growth is not None else "—")
    n_sh6 = grupo_df["sh6_cod"].nunique()
    c3.metric("Nº de SH6 no grupo", n_sh6)

    fig_evol = px.line(evol_grupo.reset_index(), x="ano", y="valor_fob", markers=True,
                        title=f"Evolução das exportações — {selected_cuci}")
    fig_evol.update_layout(yaxis_title="US$ FOB", xaxis_title="Ano")
    st.plotly_chart(fig_evol, use_container_width=True)

    st.markdown("**Composição por ISIC Divisão / Seção dentro do grupo**")
    c1, c2 = st.columns(2)
    with c1:
        if "isic_div_desc" in grupo_df.columns:
            comp_div = grupo_df[grupo_df["ano"] == last_year].groupby("isic_div_desc")["valor_fob"].sum().sort_values(ascending=False).reset_index()
            if not comp_div.empty:
                fig_div = px.pie(comp_div, names="isic_div_desc", values="valor_fob", title="ISIC Divisão")
                st.plotly_chart(fig_div, use_container_width=True)
    with c2:
        if "isic_sec_desc" in grupo_df.columns:
            comp_sec = grupo_df[grupo_df["ano"] == last_year].groupby("isic_sec_desc")["valor_fob"].sum().sort_values(ascending=False).reset_index()
            if not comp_sec.empty:
                fig_sec = px.pie(comp_sec, names="isic_sec_desc", values="valor_fob", title="ISIC Seção")
                st.plotly_chart(fig_sec, use_container_width=True)

    st.markdown("**SH6 que compõem o grupo**")
    group_cols = ["sh6_cod", "sh6_desc"]
    for c in ["isic_div_desc", "isic_sec_desc"]:
        if c in grupo_df.columns:
            group_cols.append(c)

    sh6_list = (
        grupo_df[grupo_df["ano"] == last_year].groupby(group_cols)["valor_fob"]
        .sum().reset_index().sort_values("valor_fob", ascending=False)
    )

    if product_metrics is not None:
        sh6_list = sh6_list.merge(
            product_metrics[["sh6", "cagr_mundo", "var_rca", "quadrante"]],
            left_on="sh6_cod", right_on="sh6", how="left",
        ).drop(columns=["sh6"])

    show = sh6_list.copy()
    show["valor_fob"] = show["valor_fob"].apply(format_usd)
    if "cagr_mundo" in show.columns:
        show["cagr_mundo"] = show["cagr_mundo"].apply(format_pct)
    st.dataframe(
        show.rename(columns={
            "sh6_cod": "SH6", "sh6_desc": "Descrição", "valor_fob": "Valor FOB",
            "isic_div_desc": "ISIC Divisão", "isic_sec_desc": "ISIC Seção",
            "cagr_mundo": "CAGR Mundo", "var_rca": "Var. RCA", "quadrante": "Quadrante",
        }),
        use_container_width=True, hide_index=True, height=400,
    )

    if product_metrics is not None and "quadrante" in sh6_list.columns:
        st.markdown("**Distribuição dos SH6 do grupo por quadrante**")
        active_q = SINGLE_YEAR_QUADRANTS if st.session_state.get("single_year_mode") else QUADRANT_ORDER
        qc = sh6_list["quadrante"].value_counts().reindex(active_q).fillna(0).reset_index()
        qc.columns = ["quadrante", "n"]
        qc["quadrante_curto"] = qc["quadrante"].map(QUADRANT_SHORT)
        fig_q = px.bar(qc, x="quadrante_curto", y="n", color="quadrante", color_discrete_map=QUADRANT_COLORS)
        fig_q.update_layout(showlegend=False, xaxis_title="", yaxis_title="Nº de produtos SH6")
        st.plotly_chart(fig_q, use_container_width=True)


# ==============================================================================
# 9. PÁGINA 4 — ANÁLISE POR ESTADO
# ==============================================================================


def page_estado():
    st.title("🗺️ Análise por Estado Exportador")
    st.caption(
        "Exportações do Brasil por Estado (UF), CUCI Grupo e SH6. O RCA e o crescimento do "
        "mercado mundial (CAGR Mundo) exibidos aqui vêm dos dados nacionais calculados a "
        "partir do arquivo UN Comtrade — não há um recorte mundial por Estado."
    )

    if not has_comexstat_uf:
        st.info(
            "⬅️ Envie o arquivo **Comexstat por Estado** (opcional) na barra lateral para "
            "habilitar esta página. Ele deve ter a mesma estrutura do Comexstat nacional, "
            "acrescida de uma coluna de UF/Estado."
        )
        return

    comexstat_uf = st.session_state["comexstat_uf"]

    st.subheader("Filtros")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        uf_opts = sorted(comexstat_uf["uf"].dropna().unique().tolist())
        uf_sel = st.multiselect("Estado (UF)", uf_opts)
    with f2:
        cuci_opts = sorted(comexstat_uf["cuci_desc"].dropna().unique().tolist()) if "cuci_desc" in comexstat_uf.columns else []
        cuci_sel = st.multiselect("CUCI Grupo", cuci_opts)
    with f3:
        div_opts = sorted(comexstat_uf["isic_div_desc"].dropna().unique().tolist()) if "isic_div_desc" in comexstat_uf.columns else []
        div_sel = st.multiselect("ISIC Divisão", div_opts)
    with f4:
        sec_opts = sorted(comexstat_uf["isic_sec_desc"].dropna().unique().tolist()) if "isic_sec_desc" in comexstat_uf.columns else []
        sec_sel = st.multiselect("ISIC Seção", sec_opts)

    search_sh6 = st.text_input("🔎 Buscar por código ou descrição do SH6 (estado)")

    df = comexstat_uf.copy()
    if uf_sel:
        df = df[df["uf"].isin(uf_sel)]
    if cuci_sel and "cuci_desc" in df.columns:
        df = df[df["cuci_desc"].isin(cuci_sel)]
    if div_sel and "isic_div_desc" in df.columns:
        df = df[df["isic_div_desc"].isin(div_sel)]
    if sec_sel and "isic_sec_desc" in df.columns:
        df = df[df["isic_sec_desc"].isin(sec_sel)]
    if search_sh6:
        mask = df["sh6_cod"].astype(str).str.contains(search_sh6, case=False, na=False)
        if "sh6_desc" in df.columns:
            mask = mask | df["sh6_desc"].astype(str).str.contains(search_sh6, case=False, na=False)
        df = df[mask]

    if df.empty:
        st.info("Nenhum registro para os filtros selecionados.")
        return

    last_year_uf = end_year if (end_year and end_year in df["ano"].unique()) else df["ano"].max()

    st.divider()

    # ---------------- KPIs ----------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Valor exportado em {last_year_uf}", format_usd(df[df["ano"] == last_year_uf]["valor_fob"].sum()))
    k2.metric("Nº de Estados exportadores", df[df["ano"] == last_year_uf]["uf"].nunique())
    top_uf_series = df[df["ano"] == last_year_uf].groupby("uf")["valor_fob"].sum().sort_values(ascending=False)
    k3.metric("Estado líder", top_uf_series.index[0] if not top_uf_series.empty else "—")
    k4.metric("Nº de SH6 no recorte", df["sh6_cod"].nunique())

    st.divider()

    # ---------------- Ranking de Estados ----------------
    st.subheader(f"Ranking de Estados por valor exportado em {last_year_uf}")
    rank_uf = top_uf_series.reset_index()
    fig_uf = px.bar(rank_uf.head(27), x="valor_fob", y="uf", orientation="h",
                     color="valor_fob", color_continuous_scale="Blues")
    fig_uf.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="",
                          xaxis_title="US$ FOB", height=600, coloraxis_showscale=False)
    st.plotly_chart(fig_uf, use_container_width=True)

    # ---------------- Evolução dos principais Estados ----------------
    st.subheader("Evolução temporal — principais Estados no recorte filtrado")
    top_n = st.slider("Número de Estados no gráfico de evolução", 3, 15, 6)
    top_ufs_list = top_uf_series.head(top_n).index.tolist()
    evol_uf = df[df["uf"].isin(top_ufs_list)].groupby(["ano", "uf"])["valor_fob"].sum().reset_index()
    fig_evol_uf = px.line(evol_uf, x="ano", y="valor_fob", color="uf", markers=True)
    fig_evol_uf.update_layout(yaxis_title="US$ FOB", xaxis_title="Ano", height=420)
    st.plotly_chart(fig_evol_uf, use_container_width=True)

    st.divider()

    # ---------------- Detalhamento por Estado ----------------
    st.subheader("Detalhamento por Estado")
    selected_uf = st.selectbox("Selecione um Estado", sorted(df["uf"].unique().tolist()))
    uf_df = df[df["uf"] == selected_uf]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**CUCI Grupos exportados por {selected_uf} ({last_year_uf})**")
        if "cuci_desc" in uf_df.columns:
            comp_cuci = uf_df[uf_df["ano"] == last_year_uf].groupby("cuci_desc")["valor_fob"].sum().sort_values(ascending=False).head(10).reset_index()
            if not comp_cuci.empty:
                fig_cuci_uf = px.bar(comp_cuci, x="valor_fob", y="cuci_desc", orientation="h")
                fig_cuci_uf.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="",
                                           xaxis_title="US$ FOB", height=400)
                st.plotly_chart(fig_cuci_uf, use_container_width=True)
    with c2:
        st.markdown(f"**Principais SH6 exportados por {selected_uf} ({last_year_uf})**")
        top_sh6_uf = uf_df[uf_df["ano"] == last_year_uf].groupby(["sh6_cod", "sh6_desc"])["valor_fob"].sum().sort_values(ascending=False).head(10).reset_index()
        if not top_sh6_uf.empty:
            fig_sh6_uf = px.bar(top_sh6_uf, x="valor_fob", y="sh6_desc", orientation="h")
            fig_sh6_uf.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="",
                                      xaxis_title="US$ FOB", height=400)
            st.plotly_chart(fig_sh6_uf, use_container_width=True)

    st.divider()

    # ---------------- Cruzamento com RCA/Quadrante nacional (base Comtrade) ----------------
    if product_metrics is not None:
        st.subheader("🎯 Exposição do Estado às dinâmicas globais (RCA e Quadrante nacional)")
        st.caption(
            "Cada SH6 exportado pelo Estado é cruzado com o quadrante de competitividade "
            "nacional (calculado a partir do Comtrade), mostrando quanto da pauta do Estado "
            "está em produtos com dinâmica mundial favorável ou desfavorável."
        )

        uf_year = uf_df[uf_df["ano"] == last_year_uf]
        uf_merged = uf_year.merge(
            product_metrics[["sh6", "cagr_mundo", "var_rca", "quadrante"]],
            left_on="sh6_cod", right_on="sh6", how="left",
        )
        uf_merged["quadrante"] = uf_merged["quadrante"].fillna("Sem dados suficientes")

        quad_val = uf_merged.groupby("quadrante")["valor_fob"].sum().reindex(
            QUADRANT_ORDER + ["Sem dados suficientes"]
        ).fillna(0)
        total_val = quad_val.sum()

        cols_q = st.columns(4)
        for i, q in enumerate(QUADRANT_ORDER):
            pct = (quad_val.get(q, 0) / total_val) if total_val else np.nan
            cols_q[i].metric(QUADRANT_SHORT[q], format_pct(pct))

        fig_quad_uf = px.pie(
            names=quad_val.index, values=quad_val.values,
            color=quad_val.index, color_discrete_map=QUADRANT_COLORS,
            title=f"Composição da pauta de {selected_uf} por quadrante de competitividade",
        )
        st.plotly_chart(fig_quad_uf, use_container_width=True)

    st.divider()

    # ---------------- RCA Regional (dentro do Brasil) ----------------
    st.subheader("🧮 RCA Regional — especialização do Estado frente ao padrão nacional")
    st.caption(
        "RCA Regional > 1 indica que o Estado é proporcionalmente mais especializado naquele "
        "produto do que o Brasil como um todo (calculado apenas com dados do Comexstat, sem "
        "depender do arquivo Comtrade)."
    )
    reg_rca = compute_regional_rca(comexstat_uf, last_year_uf)
    if reg_rca.empty:
        st.info("Não foi possível calcular o RCA Regional para o ano selecionado.")
    else:
        reg_rca_uf = reg_rca[reg_rca["uf"] == selected_uf].merge(
            comexstat_uf[comexstat_uf["ano"] == last_year_uf][["sh6_cod", "sh6_desc"]].drop_duplicates("sh6_cod"),
            on="sh6_cod", how="left",
        )
        reg_rca_uf = reg_rca_uf.sort_values("rca_regional", ascending=False)
        show_reg = reg_rca_uf[["sh6_cod", "sh6_desc", "exp_estado_produto", "rca_regional"]].rename(columns={
            "sh6_cod": "SH6", "sh6_desc": "Descrição",
            "exp_estado_produto": "Valor Exportado", "rca_regional": "RCA Regional",
        })
        show_reg["Valor Exportado"] = show_reg["Valor Exportado"].apply(format_usd)
        show_reg["RCA Regional"] = show_reg["RCA Regional"].apply(lambda x: format_num(x, 2))
        st.dataframe(show_reg.head(20), use_container_width=True, hide_index=True, height=400)

    st.divider()

    # ---------------- Tabela pivotada Estado x Ano ----------------
    st.subheader("Tabela: valor exportado por Estado e Ano (recorte filtrado)")
    pivot_uf = df.pivot_table(index="uf", columns="ano", values="valor_fob", aggfunc="sum", fill_value=0).reset_index()
    pivot_uf.columns = [str(c) for c in pivot_uf.columns]
    display_pivot_uf = pivot_uf.copy()
    year_cols_uf = [c for c in display_pivot_uf.columns if c.isdigit()]
    for yc in year_cols_uf:
        display_pivot_uf[yc] = display_pivot_uf[yc].apply(format_usd)
    st.dataframe(display_pivot_uf.rename(columns={"uf": "UF"}), use_container_width=True, hide_index=True, height=380)

    st.download_button(
        "⬇️ Baixar tabela Estado x Ano (CSV)",
        pivot_uf.to_csv(index=False).encode("utf-8"),
        file_name="exportacoes_por_estado_ano.csv",
        mime="text/csv",
    )


# ==============================================================================
# 10. ROTEAMENTO DE PÁGINAS
# ==============================================================================

if PAGE == "📊 Dashboard Resumo":
    page_dashboard()
elif PAGE == "📋 Tabela Completa (SH6)":
    page_tabela_completa()
elif PAGE == "📦 CUCI Grupo":
    page_cuci_grupo()
elif PAGE == "🗺️ Análise por Estado":
    page_estado()
