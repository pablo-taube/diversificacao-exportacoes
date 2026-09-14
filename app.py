# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS (GLASSMORMISMO CUPERTINO)
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
# 0. CONFIGURAÇÃO DA PÁGINA
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
# 2. PADRONIZAÇÃO COMEXSTAT
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

_COMEXSTAT_FIELD_ORDER = [
    "ano", "pais", "sh6_cod", "sh6_desc",
    "cgce2_cod", "cgce2_desc", "cgce1_cod", "cgce1_desc",
    "cuci_cod", "cuci_desc",
    "isic_div_cod", "isic_div_desc", "isic_sec_cod", "isic_sec_desc",
    "valor_fob", "uf",
]


def standardize_comexstat(df: pd.DataFrame) -> pd.DataFrame:
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


# ==============================================================================
# 3. PADRONIZAÇÃO COMTRADE (CÁLCULO RCA E RSCA MULTI-PAÍS)
# ==============================================================================

def guess_comtrade_columns(df: pd.DataFrame) -> dict:
    return {
        "year": find_column(df, ["refyear", "ano", "year", "period"]),
        "reporter": find_column(df, ["reporterdesc", "reporter_desc", "reporter", "pais_origem"]),
        "partner": find_column(df, ["partnerdesc", "partner_desc", "partner", "parceiro"]),
        "sh6": find_column(df, ["cmdcode", "sh6_cod", "sh6", "codigo_sh6", "commoditycode"]),
        "sh6_desc": find_column(df, ["cmddesc", "sh6_desc", "descricao_sh6", "commoditydesc", "description"]),
        "value": find_column(df, ["primaryvalue", "tradevalue", "value", "valor", "fobvalue"]),
    }


def standardize_comtrade(
    df: pd.DataFrame,
    year_col: str,
    reporter_col: str,
    partner_col: str,
    sh6_col: str,
    sh6desc_col: str | None,
    value_col: str,
    world_partner_values: list,
    selected_reporters: list = None,
) -> pd.DataFrame:
    d = df.copy()
    cols = {
        year_col: "ano",
        reporter_col: "reporter",
        partner_col: "partner",
        sh6_col: "sh6",
        value_col: "valor",
    }
    if sh6desc_col and sh6desc_col in d.columns:
        cols[sh6desc_col] = "sh6_desc"
    d = d.rename(columns=cols)

    d["ano"] = pd.to_numeric(d["ano"], errors="coerce").astype("Int64")
    extracted = d["sh6"].astype(str).str.extract(r"(\d+)")[0]
    d["sh6"] = extracted.fillna(d["sh6"].astype(str)).astype(str).str.zfill(6)
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce").fillna(0)

    if "sh6_desc" not in d.columns:
        d["sh6_desc"] = d["sh6"]

    world_str_list = [str(v).lower() for v in world_partner_values]
    d["is_world"] = d["partner"].astype(str).str.lower().isin(world_str_list)
    d = d[d["is_world"]].copy()

    if selected_reporters:
        rep_str_list = [str(r).lower() for r in selected_reporters]
        d = d[d["reporter"].astype(str).str.lower().isin(rep_str_list)]

    agg = d.groupby(["ano", "reporter", "sh6", "sh6_desc"], as_index=False)["valor"].sum()
    return agg


# ==============================================================================
# 4. MOTOR DE CÁLCULO DE METRICAS (RCA, RSCA, CAGR, DIVERSIFICAÇÃO)
# ==============================================================================

@st.cache_data(show_spinner=False)
def compute_comtrade_metrics(comtrade_tidy: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    years_needed = [start_year] if start_year == end_year else [start_year, end_year]
    df_filtered = comtrade_tidy[comtrade_tidy["ano"].isin(years_needed)].copy()
    
    rows = []
    for yr in years_needed:
        df_yr = df_filtered[df_filtered["ano"] == yr]
        if df_yr.empty:
            continue
        
        pv = df_yr.pivot_table(index="sh6", columns="reporter", values="valor", aggfunc="sum", fill_value=0)
        X_ij = pv
        X_i = pv.sum(axis=0)
        X_wj = pv.sum(axis=1)
        X_w = pv.values.sum()

        share_pais = X_ij.div(X_i, axis=1)
        share_mundo = X_wj / X_w if X_w > 0 else 0
        
        rca_mat = share_pais.div(share_mundo, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
        rsca_mat = (rca_mat - 1) / (rca_mat + 1)
        rsca_mat = rsca_mat.fillna(-1)

        sh6_descs = df_yr.groupby("sh6")["sh6_desc"].first()

        for sh6_code in pv.index:
            desc = sh6_descs.get(sh6_code, sh6_code)
            world_val = X_wj.get(sh6_code, 0.0)
            for rep in pv.columns:
                val = pv.loc[sh6_code, rep]
                rca = rca_mat.loc[sh6_code, rep]
                rsca = rsca_mat.loc[sh6_code, rep]
                rows.append({
                    "ano": yr,
                    "reporter": rep,
                    "sh6": sh6_code,
                    "sh6_desc": desc,
                    "valor": val,
                    "mundo_valor": world_val,
                    "rca": rca,
                    "rsca": rsca,
                })

    df_metrics = pd.DataFrame(rows)
    return df_metrics


def compute_state_diversification_potentials(
    comexstat_uf: pd.DataFrame, comtrade_metrics: pd.DataFrame, year: int
) -> pd.DataFrame:
    base_uf = comexstat_uf[comexstat_uf["ano"] == year].copy() if "ano" in comexstat_uf.columns else comexstat_uf.copy()
    if base_uf.empty or "uf" not in base_uf.columns:
        return pd.DataFrame()

    brazil_rep_name = next((r for r in comtrade_metrics["reporter"].unique() if str(r).lower() in ["brazil", "brasil", "bra"]), None)
    if not brazil_rep_name:
        brazil_rep_name = comtrade_metrics["reporter"].unique()[0]

    br_metrics = comtrade_metrics[
        (comtrade_metrics["reporter"] == brazil_rep_name) & (comtrade_metrics["ano"] == year)
    ].copy()

    br_metrics["vantagem_nacional"] = br_metrics["rca"] >= 1.0

    tot_uf = base_uf.groupby("uf")["valor_fob"].sum().to_dict()
    
    group_keys = ["uf", "sh6_cod"]
    extra_cols = [c for c in ["sh6_desc", "cuci_desc", "cuci_cod", "isic_div_desc", "cgce1_desc"] if c in base_uf.columns]
    
    exp_uf = base_uf.groupby(group_keys + extra_cols, as_index=False)["valor_fob"].sum()
    exp_uf["uf_total"] = exp_uf["uf"].map(tot_uf)
    exp_uf["share_local"] = exp_uf["valor_fob"] / exp_uf["uf_total"]

    merged = pd.merge(exp_uf, br_metrics[["sh6", "rca", "rsca", "mundo_valor"]], left_on="sh6_cod", right_on="sh6", how="right")
    merged["uf"] = merged["uf"].fillna("Sem Exportação Registrada")
    merged["valor_fob"] = merged["valor_fob"].fillna(0)
    merged["share_local"] = merged["share_local"].fillna(0)

    merged["potencial_score"] = np.where(
        merged["rca"] >= 1.0,
        merged["rca"] * (1 - merged["share_local"]),
        0
    )

    return merged.sort_values("potencial_score", ascending=False)


# ==============================================================================
# 5. ESTRUTURA VISUAL (DESIGN CUPERTINO, PASTEL & GLASSMORPHISM)
# ==============================================================================

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Global Reset & Cupertino Canvas */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Plus Jakarta Sans", sans-serif;
            background-color: #f6f8fa;
            color: #1c1c1e;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3.5rem;
            max-width: 95%;
        }

        /* Glassmorphism Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.45) !important;
            backdrop-filter: blur(25px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(25px) saturate(190%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.7) !important;
        }

        /* Glass Dashboard Base Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.03);
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 40px 0 rgba(0, 0, 0, 0.06);
        }

        /* Stat Metrics Bar (Estilo Imagem 1 e 2) */
        .stat-row {
            display: flex;
            gap: 32px;
            flex-wrap: wrap;
            margin: 12px 0 28px 0;
            padding: 20px 28px;
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02);
        }

        .stat-item .stat-value {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
            color: #1d1d1f;
        }

        .stat-item .stat-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #86868b;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Quadrantes CNI Pastel & Glass */
        .quadrant-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 14px 0 28px 0;
        }

        @media (max-width: 960px) {
            .quadrant-grid { grid-template-columns: 1fr; }
        }

        .quadrant-card {
            border-radius: 24px;
            padding: 26px 30px;
            backdrop-filter: blur(25px) saturate(180%);
            -webkit-backdrop-filter: blur(25px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.02);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 210px;
            transition: all 0.25s ease;
        }

        .quadrant-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.05);
        }

        .quadrant-card .qc-title {
            font-weight: 800;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .quadrant-card .qc-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }

        .quadrant-card .qc-desc {
            font-size: 0.82rem;
            color: #48484a;
            margin: 12px 0 18px 0;
            line-height: 1.55;
            font-weight: 450;
        }

        .quadrant-card .qc-bottom {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-top: auto;
        }

        .quadrant-card .qc-count {
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1;
            color: #1c1c1e;
        }

        .quadrant-card .qc-count-label {
            font-size: 0.75rem;
            color: #8e8e93;
            font-weight: 600;
            margin-top: 2px;
        }

        .quadrant-card .qc-value {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #1c1c1e;
            text-align: right;
        }

        .quadrant-card .qc-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 6px;
            letter-spacing: 0.01em;
        }

        /* Glass Tables (Mac OS Style) */
        .cni-table-wrapper {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(20px) saturate(180%);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
        }

        .cni-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.86rem;
        }

        .cni-table th {
            text-align: left;
            font-size: 0.72rem;
            color: #8e8e93;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 14px 18px;
            background: rgba(245, 245, 247, 0.6);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            font-weight: 700;
        }

        .cni-table td {
            padding: 14px 18px;
            border-bottom: 1px solid rgba(0, 0, 0, 0.03);
            vertical-align: middle;
            color: #1c1c1e;
        }

        .cni-table tr:hover td {
            background: rgba(255, 255, 255, 0.5);
        }

        /* Cupertino Custom Buttons */
        .stButton>button {
            border-radius: 14px !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            background: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(10px) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            background: rgba(255, 255, 255, 0.95) !important;
            transform: scale(1.01);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Palette Pastel Inspirada na Imagem 2 (CNI)
PASTEL_COLORS = {
    "green_main": "#10b981",
    "green_bg": "rgba(236, 253, 245, 0.75)",
    "amber_main": "#f59e0b",
    "amber_bg": "rgba(254, 243, 199, 0.75)",
    "blue_main": "#3b82f6",
    "blue_bg": "rgba(239, 246, 255, 0.75)",
    "red_main": "#ef4444",
    "red_bg": "rgba(254, 242, 242, 0.75)",
    "gray_main": "#9ca3af",
    "gray_bg": "rgba(243, 244, 246, 0.75)",
}


# ==============================================================================
# 6. GERENCIAMENTO DE ESTADO E SIDEBAR
# ==============================================================================

inject_custom_css()

st.sidebar.title("🌎 Navegação")
PAGE = st.sidebar.radio(
    "Selecione o Módulo:",
    [
        "📊 RCA/RSCA Global (UN Comtrade)",
        "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)",
        "🗺️ Potencial de Diversificação por Estado (UF)",
    ],
)

st.sidebar.divider()
st.sidebar.header("📁 Carga de Dados")

file_comtrade = st.sidebar.file_uploader("1. UN Comtrade (CSV, XLSX, Parquet, JSON)", type=["csv", "xlsx", "xls", "parquet", "json"])
file_comexstat = st.sidebar.file_uploader("2. ComexStat Brasil (Nacional)", type=["csv", "xlsx", "xls", "parquet"])
file_comexstat_uf = st.sidebar.file_uploader("3. ComexStat por Estado (UF)", type=["csv", "xlsx", "xls", "parquet"])

if file_comtrade:
    if st.session_state.get("_file_comtrade_name") != file_comtrade.name:
        raw_ct = read_any_file(file_comtrade.getvalue(), file_comtrade.name)
        st.session_state["raw_comtrade"] = raw_ct
        st.session_state["_file_comtrade_name"] = file_comtrade.name

if file_comexstat:
    if st.session_state.get("_file_comexstat_name") != file_comexstat.name:
        raw_cs = read_any_file(file_comexstat.getvalue(), file_comexstat.name)
        st.session_state["comexstat"] = standardize_comexstat(raw_cs)
        st.session_state["_file_comexstat_name"] = file_comexstat.name

if file_comexstat_uf:
    if st.session_state.get("_file_comexstat_uf_name") != file_comexstat_uf.name:
        raw_cs_uf = read_any_file(file_comexstat_uf.getvalue(), file_comexstat_uf.name)
        st.session_state["comexstat_uf"] = standardize_comexstat(raw_cs_uf)
        st.session_state["_file_comexstat_uf_name"] = file_comexstat_uf.name

if "raw_comtrade" in st.session_state:
    df_ct = st.session_state["raw_comtrade"]
    guesses = guess_comtrade_columns(df_ct)
    cols = list(df_ct.columns)

    with st.sidebar.expander("⚙️ Configurações UN Comtrade", expanded=False):
        c_yr = st.selectbox("Ano", cols, index=cols.index(guesses["year"]) if guesses["year"] in cols else 0)
        c_rep = st.selectbox("Reporter", cols, index=cols.index(guesses["reporter"]) if guesses["reporter"] in cols else 0)
        c_prt = st.selectbox("Partner", cols, index=cols.index(guesses["partner"]) if guesses["partner"] in cols else 0)
        c_sh6 = st.selectbox("Código SH6", cols, index=cols.index(guesses["sh6"]) if guesses["sh6"] in cols else 0)
        c_desc = st.selectbox("Descrição SH6", ["(Nenhuma)"] + cols, index=(cols.index(guesses["sh6_desc"])+1) if guesses["sh6_desc"] in cols else 0)
        c_val = st.selectbox("Valor (US$)", cols, index=cols.index(guesses["value"]) if guesses["value"] in cols else 0)

        partners_list = sorted(df_ct[c_prt].dropna().astype(str).unique().tolist())
        world_vals = st.multiselect("Valores equivalentes a 'World':", partners_list, default=[p for p in partners_list if "world" in p.lower() or "mundo" in p.lower()])

        reporters_list = sorted(df_ct[c_rep].dropna().astype(str).unique().tolist())
        sel_reporters = st.multiselect("Filtrar Países Reporters (Vazio = Todos):", reporters_list, default=[])

        if st.button("Processar Métricas Comtrade", type="primary"):
            desc_col = None if c_desc == "(Nenhuma)" else c_desc
            tidy_ct = standardize_comtrade(df_ct, c_yr, c_rep, c_prt, c_sh6, desc_col, c_val, world_vals, sel_reporters)
            st.session_state["comtrade_tidy"] = tidy_ct
            st.success("Dados do UN Comtrade processados com sucesso!")

# ==============================================================================
# 7. MÓDULOS DA APLICAÇÃO (LAYOUTS REFINADOS COM GLASS & PASTEL)
# ==============================================================================

# --- PÁGINA 1: UN COMTRADE RCA / RSCA ---
def page_comtrade_global():
    st.title("📊 Análise de RCA e RSCA Global (UN Comtrade)")
    st.caption("Cálculo direto de Vantagem Comparativa Revelada (Balassa) e Vantagem Comparativa Revelada Simétrica (Laursen).")

    if "comtrade_tidy" not in st.session_state:
        st.info("👈 Por favor, carregue e processe o arquivo do UN Comtrade na barra lateral.")
        return

    tidy = st.session_state["comtrade_tidy"]
    anos = sorted(tidy["ano"].dropna().unique().tolist())
    
    col_a, col_b = st.columns(2)
    with col_a:
        start_year = st.selectbox("Ano de Início", anos, index=0)
    with col_b:
        end_year = st.selectbox("Ano Final", anos, index=len(anos)-1)

    df_metrics = compute_comtrade_metrics(tidy, start_year, end_year)

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        reps = st.multiselect("Filtrar por Reporter (País):", sorted(df_metrics["reporter"].unique()))
    with f_col2:
        sh6s = st.multiselect("Filtrar por SH6:", sorted(df_metrics["sh6"].unique()))
    with f_col3:
        only_advantage = st.checkbox("Apenas com Vantagem Comparativa (RCA >= 1)")

    filtered_df = df_metrics.copy()
    if reps:
        filtered_df = filtered_df[filtered_df["reporter"].isin(reps)]
    if sh6s:
        filtered_df = filtered_df[filtered_df["sh6"].isin(sh6s)]
    if only_advantage:
        filtered_df = filtered_df[filtered_df["rca"] >= 1.0]

    # Glass Stat Banner
    html_stats = f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-value">{len(filtered_df):,}</div>
            <div class="stat-label">Registros Processados</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{format_num(filtered_df['rca'].mean(), 2)}</div>
            <div class="stat-label">Média do RCA (Balassa)</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{format_num(filtered_df['rsca'].mean(), 2)}</div>
            <div class="stat-label">Média do RSCA (Laursen)</div>
        </div>
    </div>
    """
    st.markdown(html_stats, unsafe_allow_html=True)

    st.subheader("Resultados Detalhados")
    st.dataframe(
        filtered_df.style.format({
            "valor": "${:,.2f}",
            "mundo_valor": "${:,.2f}",
            "rca": "{:.4f}",
            "rsca": "{:.4f}"
        }),
        use_container_width=True, height=400
    )

    fig = px.scatter(
        filtered_df, x="rca", y="rsca", color="reporter", hover_name="sh6_desc",
        title="Distribuição entre RCA (Balassa) e RSCA (Laursen)",
        labels={"rca": "Índice RCA", "rsca": "Índice RSCA (-1 a +1)"},
        template="plotly_white"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.add_hline(y=0, line_dash="dash", line_color=PASTEL_COLORS["red_main"], annotation_text="Ponto Neutro (RSCA = 0)")
    st.plotly_chart(fig, use_container_width=True)


# --- PÁGINA 2: CRUZAMENTO BRASIL COMEXSTAT X COMTRADE ---
def page_comexstat_cross():
    st.title("🇧🇷 Cruzamento das Exportações do Brasil com Competitividade Global")
    st.caption("Cruzamento entre a pauta detalhada do ComexStat e os índices globais de RCA/RSCA.")

    if "comexstat" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ É necessário carregar AMBOS os arquivos (ComexStat Brasil e UN Comtrade) na barra lateral.")
        return

    cs = st.session_state["comexstat"]
    ct = st.session_state["comtrade_tidy"]

    anos_cs = sorted(cs["ano"].dropna().unique().tolist())
    selected_year = st.selectbox("Selecione o Ano para Análise Cruzada:", anos_cs, index=len(anos_cs)-1)

    ct_metrics = compute_comtrade_metrics(ct, selected_year, selected_year)
    
    brazil_rep = next((r for r in ct_metrics["reporter"].unique() if str(r).lower() in ["brazil", "brasil", "bra"]), ct_metrics["reporter"].unique()[0])
    br_ct = ct_metrics[ct_metrics["reporter"] == brazil_rep].copy()

    merged = pd.merge(cs[cs["ano"] == selected_year], br_ct[["sh6", "rca", "rsca", "mundo_valor"]], left_on="sh6_cod", right_on="sh6", how="left")
    merged["rca"] = merged["rca"].fillna(0)
    merged["rsca"] = merged["rsca"].fillna(-1)

    st.subheader("🔍 Filtros de Segmentação da Pauta Exportadora")
    col1, col2, col3 = st.columns(3)
    with col1:
        f_sh6 = st.multiselect("SH6:", sorted(merged["sh6_cod"].dropna().unique()))
        f_cuci = st.multiselect("CUCI Grupo:", sorted(merged["cuci_desc"].dropna().unique()) if "cuci_desc" in merged.columns else [])
    with col2:
        f_isic_div = st.multiselect("ISIC Divisão:", sorted(merged["isic_div_desc"].dropna().unique()) if "isic_div_desc" in merged.columns else [])
        f_isic_sec = st.multiselect("ISIC Seção:", sorted(merged["isic_sec_desc"].dropna().unique()) if "isic_sec_desc" in merged.columns else [])
    with col3:
        f_cgce1 = st.multiselect("CGCE Nível 1:", sorted(merged["cgce1_desc"].dropna().unique()) if "cgce1_desc" in merged.columns else [])
        f_cgce2 = st.multiselect("CGCE Nível 2:", sorted(merged["cgce2_desc"].dropna().unique()) if "cgce2_desc" in merged.columns else [])

    df_f = merged.copy()
    if f_sh6: df_f = df_f[df_f["sh6_cod"].isin(f_sh6)]
    if f_cuci and "cuci_desc" in df_f.columns: df_f = df_f[df_f["cuci_desc"].isin(f_cuci)]
    if f_isic_div and "isic_div_desc" in df_f.columns: df_f = df_f[df_f["isic_div_desc"].isin(f_isic_div)]
    if f_isic_sec and "isic_sec_desc" in df_f.columns: df_f = df_f[df_f["isic_sec_desc"].isin(f_isic_sec)]
    if f_cgce1 and "cgce1_desc" in df_f.columns: df_f = df_f[df_f["cgce1_desc"].isin(f_cgce1)]
    if f_cgce2 and "cgce2_desc" in df_f.columns: df_f = df_f[df_f["cgce2_desc"].isin(f_cgce2)]

    val_tot = df_f["valor_fob"].sum()
    produtos_vantagem = df_f[df_f["rca"] >= 1.0]["sh6_cod"].nunique()
    
    html_stats = f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-value">{format_usd(val_tot)}</div>
            <div class="stat-label">Valor Total Exportado (FOB)</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{df_f['sh6_cod'].nunique():,}</div>
            <div class="stat-label">Total de Produtos SH6</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color:{PASTEL_COLORS['green_main']};">{produtos_vantagem:,}</div>
            <div class="stat-label">Produtos com RCA ≥ 1</div>
        </div>
    </div>
    """
    st.markdown(html_stats, unsafe_allow_html=True)

    st.subheader("📊 Distribuição de Exportação x Competitividade por Setor")
    group_opt = st.selectbox("Agrupar Visualização por:", ["CUCI Grupo", "ISIC Divisão", "ISIC Seção", "CGCE Nível 1", "CGCE Nível 2"])

    col_map = {
        "CUCI Grupo": "cuci_desc",
        "ISIC Divisão": "isic_div_desc",
        "ISIC Seção": "isic_sec_desc",
        "CGCE Nível 1": "cgce1_desc",
        "CGCE Nível 2": "cgce2_desc"
    }
    selected_col = col_map[group_opt]

    if selected_col in df_f.columns:
        agg_sector = df_f.groupby(selected_col).agg(
            Valor_FOB=("valor_fob", "sum"),
            RCA_Medio=("rca", "mean"),
            RSCA_Medio=("rsca", "mean"),
            N_Produtos=("sh6_cod", "nunique")
        ).reset_index().sort_values("Valor_FOB", ascending=False)

        fig_sec = px.bar(
            agg_sector.head(15), x="Valor_FOB", y=selected_col, orientation="h",
            color="RSCA_Medio", color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            title=f"Top 15 Setores por Valor Exportado ({group_opt}) e RSCA Médio",
            labels={"Valor_FOB": "Valor FOB (US$)", selected_col: "Setor", "RSCA_Medio": "RSCA Médio"},
            template="plotly_white"
        )
        fig_sec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sec, use_container_width=True)

        st.dataframe(agg_sector.style.format({"Valor_FOB": "${:,.2f}", "RCA_Medio": "{:.2f}", "RSCA_Medio": "{:.2f}"}), use_container_width=True)


# --- PÁGINA 3: POTENCIAL DE DIVERSIFICAÇÃO POR ESTADO (UF) ---
def page_state_diversification():
    st.title("🗺️ Potencial de Diversificação Exportadora por Estado (UF)")
    st.caption("Cruzamento subnacional para identificar produtos com alta competitividade nacional (RCA >= 1) subaproveitados nos estados.")

    if "comexstat_uf" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ É necessário carregar a planilha do ComexStat por Estado (UF) e o UN Comtrade.")
        return

    cs_uf = st.session_state["comexstat_uf"]
    ct = st.session_state["comtrade_tidy"]

    anos_uf = sorted(cs_uf["ano"].dropna().unique().tolist())
    selected_year = st.selectbox("Ano de Análise Subnacional:", anos_uf, index=len(anos_uf)-1)

    ct_metrics = compute_comtrade_metrics(ct, selected_year, selected_year)

    df_potencial = compute_state_diversification_potentials(cs_uf, ct_metrics, selected_year)

    if df_potencial.empty:
        st.error("Não foi possível calcular o potencial com os dados fornecidos.")
        return

    # --- QUADRANTES DE DIVERSIFICAÇÃO ESTADUAL (INSPIRAÇÃO IMAGEM 2 - CNI) ---
    st.subheader("🧭 Radar de Oportunidades Estaduais (Matriz de Vantagem & Presença)")
    
    n_alto_potencial = len(df_potencial[df_potencial["potencial_score"] > 1.0])
    n_vantagem_nacional = len(df_potencial[df_potencial["rca"] >= 1.0])
    val_mercado_oportunidade = df_potencial[df_potencial["potencial_score"] > 1.0]["mundo_valor"].sum()

    html_quadrants = f"""
    <div class="quadrant-grid">
        <div class="quadrant-card" style="background:{PASTEL_COLORS['green_bg']}; border-color:rgba(16, 185, 129, 0.3);">
            <div>
                <div class="qc-title" style="color:{PASTEL_COLORS['green_main']};">
                    <span class="qc-dot" style="background:{PASTEL_COLORS['green_main']};"></span>Alta Oportunidade Local
                </div>
                <div class="qc-desc">Produtos onde o Brasil possui vantagem comparativa global (RCA ≥ 1), mas o estado possui produção/exportação nula ou inexpressiva.</div>
            </div>
            <div class="qc-bottom">
                <div>
                    <div class="qc-count">{n_alto_potencial:,}</div>
                    <div class="qc-count-label">produtos chave</div>
                </div>
                <div>
                    <div class="qc-value">{format_usd(val_mercado_oportunidade)}</div>
                    <div class="qc-pill" style="background:rgba(16, 185, 129, 0.15); color:{PASTEL_COLORS['green_main']};">Demanda Global</div>
                </div>
            </div>
        </div>

        <div class="quadrant-card" style="background:{PASTEL_COLORS['blue_bg']}; border-color:rgba(59, 130, 246, 0.3);">
            <div>
                <div class="qc-title" style="color:{PASTEL_COLORS['blue_main']};">
                    <span class="qc-dot" style="background:{PASTEL_COLORS['blue_main']};"></span>Vantagem Comparativa Nacional
                </div>
                <div class="qc-desc">Total de produtos no portfólio brasileiro com alto índice de especialização e inserção no comércio internacional.</div>
            </div>
            <div class="qc-bottom">
                <div>
                    <div class="qc-count">{n_vantagem_nacional:,}</div>
                    <div class="qc-count-label">produtos competitivos</div>
                </div>
                <div>
                    <div class="qc-value">RCA ≥ 1.0</div>
                    <div class="qc-pill" style="background:rgba(59, 130, 246, 0.15); color:{PASTEL_COLORS['blue_main']};">Base Brasil</div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_quadrants, unsafe_allow_html=True)

    st.subheader("🏆 Ranking de Estados por Score de Potencial de Diversificação")
    
    rank_uf = df_potencial.groupby("uf").agg(
        Score_Potencial_Total=("potencial_score", "sum"),
        Oportunidades_SH6=("sh6_cod", "nunique"),
        Exportacao_Atual_FOB=("valor_fob", "sum")
    ).reset_index().sort_values("Score_Potencial_Total", ascending=False)

    fig_uf_rank = px.bar(
        rank_uf[rank_uf["uf"] != "Sem Exportação Registrada"], x="Score_Potencial_Total", y="uf", orientation="h",
        color="Exportacao_Atual_FOB", color_continuous_scale=["#eff6ff", "#3b82f6", "#1d4ed8"],
        title="Estados com Maior Potencial de Diversificação",
        template="plotly_white"
    )
    fig_uf_rank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_uf_rank, use_container_width=True)

    st.divider()

    st.subheader("🔍 Detalhamento e Segmentação por Estado (UF)")
    
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        uf_target = st.selectbox("Selecione o Estado (UF):", sorted(df_potencial["uf"].unique()))
    
    df_uf_filtered = df_potencial[df_potencial["uf"] == uf_target]

    with col_sel2:
        cuci_options = sorted(df_uf_filtered["cuci_desc"].dropna().unique()) if "cuci_desc" in df_uf_filtered.columns else []
        cuci_target = st.multiselect("Filtrar por CUCI Grupo:", cuci_options)

    with col_sel3:
        only_new = st.checkbox("Exibir Apenas Produtos NÃO Exportados Atualmente pelo Estado", value=False)

    df_uf_seg = df_uf_filtered.copy()
    if cuci_target and "cuci_desc" in df_uf_seg.columns:
        df_uf_seg = df_uf_seg[df_uf_seg["cuci_desc"].isin(cuci_target)]
    if only_new:
        df_uf_seg = df_uf_seg[df_uf_seg["valor_fob"] == 0]

    st.markdown(f"### Oportunidades Prioritárias para **{uf_target}**")

    disp_cols = ["sh6_cod", "sh6_desc", "cuci_desc", "rca", "rsca", "valor_fob", "potencial_score"]
    available_disp_cols = [c for c in disp_cols if c in df_uf_seg.columns]

    st.dataframe(
        df_uf_seg[available_disp_cols].head(30).style.format({
            "rca": "{:.2f}",
            "rsca": "{:.2f}",
            "valor_fob": "${:,.2f}",
            "potencial_score": "{:.2f}"
        }),
        use_container_width=True, height=400
    )


# ==============================================================================
# 8. ROTEADOR DE PÁGINAS
# ==============================================================================

if PAGE == "📊 RCA/RSCA Global (UN Comtrade)":
    page_comtrade_global()
elif PAGE == "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)":
    page_comexstat_cross()
elif PAGE == "🗺️ Potencial de Diversificação por Estado (UF)":
    page_state_diversification()
