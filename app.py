# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS
 (Cupertino Executive Edition) — Fixed Upload Layout & Metrics
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
import streamlit as st

# ==============================================================================
# 0. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Diversificação das Exportações Brasileiras",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
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
    return re.sub(r"_+", "_", s).strip("_")


def is_world_label(value) -> bool:
    nv = normalize_text(value)
    return nv in ("world", "mundo") or nv.startswith("world") or nv.startswith("mundo")


def detect_world_label(values) -> str | None:
    for v in values:
        if is_world_label(v):
            return v
    return None


@st.cache_data(show_spinner=False)
def read_any_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    name = file_name.lower()
    raw = file_bytes

    if name.endswith((".csv", ".txt")):
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

    if name.endswith((".xlsx", ".xls")):
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


def find_column(df: pd.DataFrame, keywords: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    norm_map = {c: normalize_text(c) for c in df.columns}
    for col, norm in norm_map.items():
        if any(ex in norm for ex in exclude):
            continue
        for kw in keywords:
            if kw in norm:
                return col
    return None


def format_usd(x: float | int | None) -> str:
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


def format_num(x: float | int | None, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


# ==============================================================================
# 2. PADRONIZAÇÃO COMEXSTAT & COMTRADE
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

_COMEXSTAT_FIELD_ORDER = list(COMEXSTAT_FIELD_KEYWORDS.keys())
COMEXSTAT_DESC_COLUMNS = [
    "sh6_desc", "cuci_cod", "cuci_desc",
    "isic_div_desc", "isic_sec_desc", "cgce1_desc", "cgce2_desc",
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
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)

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
        df["valor_fob"] = df["valor_fob"].fillna(0.0)

    if "sh6_cod" in df.columns:
        sh_clean = pd.to_numeric(df["sh6_cod"], errors="coerce").fillna(0).astype(int).astype(str)
        df["sh6_cod"] = sh_clean.str.zfill(6)

    if "uf" in df.columns:
        df["uf"] = df["uf"].astype(str).str.strip().str.upper()

    return df


def standardize_comtrade(
    df: pd.DataFrame,
    year_col: str,
    reporter_col: str,
    partner_col: str,
    sh6_col: str,
    sh6desc_col: str | None,
    value_col: str,
    selected_partners: list | None = None,
    selected_reporters: list | None = None,
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

    d["ano"] = pd.to_numeric(d["ano"], errors="coerce").fillna(0).astype(int)
    sh_clean = pd.to_numeric(d["sh6"], errors="coerce").fillna(0).astype(int).astype(str)
    d["sh6"] = sh_clean.str.zfill(6)
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce").fillna(0.0)

    if "sh6_desc" not in d.columns:
        d["sh6_desc"] = d["sh6"]

    if selected_partners:
        prt_str_list = [str(p).lower() for p in selected_partners]
        d = d[d["partner"].astype(str).str.lower().isin(prt_str_list)]

    if selected_reporters:
        rep_str_list = [str(r).lower() for r in selected_reporters]
        world_reporters_in_data = [r for r in d["reporter"].dropna().unique() if is_world_label(r)]
        keep_list = set(rep_str_list) | {str(r).lower() for r in world_reporters_in_data}
        d = d[d["reporter"].astype(str).str.lower().isin(keep_list)]

    return d.groupby(["ano", "reporter", "partner", "sh6", "sh6_desc"], as_index=False)["valor"].sum()


# ==============================================================================
# 3. CÁLCULO DE MÉTRICAS BILATERAIS E POTENCIAIS
# ==============================================================================

@st.cache_data(show_spinner="Calculando vantagens comparativas...")
def compute_comtrade_metrics(comtrade_tidy: pd.DataFrame, years: tuple) -> pd.DataFrame:
    years = tuple(sorted(set(int(y) for y in years)))
    df_filtered = comtrade_tidy[comtrade_tidy["ano"].isin(years)].copy()
    empty_res = pd.DataFrame(columns=["ano", "reporter", "partner", "sh6", "sh6_desc", "valor", "mundo_valor", "rca_partner", "rsca_partner"])
    
    if df_filtered.empty:
        return empty_res

    year_partner_frames = []

    for yr in years:
        df_yr = df_filtered[df_filtered["ano"] == yr]
        if df_yr.empty:
            continue

        for prt in df_yr["partner"].unique():
            df_prt = df_yr[df_yr["partner"] == prt]
            if df_prt.empty:
                continue

            pv = df_prt.pivot_table(index="sh6", columns="reporter", values="valor", aggfunc="sum", fill_value=0.0)
            sh6_descs = df_prt.groupby("sh6")["sh6_desc"].first()

            world_reporter = detect_world_label(pv.columns.tolist())
            uses_world_ref = world_reporter is not None and world_reporter in pv.columns

            if uses_world_ref:
                X_wj = pv[world_reporter]
                X_w = float(X_wj.sum())
                reporters = [r for r in pv.columns if r != world_reporter]
            else:
                X_wj = pv.sum(axis=1)
                X_w = float(pv.values.sum())
                reporters = list(pv.columns)

            if not reporters:
                continue

            pv_r = pv[reporters]
            X_i = pv_r.sum(axis=0)

            share_pais = pv_r.div(X_i.replace(0, np.nan), axis=1).fillna(0.0)
            share_mundo = (X_wj / X_w) if X_w > 0 else pd.Series(0.0, index=X_wj.index)

            rca_df = share_pais.div(share_mundo.replace(0, np.nan), axis=0)
            rca_df = rca_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            rsca_df = ((rca_df - 1) / (rca_df + 1)).fillna(-1.0)

            val_long = pv_r.stack().rename("valor").reset_index()
            val_long.columns = ["sh6", "reporter", "valor"]
            rca_long = rca_df.stack().rename("rca_partner").reset_index()
            rca_long.columns = ["sh6", "reporter", "rca_partner"]
            rsca_long = rsca_df.stack().rename("rsca_partner").reset_index()
            rsca_long.columns = ["sh6", "reporter", "rsca_partner"]

            merged = val_long.merge(rca_long, on=["sh6", "reporter"]).merge(rsca_long, on=["sh6", "reporter"])
            merged["ano"] = yr
            merged["partner"] = prt
            merged["sh6_desc"] = merged["sh6"].map(sh6_descs)
            merged["mundo_valor"] = merged["sh6"].map(X_wj)

            year_partner_frames.append(merged)

    if not year_partner_frames:
        return empty_res

    out = pd.concat(year_partner_frames, ignore_index=True)
    return out[["ano", "reporter", "partner", "sh6", "sh6_desc", "valor", "mundo_valor", "rca_partner", "rsca_partner"]]


def compute_state_diversification_potentials(
    comexstat_uf: pd.DataFrame, comtrade_metrics: pd.DataFrame, year: int
) -> pd.DataFrame:
    base_uf = comexstat_uf[comexstat_uf["ano"] == year].copy() if "ano" in comexstat_uf.columns else comexstat_uf.copy()

    required_cols = {"uf", "sh6_cod", "valor_fob"}
    if base_uf.empty or not required_cols.issubset(base_uf.columns):
        return pd.DataFrame()

    brazil_rep_name = next(
        (r for r in comtrade_metrics["reporter"].dropna().unique()
         if normalize_text(r) in ("brazil", "brasil", "bra")),
        None,
    )
    if brazil_rep_name is None:
        return pd.DataFrame()

    br_metrics = comtrade_metrics[
        (comtrade_metrics["reporter"] == brazil_rep_name) & (comtrade_metrics["ano"] == year)
    ].copy()
    if br_metrics.empty:
        return pd.DataFrame()

    advantage = br_metrics.groupby(["sh6", "sh6_desc"], as_index=False).agg(
        rca=("rca_partner", "mean"),
        rsca=("rsca_partner", "mean"),
        mundo_valor=("mundo_valor", "sum")
    )
    advantage = advantage[advantage["rca"] >= 1.0].copy()
    
    if advantage.empty:
        return pd.DataFrame()

    desc_cols = [c for c in COMEXSTAT_DESC_COLUMNS if c in base_uf.columns and c != "sh6_desc"]
    desc_map = base_uf.groupby("sh6_cod")[desc_cols].first() if desc_cols else pd.DataFrame(index=pd.Index([], name="sh6_cod"))

    ufs = sorted(base_uf["uf"].dropna().unique().tolist())
    sh6_advantage = advantage["sh6"].dropna().unique().tolist()
    if not ufs or not sh6_advantage:
        return pd.DataFrame()

    grid = pd.MultiIndex.from_product([ufs, sh6_advantage], names=["uf", "sh6_cod"]).to_frame(index=False)
    uf_totals = base_uf.groupby("uf")["valor_fob"].sum()
    grid["uf_total"] = grid["uf"].map(uf_totals).fillna(0.0)

    actual_exports = base_uf.groupby(["uf", "sh6_cod"], as_index=False)["valor_fob"].sum()
    grid = grid.merge(actual_exports, on=["uf", "sh6_cod"], how="left")
    grid["valor_fob"] = grid["valor_fob"].fillna(0.0)
    grid["share_local"] = np.where(grid["uf_total"] > 0, grid["valor_fob"] / grid["uf_total"], 0.0)

    adv_slim = advantage[["sh6", "sh6_desc", "rca", "rsca", "mundo_valor"]].rename(columns={"sh6": "sh6_cod"})
    grid = grid.merge(adv_slim, on="sh6_cod", how="left")

    if not desc_map.empty:
        grid = grid.merge(desc_map, on="sh6_cod", how="left")

    grid["potencial_score"] = grid["rca"] * (1 - grid["share_local"])
    grid["ja_exportado"] = grid["valor_fob"] > 0

    return grid.sort_values("potencial_score", ascending=False)


# ==============================================================================
# 4. DESIGN EXEC & STYLES (CLEAN CUPERTINO)
# ==============================================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        :root {
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }

        html, body, [class*="st-"] {
            font-family: -apple-system, BlinkMacSystemFont, "Plus Jakarta Sans", sans-serif;
        }

        .stApp {
            background: var(--bg);
            color: var(--text-primary);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }

        /* Headings */
        h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; font-size: 2.1rem !important; }
        h2 { font-weight: 750 !important; letter-spacing: -0.02em !important; font-size: 1.4rem !important; }
        h3 { font-weight: 700 !important; letter-spacing: -0.01em !important; font-size: 1.15rem !important; }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid var(--border) !important;
        }
        
        .sidebar-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.2rem;
        }

        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin: 1.2rem 0 0.5rem;
        }

        .sidebar-file-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 0.8rem;
            margin-bottom: 0.2rem;
        }

        /* KPI Cards */
        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: var(--shadow-sm);
            position: relative;
            overflow: hidden;
            height: 100%;
        }

        .metric-card .label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .metric-card .value-container {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .metric-card .value {
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.03em;
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.65rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .badge-green { background: #ecfdf5; color: #047857; }
        .badge-blue  { background: #eff6ff; color: #1d4ed8; }
        .badge-amber { background: #fffbeb; color: #b45309; }
        .badge-red   { background: #fef2f2; color: #b91c1c; }

        /* Container Cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: var(--shadow-sm) !important;
            padding: 1rem !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        }

        .checkbox-fix {
            margin-top: 1.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PASTEL_COLORS = {
    "green": "#10b981",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "red": "#ef4444",
}

# ==============================================================================
# 5. INICIALIZAÇÃO DE ESTADO E SIDEBAR
# ==============================================================================

inject_custom_css()

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌎 Navegação</div>', unsafe_allow_html=True)
    st.caption("Inteligência e Diversificação Comercial")

    PAGE = st.radio(
        "Módulo Principal",
        [
            "📊 RCA/RSCA Global (UN Comtrade)",
            "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)",
            "🗺️ Potencial de Diversificação por Estado (UF)",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown('<div class="sidebar-section-title">Bases de Dados</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-file-label">01 · UN Comtrade</div>', unsafe_allow_html=True)
    file_comtrade = st.file_uploader(
        "01 · UN Comtrade",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        label_visibility="collapsed",
        help="CSV, XLSX, Parquet ou JSON do UN Comtrade.",
    )

    st.markdown('<div class="sidebar-file-label">02 · ComexStat Nacional</div>', unsafe_allow_html=True)
    file_comexstat = st.file_uploader(
        "02 · ComexStat Nacional",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Base nacional ComexStat por código SH6.",
    )

    st.markdown('<div class="sidebar-file-label">03 · ComexStat por Estado (UF)</div>', unsafe_allow_html=True)
    file_comexstat_uf = st.file_uploader(
        "03 · ComexStat por Estado (UF)",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Base ComexStat dividida por UF e SH6.",
    )

# Processamento de Uploads na Session State
if file_comtrade and st.session_state.get("_file_comtrade_name") != file_comtrade.name:
    st.session_state["raw_comtrade"] = read_any_file(file_comtrade.getvalue(), file_comtrade.name)
    st.session_state["_file_comtrade_name"] = file_comtrade.name
    st.session_state.pop("comtrade_tidy", None)

if file_comexstat and st.session_state.get("_file_comexstat_name") != file_comexstat.name:
    raw_cs = read_any_file(file_comexstat.getvalue(), file_comexstat.name)
    st.session_state["comexstat"] = standardize_comexstat(raw_cs)
    st.session_state["_file_comexstat_name"] = file_comexstat.name

if file_comexstat_uf and st.session_state.get("_file_comexstat_uf_name") != file_comexstat_uf.name:
    raw_cs_uf = read_any_file(file_comexstat_uf.getvalue(), file_comexstat_uf.name)
    st.session_state["comexstat_uf"] = standardize_comexstat(raw_cs_uf)
    st.session_state["_file_comexstat_uf_name"] = file_comexstat_uf.name

# Processamento dinâmico de configurações Comtrade
if "raw_comtrade" in st.session_state:
    df_ct = st.session_state["raw_comtrade"]
    cols = list(df_ct.columns)

    col_yr = find_column(df_ct, ["refyear", "year", "ano"]) or "refYear"
    col_rep = find_column(df_ct, ["reporterdesc", "reporter"]) or "ReporterDesc"
    col_prt = find_column(df_ct, ["partnerdesc", "partner"]) or "PartnerDesc"
    col_sh6 = find_column(df_ct, ["cmdcode", "sh6"]) or "cmdCode"
    col_desc = find_column(df_ct, ["cmddesc", "sh6_desc"]) or "cmdDesc"

    with st.sidebar.expander("⚙️ Configurações Comtrade", expanded=False):
        val_candidates = [c for c in cols if any(v in normalize_text(c) for v in ["value", "val", "fob", "primaryvalue"])]
        c_val = st.selectbox("Campo de Valor", cols, index=cols.index(val_candidates[0]) if val_candidates else 0)

        partners_list = sorted(df_ct[col_prt].dropna().astype(str).unique()) if col_prt in df_ct.columns else []
        sel_partners = st.multiselect("Filtrar Parceiros", partners_list, default=[])

        reporters_list = sorted(df_ct[col_rep].dropna().astype(str).unique()) if col_rep in df_ct.columns else []
        sel_reporters = st.multiselect("Filtrar Declarantres", reporters_list, default=[])

        if st.button("Processar Base Comtrade", type="primary", use_container_width=True):
            try:
                st.session_state["comtrade_tidy"] = standardize_comtrade(
                    df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
                )
                st.success("Comtrade processado!")
            except Exception as exc:
                st.error(f"Erro ao padronizar: {exc}")

    # Processamento automático inicial
    if "comtrade_tidy" not in st.session_state and all(c in df_ct.columns for c in [col_yr, col_rep, col_prt, col_sh6]):
        try:
            st.session_state["comtrade_tidy"] = standardize_comtrade(
                df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
            )
        except Exception:
            pass


# ==============================================================================
# 6. MÓDULOS DE ANÁLISE
# ==============================================================================

# --- MÓDULO 1: UN COMTRADE RCA / RSCA ---
def page_comtrade_global():
    st.title("📊 Análise de Vantagens Comparativas (RCA / RSCA)")
    st.caption("Cálculo bilateral de vantagens comparativas reveladas (Balassa & Laursen)")

    if "comtrade_tidy" not in st.session_state:
        st.info("👈 Por favor, carregue e processe a base UN Comtrade na barra lateral.")
        return

    tidy = st.session_state["comtrade_tidy"]
    anos = sorted(int(a) for a in tidy["ano"].dropna().unique())
    if not anos:
        st.error("Nenhum ano válido encontrado nos dados.")
        return

    with st.container(border=True):
        st.markdown("**🔍 Filtros da Análise**")
        c1, c2, c3, c4 = st.columns([1, 1, 1.5, 1.5])
        with c1:
            start_year = st.selectbox("Ano Inicial", anos, index=0)
        with c2:
            end_year = st.selectbox("Ano Final", anos, index=len(anos) - 1)

        years_range = [y for y in anos if start_year <= y <= end_year] or [start_year]
        df_metrics = compute_comtrade_metrics(tidy, tuple(years_range))

        if df_metrics.empty:
            st.warning("Sem dados para o período selecionado.")
            return

        with c3:
            reps = st.multiselect("Declarante (Reporter)", sorted(df_metrics["reporter"].unique()))
        with c4:
            prts = st.multiselect("Partner (Parceiro)", sorted(df_metrics["partner"].unique()))

        c5, c6 = st.columns([3, 1])
        with c5:
            sh6s = st.multiselect("Código SH6", sorted(df_metrics["sh6"].unique()))
        with c6:
            st.markdown('<div class="checkbox-fix">', unsafe_allow_html=True)
            only_adv = st.checkbox("Apenas RCA ≥ 1.0")
            st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df_metrics.copy()
    if reps: filtered_df = filtered_df[filtered_df["reporter"].isin(reps)]
    if prts: filtered_df = filtered_df[filtered_df["partner"].isin(prts)]
    if sh6s: filtered_df = filtered_df[filtered_df["sh6"].isin(sh6s)]
    if only_adv: filtered_df = filtered_df[filtered_df["rca_partner"] >= 1.0]

    # KPIs Executivos
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Registros Selecionados</div>
                <div class="value-container">
                    <div class="value">{len(filtered_df):,}</div>
                    <span class="badge badge-blue">Tidy Dataset</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with k2:
        rca_avg = filtered_df['rca_partner'].mean()
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Média RCA Bilateral</div>
                <div class="value-container">
                    <div class="value">{format_num(rca_avg, 2)}</div>
                    <span class="badge badge-green">Balassa</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with k3:
        rsca_avg = filtered_df['rsca_partner'].mean()
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Média RSCA Simétrico</div>
                <div class="value-container">
                    <div class="value">{format_num(rsca_avg, 2)}</div>
                    <span class="badge badge-amber">Laursen</span>
                </div>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("---")
    
    partner_title = prts[0] if len(prts) == 1 else "Parceiro Selecionado"
    rca_col_name = f"RCA ({partner_title})"
    rsca_col_name = f"RSCA ({partner_title})"

    display_df = filtered_df.rename(columns={
        "rca_partner": rca_col_name,
        "rsca_partner": rsca_col_name
    })

    st.dataframe(
        display_df,
        column_config={
            "valor": st.column_config.NumberColumn("Valor (US$)", format="$ %,.2f"),
            "mundo_valor": st.column_config.NumberColumn("Total Mundo (US$)", format="$ %,.2f"),
            rca_col_name: st.column_config.NumberColumn(format="%.4f"),
            rsca_col_name: st.column_config.NumberColumn(format="%.4f"),
            "ano": st.column_config.NumberColumn("Ano", format="%d"),
        },
        use_container_width=True,
        height=350,
    )

    st.subheader("📈 Distribuições e Desempenho")
    g1, g2 = st.columns(2)

    with g1:
        fig_rca = px.histogram(
            filtered_df, x="rca_partner", color="partner",
            hover_data=["sh6_desc", "reporter"],
            title="Distribuição de RCA por Parceiro",
            labels={"rca_partner": "Índice RCA", "partner": "Parceiro"},
            template="plotly_white", nbins=30
        )
        fig_rca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_rca.add_vline(x=1.0, line_dash="dash", line_color=PASTEL_COLORS["blue"])
        st.plotly_chart(fig_rca, use_container_width=True)

    with g2:
        fig_rsca = px.box(
            filtered_df, x="partner", y="rsca_partner", color="partner",
            hover_data=["sh6_desc", "reporter"],
            title="Simetria do RSCA (-1 a +1)",
            labels={"rsca_partner": "Índice RSCA", "partner": "Parceiro"},
            template="plotly_white"
        )
        fig_rsca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_rsca.add_hline(y=0, line_dash="dash", line_color=PASTEL_COLORS["red"])
        st.plotly_chart(fig_rsca, use_container_width=True)


# --- MÓDULO 2: CRUZAMENTO COMEXSTAT X COMTRADE (NOVOS CARTÕES + SCATTER) ---
def page_comexstat_cross():
    st.title("🇧🇷 Cruzamento Pauta Brasil x Competitividade Global")
    st.caption("Pauta nacional ComexStat alinhada às vantagens globais calculadas via Comtrade")

    if "comexstat" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ Carregue as bases ComexStat Brasil e UN Comtrade na barra lateral.")
        return

    cs = st.session_state["comexstat"]
    ct = st.session_state["comtrade_tidy"]

    anos_cs = sorted(int(a) for a in cs["ano"].dropna().unique())
    selected_year = st.selectbox("Ano de Referência:", anos_cs, index=len(anos_cs) - 1)

    ct_metrics = compute_comtrade_metrics(ct, (selected_year,))
    if ct_metrics.empty:
        st.warning("Sem dados do Comtrade para o ano selecionado.")
        return

    brazil_rep = next((r for r in ct_metrics["reporter"].unique() if normalize_text(r) in ("brazil", "brasil", "bra")), None)
    if not brazil_rep:
        st.error("País 'Brazil' não identificado nos dados do Comtrade.")
        return

    br_ct = ct_metrics[ct_metrics["reporter"] == brazil_rep].groupby("sh6").agg(
        rca=("rca_partner", "mean"),
        rsca=("rsca_partner", "mean"),
        mundo_valor=("mundo_valor", "sum")
    ).reset_index()

    merged = pd.merge(cs[cs["ano"] == selected_year], br_ct, left_on="sh6_cod", right_on="sh6", how="left")
    merged["rca"] = merged["rca"].fillna(0)
    merged["rsca"] = merged["rsca"].fillna(-1)

    with st.container(border=True):
        st.markdown("**🔍 Filtros Setoriais**")
        c1, c2, c3 = st.columns(3)
        with c1:
            f_sh6 = st.multiselect("SH6", sorted(merged["sh6_cod"].dropna().unique()))
            f_cuci = st.multiselect("CUCI Grupo", sorted(merged["cuci_desc"].dropna().unique()) if "cuci_desc" in merged.columns else [])
        with c2:
            f_isic_div = st.multiselect("ISIC Divisão", sorted(merged["isic_div_desc"].dropna().unique()) if "isic_div_desc" in merged.columns else [])
            f_isic_sec = st.multiselect("ISIC Seção", sorted(merged["isic_sec_desc"].dropna().unique()) if "isic_sec_desc" in merged.columns else [])
        with c3:
            f_cgce1 = st.multiselect("CGCE Nível 1", sorted(merged["cgce1_desc"].dropna().unique()) if "cgce1_desc" in merged.columns else [])
            f_cgce2 = st.multiselect("CGCE Nível 2", sorted(merged["cgce2_desc"].dropna().unique()) if "cgce2_desc" in merged.columns else [])

    df_f = merged.copy()
    if f_sh6: df_f = df_f[df_f["sh6_cod"].isin(f_sh6)]
    if f_cuci and "cuci_desc" in df_f.columns: df_f = df_f[df_f["cuci_desc"].isin(f_cuci)]
    if f_isic_div and "isic_div_desc" in df_f.columns: df_f = df_f[df_f["isic_div_desc"].isin(f_isic_div)]
    if f_isic_sec and "isic_sec_desc" in df_f.columns: df_f = df_f[df_f["isic_sec_desc"].isin(f_isic_sec)]
    if f_cgce1 and "cgce1_desc" in df_f.columns: df_f = df_f[df_f["cgce1_desc"].isin(f_cgce1)]
    if f_cgce2 and "cgce2_desc" in df_f.columns: df_f = df_f[df_f["cgce2_desc"].isin(f_cgce2)]

    # CÁLCULOS DAS NOVAS MÉTRICAS SOLICITADAS
    sh6_rca_rsca_alto = df_f[(df_f["rca"] > 1.0) & (df_f["rsca"] > 0.0)]["sh6_cod"].nunique()
    sh6_rca_rsca_baixo = df_f[(df_f["rca"] < 1.0) & (df_f["rsca"] < 0.0)]["sh6_cod"].nunique()

    # Painel de 5 Cartões Executivos
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Exportado (FOB)</div>
                <div class="value-container">
                    <div class="value">{format_usd(df_f['valor_fob'].sum())}</div>
                    <span class="badge badge-blue">ComexStat</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Produtos Monitorados</div>
                <div class="value-container">
                    <div class="value">{df_f['sh6_cod'].nunique():,}</div>
                    <span class="badge badge-amber">SH6</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with m3:
        produtos_adv = df_f[df_f["rca"] >= 1.0]["sh6_cod"].nunique()
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Produtos Competitivos</div>
                <div class="value-container">
                    <div class="value">{produtos_adv:,}</div>
                    <span class="badge badge-green">RCA ≥ 1.0</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">SH6 Vantagem Comparativa</div>
                <div class="value-container">
                    <div class="value">{sh6_rca_rsca_alto:,}</div>
                    <span class="badge badge-green">RCA &gt; 1 · RSCA &gt; 0</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">SH6 Desvantagem Comparativa</div>
                <div class="value-container">
                    <div class="value">{sh6_rca_rsca_baixo:,}</div>
                    <span class="badge badge-red">RCA &lt; 1 · RSCA &lt; 0</span>
                </div>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("---")

    group_opt = st.selectbox("Agrupar Visualização por:", ["CUCI Grupo", "ISIC Divisão", "ISIC Seção", "CGCE Nível 1", "CGCE Nível 2"])
    col_map = {
        "CUCI Grupo": "cuci_desc", "ISIC Divisão": "isic_div_desc",
        "ISIC Seção": "isic_sec_desc", "CGCE Nível 1": "cgce1_desc", "CGCE Nível 2": "cgce2_desc"
    }
    selected_col = col_map[group_opt]

    if selected_col in df_f.columns:
        agg_sector = df_f.groupby(selected_col).agg(
            Valor_FOB=("valor_fob", "sum"),
            RCA_Medio=("rca", "mean"),
            RSCA_Medio=("rsca", "mean"),
            N_Produtos=("sh6_cod", "nunique"),
        ).reset_index().sort_values("Valor_FOB", ascending=False)

        c_sec1, c_sec2 = st.columns(2)
        with c_sec1:
            fig = px.scatter(
                agg_sector, 
                x="Valor_FOB", 
                y="RCA_Medio", 
                size="N_Produtos",
                color="RCA_Medio",
                hover_name=selected_col,
                color_continuous_scale=["#3b82f6", "#10b981"],
                title=f"Dispersão: Valor FOB x RCA Médio ({group_opt})",
                labels={"Valor_FOB": "Valor FOB (US$)", "RCA_Medio": "RCA Médio", "N_Produtos": "Qtd Produtos"},
                template="plotly_white"
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color=PASTEL_COLORS["amber"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with c_sec2:
            st.dataframe(
                agg_sector,
                column_config={
                    "Valor_FOB": st.column_config.NumberColumn("Valor FOB (US$)", format="$ %,.2f"),
                    "RCA_Medio": st.column_config.NumberColumn("RCA Médio", format="%.2f"),
                    "RSCA_Medio": st.column_config.NumberColumn("RSCA Médio", format="%.2f"),
                },
                use_container_width=True,
                height=380,
            )


# --- MÓDULO 3: DIVERSIFICAÇÃO POR ESTADO (SCATTER PLOTS) ---
def page_state_diversification():
    st.title("🗺️ Potencial de Diversificação por Estado (UF)")
    st.caption("Oportunidades subnacionais baseadas na pauta local e vantagens nacionais")

    if "comexstat_uf" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ Carregue a base do ComexStat por Estado (UF) e o UN Comtrade.")
        return

    cs_uf = st.session_state["comexstat_uf"]
    ct = st.session_state["comtrade_tidy"]

    anos_uf = sorted(int(a) for a in cs_uf["ano"].dropna().unique())
    selected_year = st.selectbox("Ano de Análise:", anos_uf, index=len(anos_uf) - 1)

    ct_metrics = compute_comtrade_metrics(ct, (selected_year,))
    df_potencial = compute_state_diversification_potentials(cs_uf, ct_metrics, selected_year)

    if df_potencial.empty:
        st.error("Não foi possível calcular os potenciais. Verifique a compatibilidade dos dados.")
        return

    high_pot = df_potencial[(df_potencial["potencial_score"] > 1.0) & (~df_potencial["ja_exportado"])]
    
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Oportunidades de Alta Prioridade (Não Exportadas)</div>
                <div class="value-container">
                    <div class="value">{high_pot['sh6_cod'].nunique():,}</div>
                    <span class="badge badge-green">Score > 1.0</span>
                </div>
            </div>""", unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Portfólio com Vantagem Nacional</div>
                <div class="value-container">
                    <div class="value">{df_potencial['sh6_cod'].nunique():,}</div>
                    <span class="badge badge-blue">RCA Brasil ≥ 1.0</span>
                </div>
            </div>""", unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("🏆 Matriz de Potencial por UF")
    rank_uf = df_potencial.groupby("uf").agg(
        Score_Potencial_Total=("potencial_score", "sum"),
        Produtos_Nao_Explorados=("ja_exportado", lambda s: int((~s).sum())),
        Exportacao_Atual_FOB=("uf_total", "first"),
    ).reset_index().sort_values("Score_Potencial_Total", ascending=False)

    fig_rank = px.scatter(
        rank_uf, 
        x="Exportacao_Atual_FOB", 
        y="Score_Potencial_Total",
        size="Produtos_Nao_Explorados",
        color="Score_Potencial_Total",
        text="uf",
        color_continuous_scale="Blues",
        title="Dispersão: Exportação Atual (US$) x Score de Potencial Total",
        labels={
            "Exportacao_Atual_FOB": "Exportação Atual (US$)", 
            "Score_Potencial_Total": "Score de Potencial Total",
            "Produtos_Nao_Explorados": "Prod. Não Explorados"
        },
        template="plotly_white"
    )
    fig_rank.update_traces(textposition='top center')
    fig_rank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("🔍 Oportunidades por Estado (UF)")
    uf_target = st.selectbox("Selecione o Estado:", sorted(df_potencial["uf"].unique()))
    df_uf_filtered = df_potencial[df_potencial["uf"] == uf_target]

    only_new = st.checkbox("Mostrar apenas produtos NÃO exportados pela UF", value=True)
    df_uf_seg = df_uf_filtered[~df_uf_filtered["ja_exportado"]] if only_new else df_uf_filtered

    top_n = df_uf_seg.sort_values("potencial_score", ascending=False).head(20)
    if not top_n.empty:
        label_col = "sh6_desc" if "sh6_desc" in top_n.columns and top_n["sh6_desc"].notna().any() else "sh6_cod"
        fig_top = px.scatter(
            top_n, 
            x="rca", 
            y="potencial_score", 
            size="mundo_valor",
            color="potencial_score",
            hover_name=label_col,
            title=f"Top Produtos por RCA e Score de Potencial — {uf_target}",
            labels={"rca": "RCA Brasil", "potencial_score": "Score de Potencial", "mundo_valor": "Demanda Global"},
            template="plotly_white"
        )
        fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_top, use_container_width=True)

    disp_cols = ["sh6_cod", "sh6_desc", "cuci_desc", "rca", "valor_fob", "share_local", "potencial_score"]
    cols_to_show = [c for c in disp_cols if c in df_uf_seg.columns]

    st.dataframe(
        df_uf_seg[cols_to_show].sort_values("potencial_score", ascending=False),
        column_config={
            "rca": st.column_config.NumberColumn("RCA Brasil", format="%.2f"),
            "valor_fob": st.column_config.NumberColumn("Valor UF (US$)", format="$ %,.2f"),
            "share_local": st.column_config.NumberColumn("Share na UF", format="%.2%%"),
            "potencial_score": st.column_config.NumberColumn("Score Potencial", format="%.2f"),
        },
        use_container_width=True,
        height=400,
    )


# ==============================================================================
# 7. ROTEAMENTO
# ==============================================================================

if PAGE == "📊 RCA/RSCA Global (UN Comtrade)":
    page_comtrade_global()
elif PAGE == "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)":
    page_comexstat_cross()
elif PAGE == "🗺️ Potencial de Diversificação por Estado (UF)":
    page_state_diversification()
