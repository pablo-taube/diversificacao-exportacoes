# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS
 (Glassmorfismo Cupertino) — v7.0 (RCA Consolidado Global x Bilateral)
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
)

# ==============================================================================
# 1. UTILITÁRIOS GERAIS
# ==============================================================================

def normalize_text(s) -> str:
    """Normaliza texto: remove acentos, minúsculas, troca não-alfanuméricos por '_'."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def is_world_label(value) -> bool:
    """Detecta se um rótulo de país representa o agregado 'Mundo'/'World'."""
    nv = normalize_text(value)
    return nv in ("world", "mundo") or nv.startswith("world") or nv.startswith("mundo")


def detect_world_label(values) -> str | None:
    """Retorna o primeiro valor de uma lista que representa 'World'/'Mundo', se existir."""
    for v in values:
        if is_world_label(v):
            return v
    return None


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


# ==============================================================================
# 3. PADRONIZAÇÃO COMTRADE
# ==============================================================================

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

    agg = d.groupby(["ano", "reporter", "partner", "sh6", "sh6_desc"], as_index=False)["valor"].sum()
    return agg


# ==============================================================================
# 4. MOTOR DE CÁLCULO DUAL (SISTEMA DE DUAS CAMADAS: GLOBAL SH6 x BILATERAL)
# ==============================================================================

@st.cache_data(show_spinner="Calculando indicadores globais consolidados e bilaterais...")
def compute_comtrade_dual_metrics(comtrade_tidy: pd.DataFrame, years: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna uma tupla com dois DataFrames:
    1. df_global: 1 linha por código SH6 (Consolidado Global do País Exportador no Mundo)
    2. df_bilateral: 1 linha por combinação Reporter (País Comprador) x SH6
    """
    years = tuple(sorted(set(int(y) for y in years)))
    df_filtered = comtrade_tidy[comtrade_tidy["ano"].isin(years)].copy()
    if df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    min_year, max_year = years[0], years[-1]
    num_years = max_year - min_year

    # Detectar o rótulo do Mundo
    all_reporters = df_filtered["reporter"].unique()
    world_rep = detect_world_label(all_reporters)

    # Separação do dataset
    if world_rep:
        world_raw = df_filtered[df_filtered["reporter"] == world_rep]
        rep_raw = df_filtered[df_filtered["reporter"] != world_rep]
    else:
        world_raw = df_filtered
        rep_raw = df_filtered

    # --- CAMADA 1: CONSOLIDAÇÃO GLOBAL POR SH6 (1 LINHA POR PRODUTO) ---
    # Consolida vendas de cada país no MUNDO para o produto k
    rep_sh6_global = rep_raw.groupby(["reporter", "sh6", "sh6_desc"], as_index=False)["valor"].sum()
    
    # Demanda Global do produto k
    world_sh6_global = world_raw.groupby(["sh6"], as_index=False)["valor"].sum().rename(columns={"valor": "mundo_valor"})
    world_map = world_sh6_global.set_index("sh6")["mundo_valor"]
    X_w_total = float(world_sh6_global["mundo_valor"].sum())

    # Dinamismo da Demanda Mundial (CAGR ou % de variação)
    w_min = world_raw[world_raw["ano"] == min_year].groupby("sh6")["valor"].sum().rename("mundo_val_init")
    w_max = world_raw[world_raw["ano"] == max_year].groupby("sh6")["valor"].sum().rename("mundo_val_final")
    w_growth = pd.concat([w_min, w_max], axis=1).fillna(0.0)

    if num_years > 0:
        w_growth["crescimento_mundo_pct"] = np.where(
            w_growth["mundo_val_init"] > 0,
            ((w_growth["mundo_val_final"] / w_growth["mundo_val_init"]) ** (1.0 / num_years) - 1.0) * 100.0,
            0.0
        )
    else:
        w_growth["crescimento_mundo_pct"] = 0.0

    pv_rep_g = rep_sh6_global.pivot_table(index="sh6", columns="reporter", values="valor", aggfunc="sum", fill_value=0.0)
    X_i_total_g = pv_rep_g.sum(axis=0)

    share_pais_g = pv_rep_g.div(X_i_total_g.replace(0, np.nan), axis=1).fillna(0.0)
    share_mundo_g = (world_map / X_w_total) if X_w_total > 0 else pd.Series(0.0, index=pv_rep_g.index)
    share_mundo_g = share_mundo_g.reindex(pv_rep_g.index).fillna(0.0)

    rca_g_df = share_pais_g.div(share_mundo_g.replace(0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rsca_g_df = ((rca_g_df - 1) / (rca_g_df + 1)).fillna(-1.0)
    ms_g_df = pv_rep_g.div(world_map.replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    # Unpivot Tidy Global
    val_g_long = pv_rep_g.stack().rename("valor").reset_index()
    rca_g_long = rca_g_df.stack().rename("rca_global").reset_index()
    rsca_g_long = rsca_g_df.stack().rename("rsca_global").reset_index()
    ms_g_long = ms_g_df.stack().rename("market_share_pct").reset_index()

    df_global = val_g_long.merge(rca_g_long, on=["sh6", "reporter"])\
                          .merge(rsca_g_long, on=["sh6", "reporter"])\
                          .merge(ms_g_long, on=["sh6", "reporter"])

    sh6_descs = rep_sh6_global.groupby("sh6")["sh6_desc"].first()
    df_global["sh6_desc"] = df_global["sh6"].map(sh6_descs)
    df_global["mundo_valor"] = df_global["sh6"].map(world_map).fillna(0.0)
    df_global["crescimento_mundo_pct"] = df_global["sh6"].map(w_growth["crescimento_mundo_pct"]).fillna(0.0)

    def classify(row):
        advantage = row["rsca_global"] >= 0.0
        dynamic = row["crescimento_mundo_pct"] > 0
        if advantage and dynamic:
            return "🌟 Oportunidade (Star)"
        elif advantage and not dynamic:
            return "⚠️ Vulnerabilidade"
        elif not advantage and dynamic:
            return "🚀 Oportunidade Perdida"
        else:
            return "📉 Retirada / Neutro"

    df_global["posicao_estrategica"] = df_global.apply(classify, axis=1)

    # --- CAMADA 2: ANÁLISE BILATERAL POR PAÍS COMPRADOR (REPORTER x SH6) ---
    rep_sh6_prt = rep_raw.groupby(["reporter", "partner", "sh6", "sh6_desc"], as_index=False)["valor"].sum()
    
    # Calcular o RCA Bilateral em relação ao fluxo do país parceiro
    pv_prt = rep_sh6_prt.pivot_table(index=["sh6", "partner"], columns="reporter", values="valor", aggfunc="sum", fill_value=0.0)
    tot_prt_rep = pv_prt.sum(axis=0)

    share_prt_pais = pv_prt.div(tot_prt_rep.replace(0, np.nan), axis=1).fillna(0.0)
    
    # Total de compras do país parceiro (Reporter)
    tot_prt_world = rep_sh6_prt.groupby("partner")["valor"].sum()
    val_prt_sh6 = rep_sh6_prt.groupby(["sh6", "partner"])["valor"].sum()
    share_prt_world = val_prt_sh6.div(val_prt_sh6.index.get_level_values("partner").map(tot_prt_world), axis=0).fillna(0.0)

    rca_bila_df = share_prt_pais.div(share_prt_world.replace(0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rsca_bila_df = ((rca_bila_df - 1) / (rca_bila_df + 1)).fillna(-1.0)

    val_b_long = pv_prt.stack().rename("valor").reset_index()
    rca_b_long = rca_bila_df.stack().rename("rca_bilateral").reset_index()
    rsca_b_long = rsca_bila_df.stack().rename("rsca_bilateral").reset_index()

    df_bilateral = val_b_long.merge(rca_b_long, on=["sh6", "partner", "reporter"])\
                             .merge(rsca_b_long, on=["sh6", "partner", "reporter"])
    df_bilateral["sh6_desc"] = df_bilateral["sh6"].map(sh6_descs)

    return df_global, df_bilateral


def compute_state_diversification_potentials(
    comexstat_uf: pd.DataFrame, df_global: pd.DataFrame, year: int
) -> pd.DataFrame:
    if "ano" in comexstat_uf.columns:
        base_uf = comexstat_uf[comexstat_uf["ano"] == year].copy()
    else:
        base_uf = comexstat_uf.copy()

    required_cols = {"uf", "sh6_cod", "valor_fob"}
    if base_uf.empty or not required_cols.issubset(base_uf.columns) or df_global.empty:
        return pd.DataFrame()

    brazil_rep_name = next(
        (r for r in df_global["reporter"].dropna().unique()
         if normalize_text(r) in ("brazil", "brasil", "bra")),
        None,
    )
    if brazil_rep_name is None:
        return pd.DataFrame()

    br_metrics = df_global[df_global["reporter"] == brazil_rep_name].copy()
    if br_metrics.empty:
        return pd.DataFrame()

    advantage = br_metrics[br_metrics["rsca_global"] >= 0.0].copy()

    if advantage.empty:
        return pd.DataFrame()

    desc_cols = [c for c in COMEXSTAT_DESC_COLUMNS if c in base_uf.columns and c != "sh6_desc"]
    if desc_cols:
        desc_map = base_uf.groupby("sh6_cod")[desc_cols].first()
    else:
        desc_map = pd.DataFrame(index=pd.Index([], name="sh6_cod"))

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

    adv_slim = advantage[[
        "sh6", "sh6_desc", "rca_global", "rsca_global", "market_share_pct",
        "crescimento_mundo_pct", "posicao_estrategica", "mundo_valor"
    ]].rename(columns={"sh6": "sh6_cod", "rca_global": "rca", "rsca_global": "rsca"})
    
    grid = grid.merge(adv_slim, on="sh6_cod", how="left")

    if not desc_map.empty:
        grid = grid.merge(desc_map, on="sh6_cod", how="left")

    grid["potencial_score"] = grid["rca"] * (1 - grid["share_local"])
    grid["ja_exportado"] = grid["valor_fob"] > 0

    return grid.sort_values("potencial_score", ascending=False)


# ==============================================================================
# 5. ESTRUTURA VISUAL (DESIGN CUPERTINO & GLASSMORPHISM)
# ==============================================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

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

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.45) !important;
            backdrop-filter: blur(25px) saturate(190%) !important;
            -webkit-backdrop-filter: blur(25px) saturate(190%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.7) !important;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
            margin: 14px 0 28px 0;
        }

        .metric-card {
            border-radius: 20px;
            padding: 20px 22px;
            backdrop-filter: blur(18px) saturate(180%);
            -webkit-backdrop-filter: blur(18px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.85);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
            display: flex;
            flex-direction: column;
            justify-content: center;
            transition: all 0.25s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.04);
        }

        .metric-card .m-value {
            font-size: 1.85rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.025em;
        }

        .metric-card .m-label {
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #6b7280;
        }

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


PASTEL_COLORS = {
    "green_main": "#10b981",
    "green_bg": "rgba(236, 253, 245, 0.75)",
    "amber_main": "#f59e0b",
    "amber_bg": "rgba(254, 243, 199, 0.75)",
    "blue_main": "#3b82f6",
    "blue_bg": "rgba(239, 246, 255, 0.75)",
    "purple_main": "#8b5cf6",
    "purple_bg": "rgba(245, 243, 255, 0.75)",
    "red_main": "#ef4444",
    "red_bg": "rgba(244, 242, 242, 0.75)",
}


# ==============================================================================
# 6. GERENCIAMENTO DE ESTADO E SIDEBAR
# ==============================================================================

inject_custom_css()

st.sidebar.title("🌎 Navegação")
PAGE = st.sidebar.radio(
    "Selecione o Módulo:",
    [
        "📊 Indicadores Oficiais de Competitividade (UN Comtrade)",
        "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)",
        "🗺️ Potencial de Diversificação por Estado (UF)",
    ],
)

st.sidebar.divider()
st.sidebar.header("📁 Carga de Dados")

file_comtrade = st.sidebar.file_uploader("1. UN Comtrade (CSV, XLSX, Parquet, JSON)", type=["csv", "xlsx", "xls", "parquet", "json"])
file_comexstat = st.sidebar.file_uploader("2. ComexStat Brasil (Nacional)", type=["csv", "xlsx", "xls", "parquet"])
file_comexstat_uf = st.sidebar.file_uploader("3. ComexStat por Estado (UF)", type=["csv", "xlsx", "xls", "parquet"])

COMTRADE_FIXED_COLS = {
    "year": "refYear",
    "reporter": "ReporterDesc",
    "partner": "PartnerDesc",
    "sh6": "cmdCode",
    "sh6_desc": "cmdDesc",
}

if file_comtrade:
    if st.session_state.get("_file_comtrade_name") != file_comtrade.name:
        raw_ct = read_any_file(file_comtrade.getvalue(), file_comtrade.name)
        st.session_state["raw_comtrade"] = raw_ct
        st.session_state["_file_comtrade_name"] = file_comtrade.name
        st.session_state.pop("comtrade_tidy", None)

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
    cols = list(df_ct.columns)

    col_yr = find_column(df_ct, ["refyear"]) or COMTRADE_FIXED_COLS["year"]
    col_rep = find_column(df_ct, ["reporterdesc"]) or COMTRADE_FIXED_COLS["reporter"]
    col_prt = find_column(df_ct, ["partnerdesc"]) or COMTRADE_FIXED_COLS["partner"]
    col_sh6 = find_column(df_ct, ["cmdcode"]) or COMTRADE_FIXED_COLS["sh6"]
    col_desc = find_column(df_ct, ["cmddesc"]) or COMTRADE_FIXED_COLS["sh6_desc"]

    with st.sidebar.expander("⚙️ Configurações UN Comtrade", expanded=True):
        st.caption("📌 **Campos fixos:** refYear, ReporterDesc, PartnerDesc, cmdCode, cmdDesc")

        value_candidates = [c for c in cols if any(v in normalize_text(c) for v in ["value", "val", "fob", "cif", "primaryvalue"])]
        default_val_idx = cols.index(value_candidates[0]) if value_candidates else 0
        c_val = st.selectbox("Selecione o Campo de Valor:", cols, index=default_val_idx)

        if col_prt in df_ct.columns:
            partners_list = sorted(df_ct[col_prt].dropna().astype(str).unique().tolist())
            sel_partners = st.multiselect("Filtrar Parceiros Comerciais (vazio = todos):", partners_list, default=[])
        else:
            sel_partners = []

        if col_rep in df_ct.columns:
            reporters_list = sorted(df_ct[col_rep].dropna().astype(str).unique().tolist())
            sel_reporters = st.multiselect("Filtrar Países Reporters (vazio = todos):", reporters_list, default=[])
        else:
            sel_reporters = []

        missing_cols = [c for c in [col_yr, col_rep, col_prt, col_sh6] if c not in df_ct.columns]

        if missing_cols:
            st.error(f"⚠️ As seguintes colunas obrigatórias não foram encontradas na planilha: {', '.join(missing_cols)}")
        else:
            if "comtrade_tidy" not in st.session_state:
                try:
                    st.session_state["comtrade_tidy"] = standardize_comtrade(
                        df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
                    )
                except Exception as e:
                    st.error(f"Erro ao processar automaticamente: {e}")

            if st.button("Aplicar / Re-processar Métricas", type="primary"):
                try:
                    tidy_ct = standardize_comtrade(
                        df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
                    )
                    st.session_state["comtrade_tidy"] = tidy_ct
                    st.success("Dados do UN Comtrade processados com sucesso!")
                except Exception as exc:
                    st.error(f"Falha ao processar a base do Comtrade: {exc}")

# ==============================================================================
# 7. MÓDULOS DA APLICAÇÃO
# ==============================================================================

# --- PÁGINA 1: UN COMTRADE - ESTRUTURA DUAL (GLOBAL SH6 x BILATERAL) ---
def page_comtrade_global():
    st.title("📊 Indicadores Oficiais de Competitividade Internacional")
    st.caption("Visão Dupla: RCA Consolidado Global por SH6 e RCA Bilateral Específico por País Comprador.")

    if "comtrade_tidy" not in st.session_state:
        st.info("👈 Por favor, carregue e processe o arquivo do UN Comtrade na barra lateral.")
        return

    tidy = st.session_state["comtrade_tidy"]
    anos = sorted(int(a) for a in tidy["ano"].dropna().unique().tolist())
    if not anos:
        st.error("A base processada não contém anos válidos.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        start_year = st.selectbox("Ano de Início", anos, index=0)
    with col_b:
        end_year = st.selectbox("Ano Final", anos, index=len(anos) - 1)

    years_range = [y for y in anos if start_year <= y <= end_year] or [start_year]
    df_global, df_bilateral = compute_comtrade_dual_metrics(tidy, tuple(years_range))

    if df_global.empty:
        st.warning("Não há dados suficientes para o intervalo de anos selecionado.")
        return

    # --- CÁLCULO DAS MÉTRICAS EXECUTIVAS DE FÁCIL EXTRAÇÃO ---
    total_sh6_pauta = df_global["sh6"].nunique()
    sh6_com_vantagem = df_global[df_global["rca_global"] >= 1.0]["sh6"].nunique()
    
    # Cobertura geográfica do país
    if not df_bilateral.empty:
        total_paises_destino = df_bilateral["partner"].nunique()
        paises_com_vantagem = df_bilateral[df_bilateral["rca_bilateral"] >= 1.0]["partner"].nunique()
    else:
        total_paises_destino = 0
        paises_com_vantagem = 0

    media_rca_g = df_global["rca_global"].mean()
    media_rsca_g = df_global["rsca_global"].mean()

    # --- CARTÕES DE KPI EXECUTIVOS ---
    html_kpis = f"""
    <div class="metric-grid">
        <div class="metric-card" style="background: rgba(236, 253, 245, 0.85); border-color: rgba(16, 185, 129, 0.35);">
            <div class="m-value" style="color: #059669;">{sh6_com_vantagem:,}</div>
            <div class="m-label">Produtos com RCA Global ≥ 1.0</div>
        </div>
        <div class="metric-card" style="background: rgba(239, 246, 255, 0.85); border-color: rgba(59, 130, 246, 0.35);">
            <div class="m-value" style="color: #2563eb;">{total_sh6_pauta:,}</div>
            <div class="m-label">Total de Produtos na Pauta (SH6)</div>
        </div>
        <div class="metric-card" style="background: rgba(245, 243, 255, 0.85); border-color: rgba(139, 92, 246, 0.35);">
            <div class="m-value" style="color: #7c3aed;">{paises_com_vantagem:,}</div>
            <div class="m-label">Países Destino com RCA ≥ 1.0</div>
        </div>
        <div class="metric-card" style="background: rgba(254, 243, 199, 0.85); border-color: rgba(245, 158, 11, 0.35);">
            <div class="m-value" style="color: #d97706;">{total_paises_destino:,}</div>
            <div class="m-label">Total de Países de Destino</div>
        </div>
        <div class="metric-card" style="background: rgba(243, 244, 246, 0.85); border-color: rgba(156, 163, 175, 0.35);">
            <div class="m-value" style="color: #4b5563;">{format_num(media_rsca_g, 2)}</div>
            <div class="m-label">RSCA Médio Consolidado</div>
        </div>
    </div>
    """
    st.markdown(html_kpis, unsafe_allow_html=True)

    st.divider()

    # --- ABAS DE NÍVEL DE AGREGAÇÃO ---
    tab_g, tab_b = st.tabs([
        "🌐 Visão Consolidada Global (1 Linha por SH6)",
        "🤝 Visão Bilateral Específica (Por País Comprador)"
    ])

    with tab_g:
        st.subheader("📌 Tabela Consolidada Global (Nível SH6)")
        st.caption(f"Exatamente {total_sh6_pauta:,} produtos avaliados consolidados para o país no mercado global.")

        st.dataframe(
            df_global,
            column_config={
                "sh6": "Código SH6",
                "sh6_desc": "Descrição do Produto",
                "reporter": "País Exportador",
                "valor": st.column_config.NumberColumn("Valor Exportado (US$)", format="$ %,.2f"),
                "mundo_valor": st.column_config.NumberColumn("Demanda Global (US$)", format="$ %,.2f"),
                "rca_global": st.column_config.NumberColumn("RCA Global (Balassa)", format="%.4f"),
                "rsca_global": st.column_config.NumberColumn("RSCA Global (Simétrico)", format="%.4f"),
                "market_share_pct": st.column_config.NumberColumn("Market Share Global (%)", format="%.2f%%"),
                "crescimento_mundo_pct": st.column_config.NumberColumn("Dinamismo Mundial (%)", format="%.2f%%"),
                "posicao_estrategica": "Matriz CEPAL/MAGIC",
            },
            height=380,
        )

        st.markdown("#### **Matriz Trimétrica Consolidada: RSCA Global vs Dinamismo Mundial**")
        fig_matrix = px.scatter(
            df_global,
            x="rsca_global",
            y="crescimento_mundo_pct",
            color="posicao_estrategica",
            size="market_share_pct",
            hover_data=["sh6", "sh6_desc"],
            labels={"rsca_global": "RSCA Global", "crescimento_mundo_pct": "Dinamismo Mundial (%)"},
            template="plotly_white",
        )
        fig_matrix.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_matrix.add_vline(x=0.0, line_dash="dash", line_color="#9ca3af")
        fig_matrix.add_hline(y=0.0, line_dash="dash", line_color="#9ca3af")
        st.plotly_chart(fig_matrix, width="stretch")

    with tab_b:
        st.subheader("📌 Tabela Bilateral por País Comprador (Partner x SH6)")
        st.caption("Detalhamento específico do desempenho competitivo por mercado comprador de destino.")

        if not df_bilateral.empty:
            prts_filter = st.multiselect("Filtrar por País Comprador (Partner):", sorted(df_bilateral["partner"].unique()))
            df_b_filtered = df_bilateral[df_bilateral["partner"].isin(prts_filter)] if prts_filter else df_bilateral

            st.dataframe(
                df_b_filtered,
                column_config={
                    "sh6": "Código SH6",
                    "sh6_desc": "Descrição do Produto",
                    "partner": "País Comprador",
                    "reporter": "País Exportador",
                    "valor": st.column_config.NumberColumn("Valor Exportado ao País (US$)", format="$ %,.2f"),
                    "rca_bilateral": st.column_config.NumberColumn("RCA Bilateral", format="%.4f"),
                    "rsca_bilateral": st.column_config.NumberColumn("RSCA Bilateral", format="%.4f"),
                },
                height=380,
            )
        else:
            st.info("Não há registros bilaterais disponíveis para o filtro atual.")


# --- PÁGINA 2: CRUZAMENTO BRASIL COMEXSTAT X COMTRADE ---
def page_comexstat_cross():
    st.title("🇧🇷 Cruzamento das Exportações do Brasil com Competitividade Global")
    st.caption("Integração da pauta detalhada do ComexStat aos indicadores oficiais consolidados do Comtrade.")

    if "comexstat" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ É necessário carregar AMBOS os arquivos (ComexStat Brasil e UN Comtrade) na barra lateral.")
        return

    cs = st.session_state["comexstat"]
    ct = st.session_state["comtrade_tidy"]

    if "ano" not in cs.columns or "sh6_cod" not in cs.columns:
        st.error("A base do ComexStat Brasil precisa conter colunas de Ano e Código SH6.")
        return

    anos_cs = sorted(int(a) for a in cs["ano"].dropna().unique().tolist())
    selected_year = st.selectbox("Selecione o Ano para Análise Cruzada:", anos_cs, index=len(anos_cs) - 1)

    df_global, _ = compute_comtrade_dual_metrics(ct, (selected_year,))
    if df_global.empty:
        st.warning("Não há dados do Comtrade para o ano selecionado.")
        return

    brazil_rep = next(
        (r for r in df_global["reporter"].unique() if normalize_text(r) in ("brazil", "brasil", "bra")),
        None,
    )
    if brazil_rep is None:
        st.error("Não foi encontrado um reporter 'Brazil'/'Brasil' nos dados do Comtrade processados.")
        return

    br_ct = df_global[df_global["reporter"] == brazil_rep][[
        "sh6", "rca_global", "rsca_global", "market_share_pct", "crescimento_mundo_pct", "posicao_estrategica", "mundo_valor"
    ]].copy()

    merged = pd.merge(
        cs[cs["ano"] == selected_year],
        br_ct,
        left_on="sh6_cod",
        right_on="sh6",
        how="left",
    )
    merged["rca_global"] = merged["rca_global"].fillna(0)
    merged["rsca_global"] = merged["rsca_global"].fillna(-1)
    merged["market_share_pct"] = merged["market_share_pct"].fillna(0)
    merged["crescimento_mundo_pct"] = merged["crescimento_mundo_pct"].fillna(0)

    st.subheader("🔍 Filtros de Segmentação da Pauta Exportadora")
    col1, col2, col3 = st.columns(3)
    with col1:
        f_sh6 = st.multiselect("SH6:", sorted(merged["sh6_cod"].dropna().unique()))
        f_cuci = st.multiselect(
            "CUCI Grupo:",
            sorted(merged["cuci_desc"].dropna().unique()) if "cuci_desc" in merged.columns else [],
        )
    with col2:
        f_isic_div = st.multiselect(
            "ISIC Divisão:",
            sorted(merged["isic_div_desc"].dropna().unique()) if "isic_div_desc" in merged.columns else [],
        )
        f_isic_sec = st.multiselect(
            "ISIC Seção:",
            sorted(merged["isic_sec_desc"].dropna().unique()) if "isic_sec_desc" in merged.columns else [],
        )
    with col3:
        f_cgce1 = st.multiselect(
            "CGCE Nível 1:",
            sorted(merged["cgce1_desc"].dropna().unique()) if "cgce1_desc" in merged.columns else [],
        )
        f_cgce2 = st.multiselect(
            "CGCE Nível 2:",
            sorted(merged["cgce2_desc"].dropna().unique()) if "cgce2_desc" in merged.columns else [],
        )

    df_f = merged.copy()
    if f_sh6:
        df_f = df_f[df_f["sh6_cod"].isin(f_sh6)]
    if f_cuci and "cuci_desc" in df_f.columns:
        df_f = df_f[df_f["cuci_desc"].isin(f_cuci)]
    if f_isic_div and "isic_div_desc" in df_f.columns:
        df_f = df_f[df_f["isic_div_desc"].isin(f_isic_div)]
    if f_isic_sec and "isic_sec_desc" in df_f.columns:
        df_f = df_f[df_f["isic_sec_desc"].isin(f_isic_sec)]
    if f_cgce1 and "cgce1_desc" in df_f.columns:
        df_f = df_f[df_f["cgce1_desc"].isin(f_cgce1)]
    if f_cgce2 and "cgce2_desc" in df_f.columns:
        df_f = df_f[df_f["cgce2_desc"].isin(f_cgce2)]

    val_tot = df_f["valor_fob"].sum()
    produtos_vantagem = df_f[df_f["rsca_global"] >= 0.0]["sh6_cod"].nunique()

    html_stats = f"""
    <div class="metric-grid">
        <div class="metric-card" style="background: rgba(239, 246, 255, 0.85); border-color: rgba(59, 130, 246, 0.35);">
            <div class="m-value" style="color: #2563eb;">{format_usd(val_tot)}</div>
            <div class="m-label">Valor Total Exportado (FOB)</div>
        </div>
        <div class="metric-card" style="background: rgba(245, 243, 255, 0.85); border-color: rgba(139, 92, 246, 0.35);">
            <div class="m-value" style="color: #7c3aed;">{df_f['sh6_cod'].nunique():,}</div>
            <div class="m-label">Total de Produtos SH6</div>
        </div>
        <div class="metric-card" style="background: rgba(236, 253, 245, 0.85); border-color: rgba(16, 185, 129, 0.35);">
            <div class="m-value" style="color: #059669;">{produtos_vantagem:,}</div>
            <div class="m-label">Produtos Competitivos (RSCA ≥ 0)</div>
        </div>
    </div>
    """
    st.markdown(html_stats, unsafe_allow_html=True)

    st.subheader("📊 Distribuição dos Indicadores Separados por Setor")
    group_opt = st.selectbox(
        "Agrupar Visualização por:",
        ["CUCI Grupo", "ISIC Divisão", "ISIC Seção", "CGCE Nível 1", "CGCE Nível 2"],
    )

    col_map = {
        "CUCI Grupo": "cuci_desc",
        "ISIC Divisão": "isic_div_desc",
        "ISIC Seção": "isic_sec_desc",
        "CGCE Nível 1": "cgce1_desc",
        "CGCE Nível 2": "cgce2_desc",
    }
    selected_col = col_map[group_opt]

    if selected_col in df_f.columns:
        agg_sector = (
            df_f.groupby(selected_col)
            .agg(
                Valor_FOB=("valor_fob", "sum"),
                RCA_Medio=("rca_global", "mean"),
                RSCA_Medio=("rsca_global", "mean"),
                Market_Share_Medio=("market_share_pct", "mean"),
                Dinamismo_Medio=("crescimento_mundo_pct", "mean"),
                N_Produtos=("sh6_cod", "nunique"),
            )
            .reset_index()
            .sort_values("Valor_FOB", ascending=False)
        )

        col_sec1, col_sec2 = st.columns(2)

        with col_sec1:
            fig_sec_rsca = px.bar(
                agg_sector.head(15),
                x="Valor_FOB",
                y=selected_col,
                orientation="h",
                color="RSCA_Medio",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                title=f"Top 15 Setores por Valor e RSCA Médio ({group_opt})",
                labels={"Valor_FOB": "Valor FOB (US$)", selected_col: "Setor", "RSCA_Medio": "RSCA Médio"},
                template="plotly_white",
            )
            fig_sec_rsca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec_rsca, width="stretch")

        with col_sec2:
            fig_sec_ms = px.bar(
                agg_sector.head(15),
                x="Valor_FOB",
                y=selected_col,
                orientation="h",
                color="Market_Share_Medio",
                color_continuous_scale=["#eff6ff", "#3b82f6", "#1d4ed8"],
                title=f"Top 15 Setores por Valor e Market Share Médio ({group_opt})",
                labels={"Valor_FOB": "Valor FOB (US$)", selected_col: "Setor", "Market_Share_Medio": "Market Share Médio (%)"},
                template="plotly_white",
            )
            fig_sec_ms.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec_ms, width="stretch")

        st.dataframe(
            agg_sector,
            column_config={
                "Valor_FOB": st.column_config.NumberColumn("Valor FOB (US$)", format="$ %,.2f"),
                "RCA_Medio": st.column_config.NumberColumn("RCA Médio", format="%.2f"),
                "RSCA_Medio": st.column_config.NumberColumn("RSCA Médio", format="%.2f"),
                "Market_Share_Medio": st.column_config.NumberColumn("Market Share Médio (%)", format="%.2f%%"),
                "Dinamismo_Medio": st.column_config.NumberColumn("Dinamismo Médio (%)", format="%.2f%%"),
                "N_Produtos": st.column_config.NumberColumn("Produtos SH6", format="%d"),
            },
        )
    else:
        st.info(f"A coluna '{group_opt}' não foi encontrada na base carregada.")


# --- PÁGINA 3: POTENCIAL DE DIVERSIFICAÇÃO POR ESTADO (UF) ---
def page_state_diversification():
    st.title("🗺️ Potencial de Diversificação Exportadora por Estado (UF)")
    st.caption("Cruzamento subnacional com a base de competitividade oficial (RSCA ≥ 0, Market Share e Dinamismo Mundial).")

    if "comexstat_uf" not in st.session_state or "comtrade_tidy" not in st.session_state:
        st.warning("⚠️ É necessário carregar a planilha do ComexStat por Estado (UF) e o UN Comtrade.")
        return

    cs_uf = st.session_state["comexstat_uf"]
    ct = st.session_state["comtrade_tidy"]

    if "ano" not in cs_uf.columns or "uf" not in cs_uf.columns or "sh6_cod" not in cs_uf.columns:
        st.error("A base do ComexStat por Estado precisa conter colunas de Ano, UF e Código SH6.")
        return

    anos_uf = sorted(int(a) for a in cs_uf["ano"].dropna().unique().tolist())
    selected_year = st.selectbox("Ano de Análise Subnacional:", anos_uf, index=len(anos_uf) - 1)

    df_global, _ = compute_comtrade_dual_metrics(ct, (selected_year,))
    if df_global.empty:
        st.warning("Não há dados do Comtrade para o ano selecionado.")
        return

    df_potencial = compute_state_diversification_potentials(cs_uf, df_global, selected_year)

    if df_potencial.empty:
        st.error("Não foi possível calcular o potencial com os dados fornecidos. Verifique se o Brasil possui produtos com RSCA ≥ 0 no ano selecionado.")
        return

    st.subheader("🧭 Radar de Oportunidades Estaduais (Matriz de Vantagem & Presença)")

    n_alto_potencial = df_potencial[(df_potencial["potencial_score"] > 1.0) & (~df_potencial["ja_exportado"])]["sh6_cod"].nunique()
    n_vantagem_nacional = df_potencial["sh6_cod"].nunique()
    val_mercado_oportunidade = (
        df_potencial[(df_potencial["potencial_score"] > 1.0) & (~df_potencial["ja_exportado"])]
        .drop_duplicates("sh6_cod")["mundo_valor"]
        .sum()
    )

    html_quadrants = f"""
    <div class="quadrant-grid">
        <div class="quadrant-card" style="background:{PASTEL_COLORS['green_bg']}; border-color:rgba(16, 185, 129, 0.3);">
            <div>
                <div class="qc-title" style="color:{PASTEL_COLORS['green_main']};">
                    <span class="qc-dot" style="background:{PASTEL_COLORS['green_main']};"></span>Alta Oportunidade Local
                </div>
                <div class="qc-desc">Produtos em que o Brasil possui vantagem comparativa oficial (RSCA ≥ 0), mas o estado ainda não realiza exportações.</div>
            </div>
            <div class="qc-bottom">
                <div>
                    <div class="qc-count">{n_alto_potencial:,}</div>
                    <div class="qc-count-label">produtos-chave nacionais</div>
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
                <div class="qc-desc">Produtos do portfólio brasileiro com alto índice de especialização e inserção no comércio internacional.</div>
            </div>
            <div class="qc-bottom">
                <div>
                    <div class="qc-count">{n_vantagem_nacional:,}</div>
                    <div class="qc-count-label">produtos competitivos</div>
                </div>
                <div>
                    <div class="qc-value">RSCA ≥ 0.0</div>
                    <div class="qc-pill" style="background:rgba(59, 130, 246, 0.15); color:{PASTEL_COLORS['blue_main']};">Base Brasil</div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html_quadrants, unsafe_allow_html=True)

    st.subheader("🏆 Ranking de Estados por Score de Potencial de Diversificação")

    rank_uf = (
        df_potencial.groupby("uf")
        .agg(
            Score_Potencial_Total=("potencial_score", "sum"),
            Produtos_Nao_Explorados=("ja_exportado", lambda s: int((~s).sum())),
            Exportacao_Atual_FOB=("uf_total", "first"),
        )
        .reset_index()
        .sort_values("Score_Potencial_Total", ascending=False)
    )

    fig_uf_rank = px.bar(
        rank_uf,
        x="Score_Potencial_Total",
        y="uf",
        orientation="h",
        color="Exportacao_Atual_FOB",
        color_continuous_scale=["#eff6ff", "#3b82f6", "#1d4ed8"],
        title="Estados com Maior Potencial de Diversificação",
        labels={"Score_Potencial_Total": "Score de Potencial", "uf": "Estado", "Exportacao_Atual_FOB": "Exportação Atual (US$)"},
        template="plotly_white",
    )
    fig_uf_rank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_uf_rank, width="stretch")

    st.divider()

    st.subheader("🔍 Detalhamento e Indicadores Separados por Estado (UF)")

    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
    with col_sel1:
        uf_target = st.selectbox("Selecione o Estado (UF):", sorted(df_potencial["uf"].unique()))

    df_uf_filtered = df_potencial[df_potencial["uf"] == uf_target]

    with col_sel2:
        cuci_options = sorted(df_uf_filtered["cuci_desc"].dropna().unique()) if "cuci_desc" in df_uf_filtered.columns else []
        cuci_target = st.multiselect("Filtrar por CUCI Grupo:", cuci_options)
    with col_sel3:
        sh6_options = sorted(df_uf_filtered["sh6_cod"].dropna().unique())
        sh6_target = st.multiselect("Filtrar por SH6:", sh6_options)
    with col_sel4:
        only_new = st.checkbox("Somente produtos NÃO exportados pelo estado", value=False)

    df_uf_seg = df_uf_filtered.copy()
    if cuci_target and "cuci_desc" in df_uf_seg.columns:
        df_uf_seg = df_uf_seg[df_uf_seg["cuci_desc"].isin(cuci_target)]
    if sh6_target:
        df_uf_seg = df_uf_seg[df_uf_seg["sh6_cod"].isin(sh6_target)]
    if only_new:
        df_uf_seg = df_uf_seg[~df_uf_seg["ja_exportado"]]

    st.markdown(f"### Oportunidades Prioritárias para **{uf_target}**")

    disp_cols = [
        "sh6_cod",
        "sh6_desc",
        "cuci_desc",
        "rca",
        "rsca",
        "market_share_pct",
        "crescimento_mundo_pct",
        "posicao_estrategica",
        "valor_fob",
        "share_local",
        "potencial_score",
        "ja_exportado",
    ]
    available_disp_cols = [c for c in disp_cols if c in df_uf_seg.columns]

    st.dataframe(
        df_uf_seg[available_disp_cols].sort_values("potencial_score", ascending=False).head(50),
        column_config={
            "sh6_cod": "Código SH6",
            "sh6_desc": "Descrição Produto",
            "cuci_desc": "CUCI Grupo",
            "rca": st.column_config.NumberColumn("RCA Global", format="%.2f"),
            "rsca": st.column_config.NumberColumn("RSCA Global", format="%.2f"),
            "market_share_pct": st.column_config.NumberColumn("Market Share Global (%)", format="%.2f%%"),
            "crescimento_mundo_pct": st.column_config.NumberColumn("Dinamismo Mundial (%)", format="%.2f%%"),
            "posicao_estrategica": "Matriz CEPAL",
            "valor_fob": st.column_config.NumberColumn("Exportado no Estado (US$)", format="$ %,.2f"),
            "share_local": st.column_config.NumberColumn("Participação Local", format="%.2%%"),
            "potencial_score": st.column_config.NumberColumn("Score Potencial", format="%.2f"),
            "ja_exportado": "Exporta Atualmente?",
        },
        height=420,
    )


# ==============================================================================
# 8. ROTEADOR DE PÁGINAS
# ==============================================================================

if PAGE == "📊 Indicadores Oficiais de Competitividade (UN Comtrade)":
    page_comtrade_global()
elif PAGE == "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)":
    page_comexstat_cross()
elif PAGE == "🗺️ Potencial de Diversificação por Estado (UF)":
    page_state_diversification()
