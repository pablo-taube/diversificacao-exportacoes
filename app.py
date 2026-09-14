# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS
 (Glassmorfismo Cupertino) — v4 (Gráficos Separados para RCA e RSCA)
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
        df = pd.read_csv(
            io.BytesIO(raw), sep=sep, encoding="utf-8", low_memory=False
        )
        if df.shape[1] > 1:
          return df
      except Exception:
        continue
    try:
      return pd.read_csv(
          io.BytesIO(raw),
          sep=None,
          engine="python",
          encoding="utf-8",
          low_memory=False,
      )
    except UnicodeDecodeError:
      return pd.read_csv(
          io.BytesIO(raw),
          sep=None,
          engine="python",
          encoding="latin-1",
          low_memory=False,
      )

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
    "ano",
    "pais",
    "sh6_cod",
    "sh6_desc",
    "cgce2_cod",
    "cgce2_desc",
    "cgce1_cod",
    "cgce1_desc",
    "cuci_cod",
    "cuci_desc",
    "isic_div_cod",
    "isic_div_desc",
    "isic_sec_cod",
    "isic_sec_desc",
    "valor_fob",
    "uf",
]

COMEXSTAT_DESC_COLUMNS = [
    "sh6_desc",
    "cuci_cod",
    "cuci_desc",
    "isic_div_desc",
    "isic_sec_desc",
    "cgce1_desc",
    "cgce2_desc",
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
          df["valor_fob"]
          .astype(str)
          .str.replace(r"[^\d,.\-]", "", regex=True)
          .str.replace(".", "", regex=False)
          .str.replace(",", ".", regex=False)
      )
      df["valor_fob"] = pd.to_numeric(cleaned, errors="coerce")
    else:
      df["valor_fob"] = pd.to_numeric(df["valor_fob"], errors="coerce")
    df["valor_fob"] = df["valor_fob"].fillna(0.0)

  if "sh6_cod" in df.columns:
    sh_clean = (
        pd.to_numeric(df["sh6_cod"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
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
  sh_clean = (
      pd.to_numeric(d["sh6"], errors="coerce").fillna(0).astype(int).astype(str)
  )
  d["sh6"] = sh_clean.str.zfill(6)
  d["valor"] = pd.to_numeric(d["valor"], errors="coerce").fillna(0.0)

  if "sh6_desc" not in d.columns:
    d["sh6_desc"] = d["sh6"]

  if selected_partners:
    prt_str_list = [str(p).lower() for p in selected_partners]
    d = d[d["partner"].astype(str).str.lower().isin(prt_str_list)]

  if selected_reporters:
    rep_str_list = [str(r).lower() for r in selected_reporters]
    world_reporters_in_data = [
        r for r in d["reporter"].dropna().unique() if is_world_label(r)
    ]
    keep_list = set(rep_str_list) | {
        str(r).lower() for r in world_reporters_in_data
    }
    d = d[d["reporter"].astype(str).str.lower().isin(keep_list)]

  agg = d.groupby(
      ["ano", "reporter", "partner", "sh6", "sh6_desc"], as_index=False
  )["valor"].sum()
  return agg


# ==============================================================================
# 4. MOTOR DE CÁLCULO DE MÉTRICAS BILATERAIS (RCA E RSCA POR PARTNER)
# ==============================================================================


@st.cache_data(show_spinner="Calculando RCA/RSCA por Partner...")
def compute_comtrade_metrics(
    comtrade_tidy: pd.DataFrame, years: tuple
) -> pd.DataFrame:
  years = tuple(sorted(set(int(y) for y in years)))
  df_filtered = comtrade_tidy[comtrade_tidy["ano"].isin(years)].copy()
  if df_filtered.empty:
    return pd.DataFrame(
        columns=[
            "ano",
            "reporter",
            "partner",
            "sh6",
            "sh6_desc",
            "valor",
            "mundo_valor",
            "rca_partner",
            "rsca_partner",
        ]
    )

  year_partner_frames = []

  for yr in years:
    df_yr = df_filtered[df_filtered["ano"] == yr]
    if df_yr.empty:
      continue

    partners = df_yr["partner"].unique()

    for prt in partners:
      df_prt = df_yr[df_yr["partner"] == prt]
      if df_prt.empty:
        continue

      pv = df_prt.pivot_table(
          index="sh6",
          columns="reporter",
          values="valor",
          aggfunc="sum",
          fill_value=0.0,
      )
      sh6_descs = df_prt.groupby("sh6")["sh6_desc"].first()

      world_reporter = detect_world_label(pv.columns.tolist())
      uses_world_ref = (
          world_reporter is not None and world_reporter in pv.columns
      )

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
      share_mundo = (
          (X_wj / X_w) if X_w > 0 else pd.Series(0.0, index=X_wj.index)
      )

      rca_df = share_pais.div(share_mundo.replace(0, np.nan), axis=0)
      rca_df = rca_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
      rsca_df = ((rca_df - 1) / (rca_df + 1)).fillna(-1.0)

      val_long = pv_r.stack().rename("valor").reset_index()
      val_long.columns = ["sh6", "reporter", "valor"]
      rca_long = rca_df.stack().rename("rca_partner").reset_index()
      rca_long.columns = ["sh6", "reporter", "rca_partner"]
      rsca_long = rsca_df.stack().rename("rsca_partner").reset_index()
      rsca_long.columns = ["sh6", "reporter", "rsca_partner"]

      merged = val_long.merge(rca_long, on=["sh6", "reporter"]).merge(
          rsca_long, on=["sh6", "reporter"]
      )
      merged["ano"] = yr
      merged["partner"] = prt
      merged["sh6_desc"] = merged["sh6"].map(sh6_descs)
      merged["mundo_valor"] = merged["sh6"].map(X_wj)

      year_partner_frames.append(merged)

  if not year_partner_frames:
    return pd.DataFrame(
        columns=[
            "ano",
            "reporter",
            "partner",
            "sh6",
            "sh6_desc",
            "valor",
            "mundo_valor",
            "rca_partner",
            "rsca_partner",
        ]
    )

  out = pd.concat(year_partner_frames, ignore_index=True)
  return out[[
      "ano",
      "reporter",
      "partner",
      "sh6",
      "sh6_desc",
      "valor",
      "mundo_valor",
      "rca_partner",
      "rsca_partner",
  ]]


def compute_state_diversification_potentials(
    comexstat_uf: pd.DataFrame, comtrade_metrics: pd.DataFrame, year: int
) -> pd.DataFrame:
  if "ano" in comexstat_uf.columns:
    base_uf = comexstat_uf[comexstat_uf["ano"] == year].copy()
  else:
    base_uf = comexstat_uf.copy()

  required_cols = {"uf", "sh6_cod", "valor_fob"}
  if base_uf.empty or not required_cols.issubset(base_uf.columns):
    return pd.DataFrame()

  brazil_rep_name = next(
      (
          r
          for r in comtrade_metrics["reporter"].dropna().unique()
          if normalize_text(r) in ("brazil", "brasil", "bra")
      ),
      None,
  )
  if brazil_rep_name is None:
    return pd.DataFrame()

  br_metrics = comtrade_metrics[
      (comtrade_metrics["reporter"] == brazil_rep_name)
      & (comtrade_metrics["ano"] == year)
  ].copy()
  if br_metrics.empty:
    return pd.DataFrame()

  advantage = br_metrics.groupby(["sh6", "sh6_desc"], as_index=False).agg(
      rca=("rca_partner", "mean"),
      rsca=("rsca_partner", "mean"),
      mundo_valor=("mundo_valor", "sum"),
  )
  advantage = advantage[advantage["rca"] >= 1.0].copy()

  if advantage.empty:
    return pd.DataFrame()

  desc_cols = [
      c
      for c in COMEXSTAT_DESC_COLUMNS
      if c in base_uf.columns and c != "sh6_desc"
  ]
  if desc_cols:
    desc_map = base_uf.groupby("sh6_cod")[desc_cols].first()
  else:
    desc_map = pd.DataFrame(index=pd.Index([], name="sh6_cod"))

  ufs = sorted(base_uf["uf"].dropna().unique().tolist())
  sh6_advantage = advantage["sh6"].dropna().unique().tolist()
  if not ufs or not sh6_advantage:
    return pd.DataFrame()

  grid = pd.MultiIndex.from_product(
      [ufs, sh6_advantage], names=["uf", "sh6_cod"]
  ).to_frame(index=False)

  uf_totals = base_uf.groupby("uf")["valor_fob"].sum()
  grid["uf_total"] = grid["uf"].map(uf_totals).fillna(0.0)

  actual_exports = base_uf.groupby(["uf", "sh6_cod"], as_index=False)[
      "valor_fob"
  ].sum()
  grid = grid.merge(actual_exports, on=["uf", "sh6_cod"], how="left")
  grid["valor_fob"] = grid["valor_fob"].fillna(0.0)
  grid["share_local"] = np.where(
      grid["uf_total"] > 0, grid["valor_fob"] / grid["uf_total"], 0.0
  )

  adv_slim = advantage[
      ["sh6", "sh6_desc", "rca", "rsca", "mundo_valor"]
  ].rename(columns={"sh6": "sh6_cod"})
  grid = grid.merge(adv_slim, on="sh6_cod", how="left")

  if not desc_map.empty:
    grid = grid.merge(desc_map, on="sh6_cod", how="left")

  grid["potencial_score"] = grid["rca"] * (1 - grid["share_local"])
  grid["ja_exportado"] = grid["valor_fob"] > 0

  return grid.sort_values("potencial_score", ascending=False)


# ==============================================================================
# 5. ESTRUTURA VISUAL (DESIGN CUPERTINO, PASTEL & GLASSMORPHISM)
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

file_comtrade = st.sidebar.file_uploader(
    "1. UN Comtrade (CSV, XLSX, Parquet, JSON)",
    type=["csv", "xlsx", "xls", "parquet", "json"],
)
file_comexstat = st.sidebar.file_uploader(
    "2. ComexStat Brasil (Nacional)", type=["csv", "xlsx", "xls", "parquet"]
)
file_comexstat_uf = st.sidebar.file_uploader(
    "3. ComexStat por Estado (UF)", type=["csv", "xlsx", "xls", "parquet"]
)

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
    raw_cs_uf = read_any_file(
        file_comexstat_uf.getvalue(), file_comexstat_uf.name
    )
    st.session_state["comexstat_uf"] = standardize_comexstat(raw_cs_uf)
    st.session_state["_file_comexstat_uf_name"] = file_comexstat_uf.name

if "raw_comtrade" in st.session_state:
  df_ct = st.session_state["raw_comtrade"]
  cols = list(df_ct.columns)

  col_yr = find_column(df_ct, ["refyear"]) or COMTRADE_FIXED_COLS["year"]
  col_rep = (
      find_column(df_ct, ["reporterdesc"]) or COMTRADE_FIXED_COLS["reporter"]
  )
  col_prt = (
      find_column(df_ct, ["partnerdesc"]) or COMTRADE_FIXED_COLS["partner"]
  )
  col_sh6 = find_column(df_ct, ["cmdcode"]) or COMTRADE_FIXED_COLS["sh6"]
  col_desc = find_column(df_ct, ["cmddesc"]) or COMTRADE_FIXED_COLS["sh6_desc"]

  with st.sidebar.expander("⚙️ Configurações UN Comtrade", expanded=True):
    st.caption(
        "📌 **Campos fixos:** refYear, ReporterDesc, PartnerDesc, cmdCode,"
        " cmdDesc"
    )

    value_candidates = [
        c
        for c in cols
        if any(
            v in normalize_text(c) for v in ["value", "val", "fob", "cif", "primaryvalue"]
        )
    ]
    default_val_idx = (
        cols.index(value_candidates[0]) if value_candidates else 0
    )
    c_val = st.selectbox(
        "Selecione o Campo de Valor:", cols, index=default_val_idx
    )

    if col_prt in df_ct.columns:
      partners_list = sorted(
          df_ct[col_prt].dropna().astype(str).unique().tolist()
      )
      sel_partners = st.multiselect(
          "Filtrar Parceiros Comerciais (vazio = todos):",
          partners_list,
          default=[],
      )
    else:
      sel_partners = []

    if col_rep in df_ct.columns:
      reporters_list = sorted(
          df_ct[col_rep].dropna().astype(str).unique().tolist()
      )
      sel_reporters = st.multiselect(
          "Filtrar Países Reporters (vazio = todos):",
          reporters_list,
          default=[],
      )
    else:
      sel_reporters = []

    missing_cols = [
        c for c in [col_yr, col_rep, col_prt, col_sh6] if c not in df_ct.columns
    ]

    if missing_cols:
      st.error(
          "⚠️ As seguintes colunas obrigatórias não foram encontradas na"
          f" planilha: {', '.join(missing_cols)}"
      )
    else:
      if "comtrade_tidy" not in st.session_state:
        try:
          st.session_state["comtrade_tidy"] = standardize_comtrade(
              df_ct,
              col_yr,
              col_rep,
              col_prt,
              col_sh6,
              col_desc,
              c_val,
              sel_partners,
              sel_reporters,
          )
        except Exception as e:
          st.error(f"Erro ao processar automaticamente: {e}")

      if st.button("Aplicar / Re-processar Métricas", type="primary"):
        try:
          tidy_ct = standardize_comtrade(
              df_ct,
              col_yr,
              col_rep,
              col_prt,
              col_sh6,
              col_desc,
              c_val,
              sel_partners,
              sel_reporters,
          )
          st.session_state["comtrade_tidy"] = tidy_ct
          st.success("Dados do UN Comtrade processados com sucesso!")
        except Exception as exc:
          st.error(f"Falha ao processar a base do Comtrade: {exc}")

# ==============================================================================
# 7. MÓDULOS DA APLICAÇÃO
# ==============================================================================


# --- PÁGINA 1: UN COMTRADE RCA / RSCA ---
def page_comtrade_global():
  st.title("📊 Análise de RCA e RSCA por Partner (UN Comtrade)")
  st.caption(
      "Cálculo de Vantagem Comparativa Revelada (Balassa) e Simétrica (Laursen)"
      " calculada individualmente por Parceiro Comercial."
  )

  if "comtrade_tidy" not in st.session_state:
    st.info(
        "👈 Por favor, carregue e processe o arquivo do UN Comtrade na barra"
        " lateral."
    )
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
  df_metrics = compute_comtrade_metrics(tidy, tuple(years_range))

  if df_metrics.empty:
    st.warning("Não há dados suficientes para o intervalo de anos selecionado.")
    return

  f_col1, f_col2, f_col3, f_col4 = st.columns(4)
  with f_col1:
    reps = st.multiselect(
        "Filtrar por Reporter (País):", sorted(df_metrics["reporter"].unique())
    )
  with f_col2:
    prts = st.multiselect(
        "Filtrar por Partner (Parceiro):",
        sorted(df_metrics["partner"].unique()),
    )
  with f_col3:
    sh6s = st.multiselect(
        "Filtrar por SH6:", sorted(df_metrics["sh6"].unique())
    )
  with f_col4:
    only_advantage = st.checkbox("Apenas com RCA por Partner >= 1")

  filtered_df = df_metrics.copy()
  if reps:
    filtered_df = filtered_df[filtered_df["reporter"].isin(reps)]
  if prts:
    filtered_df = filtered_df[filtered_df["partner"].isin(prts)]
  if sh6s:
    filtered_df = filtered_df[filtered_df["sh6"].isin(sh6s)]
  if only_advantage:
    filtered_df = filtered_df[filtered_df["rca_partner"] >= 1.0]

  html_stats = f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-value">{len(filtered_df):,}</div>
            <div class="stat-label">Registros Processados</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{format_num(filtered_df['rca_partner'].mean(), 2)}</div>
            <div class="stat-label">Média do RCA por Partner</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{format_num(filtered_df['rsca_partner'].mean(), 2)}</div>
            <div class="stat-label">Média do RSCA por Partner</div>
        </div>
    </div>
    """
  st.markdown(html_stats, unsafe_allow_html=True)

  st.subheader("Resultados Detalhados por Partner")

  st.dataframe(
      filtered_df,
      column_config={
          "valor": st.column_config.NumberColumn(
              "Valor (US$)", format="$ %,.2f"
          ),
          "mundo_valor": st.column_config.NumberColumn(
              "Mundo Valor (US$)", format="$ %,.2f"
          ),
          "rca_partner": st.column_config.NumberColumn(
              "RCA por Partner", format="%.4f"
          ),
          "rsca_partner": st.column_config.NumberColumn(
              "RSCA por Partner", format="%.4f"
          ),
          "ano": st.column_config.NumberColumn("Ano", format="%d"),
          "partner": "Parceiro Comercial",
          "reporter": "País Declarante",
      },
      height=350,
  )

  st.divider()

  # --- GRÁFICOS SEPARADOS PARA RCA E RSCA (AMBOS EM DISPERSÃO) ---
  st.subheader("📈 Análise de Dispersão e Desempenho (RCA vs RSCA)")

  chart_col1, chart_col2 = st.columns(2)

  with chart_col1:
    st.markdown("#### **Dispersão do Índice RCA (Balassa)**")
    fig_rca = px.scatter(
        filtered_df,
        x="valor",
        y="rca_partner",
        color="partner",
        hover_data=["sh6", "sh6_desc", "reporter"],
        title="Dispersão do RCA por Valor Exportado",
        labels={
            "valor": "Valor Exportado (US$)",
            "rca_partner": "Índice RCA por Partner",
            "partner": "Parceiro",
        },
        template="plotly_white",
    )
    fig_rca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    fig_rca.add_hline(
        y=1.0,
        line_dash="dash",
        line_color=PASTEL_COLORS["blue_main"],
        annotation_text="Limiar de Vantagem (RCA = 1)",
    )
    st.plotly_chart(fig_rca, use_container_width=True)

  with chart_col2:
    st.markdown("#### **Dispersão do Índice RSCA (Laursen - Simétrico)**")
    fig_rsca = px.scatter(
        filtered_df,
        x="valor",
        y="rsca_partner",
        color="partner",
        hover_data=["sh6", "sh6_desc", "reporter"],
        title="Dispersão do RSCA por Valor Exportado (-1 a +1)",
        labels={
            "valor": "Valor Exportado (US$)",
            "rsca_partner": "RSCA por Partner",
            "partner": "Parceiro",
        },
        template="plotly_white",
    )
    fig_rsca.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    fig_rsca.add_hline(
        y=0.0,
        line_dash="dash",
        line_color=PASTEL_COLORS["red_main"],
        annotation_text="Ponto Neutro (RSCA = 0)",
    )
    st.plotly_chart(fig_rsca, use_container_width=True)

  if len(years_range) > 1:
    st.divider()
    st.subheader("📉 Evolução Temporal Separada (RCA e RSCA)")

    c1, c2, c3 = st.columns(3)
    with c1:
      trend_reporter = st.selectbox(
          "País:", sorted(df_metrics["reporter"].unique()), key="trend_rep"
      )
    with c2:
      trend_partner = st.selectbox(
          "Parceiro:",
          sorted(
              df_metrics[df_metrics["reporter"] == trend_reporter][
                  "partner"
              ].unique()
          ),
          key="trend_prt",
      )
    opts_sh6 = sorted(
        df_metrics[
            (df_metrics["reporter"] == trend_reporter)
            & (df_metrics["partner"] == trend_partner)
        ]["sh6"].unique()
    )
    with c3:
      trend_sh6 = st.selectbox("Produto (SH6):", opts_sh6, key="trend_sh6")

    trend_df = df_metrics[
        (df_metrics["reporter"] == trend_reporter)
        & (df_metrics["partner"] == trend_partner)
        & (df_metrics["sh6"] == trend_sh6)
    ].sort_values("ano")

    if not trend_df.empty:
      t_col1, t_col2 = st.columns(2)

      with t_col1:
        fig_trend_rca = px.line(
            trend_df,
            x="ano",
            y="rca_partner",
            markers=True,
            title=(
                f"Evolução do RCA — {trend_reporter} x {trend_partner} — SH6"
                f" {trend_sh6}"
            ),
            labels={"rca_partner": "Índice RCA", "ano": "Ano"},
            template="plotly_white",
        )
        fig_trend_rca.update_traces(line_color=PASTEL_COLORS["blue_main"])
        fig_trend_rca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        fig_trend_rca.add_hline(
            y=1.0,
            line_dash="dot",
            line_color=PASTEL_COLORS["amber_main"],
            annotation_text="RCA = 1.0",
        )
        st.plotly_chart(fig_trend_rca, use_container_width=True)

      with t_col2:
        fig_trend_rsca = px.line(
            trend_df,
            x="ano",
            y="rsca_partner",
            markers=True,
            title=(
                f"Evolução do RSCA — {trend_reporter} x {trend_partner} — SH6"
                f" {trend_sh6}"
            ),
            labels={"rsca_partner": "Índice RSCA", "ano": "Ano"},
            template="plotly_white",
        )
        fig_trend_rsca.update_traces(line_color=PASTEL_COLORS["green_main"])
        fig_trend_rsca.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        fig_trend_rsca.add_hline(
            y=0.0,
            line_dash="dot",
            line_color=PASTEL_COLORS["red_main"],
            annotation_text="RSCA = 0.0",
        )
        st.plotly_chart(fig_trend_rsca, use_container_width=True)


# --- PÁGINA 2: CRUZAMENTO BRASIL COMEXSTAT X COMTRADE ---
def page_comexstat_cross():
  st.title(
      "🇧🇷 Cruzamento das Exportações do Brasil com Competitividade Global"
  )
  st.caption(
      "Cruzamento entre a pauta detalhada do ComexStat e os índices globais de"
      " RCA/RSCA por parceiro comercial."
  )

  if "comexstat" not in st.session_state or "comtrade_tidy" not in st.session_state:
    st.warning(
        "⚠️ É necessário carregar AMBOS os arquivos (ComexStat Brasil e UN"
        " Comtrade) na barra lateral."
    )
    return

  cs = st.session_state["comexstat"]
  ct = st.session_state["comtrade_tidy"]

  if "ano" not in cs.columns or "sh6_cod" not in cs.columns:
    st.error(
        "A base do ComexStat Brasil precisa conter colunas de Ano e Código"
        " SH6."
    )
    return

  anos_cs = sorted(int(a) for a in cs["ano"].dropna().unique().tolist())
  selected_year = st.selectbox(
      "Selecione o Ano para Análise Cruzada:", anos_cs, index=len(anos_cs) - 1
  )

  ct_metrics = compute_comtrade_metrics(ct, (selected_year,))
  if ct_metrics.empty:
    st.warning("Não há dados do Comtrade para o ano selecionado.")
    return

  brazil_rep = next(
      (
          r
          for r in ct_metrics["reporter"].unique()
          if normalize_text(r) in ("brazil", "brasil", "bra")
      ),
      None,
  )
  if brazil_rep is None:
    st.error(
        "Não foi encontrado um reporter 'Brazil'/'Brasil' nos dados do Comtrade"
        " processados."
    )
    return

  br_ct = (
      ct_metrics[ct_metrics["reporter"] == brazil_rep]
      .groupby(["sh6"])
      .agg(
          rca=("rca_partner", "mean"),
          rsca=("rsca_partner", "mean"),
          mundo_valor=("mundo_valor", "sum"),
      )
      .reset_index()
  )

  merged = pd.merge(
      cs[cs["ano"] == selected_year],
      br_ct,
      left_on="sh6_cod",
      right_on="sh6",
      how="left",
  )
  merged["rca"] = merged["rca"].fillna(0)
  merged["rsca"] = merged["rsca"].fillna(-1)

  st.subheader("🔍 Filtros de Segmentação da Pauta Exportadora")
  col1, col2, col3 = st.columns(3)
  with col1:
    f_sh6 = st.multiselect("SH6:", sorted(merged["sh6_cod"].dropna().unique()))
    f_cuci = st.multiselect(
        "CUCI Grupo:",
        sorted(merged["cuci_desc"].dropna().unique())
        if "cuci_desc" in merged.columns
        else [],
    )
  with col2:
    f_isic_div = st.multiselect(
        "ISIC Divisão:",
        sorted(merged["isic_div_desc"].dropna().unique())
        if "isic_div_desc" in merged.columns
        else [],
    )
    f_isic_sec = st.multiselect(
        "ISIC Seção:",
        sorted(merged["isic_sec_desc"].dropna().unique())
        if "isic_sec_desc" in merged.columns
        else [],
    )
  with col3:
    f_cgce1 = st.multiselect(
        "CGCE Nível 1:",
        sorted(merged["cgce1_desc"].dropna().unique())
        if "cgce1_desc" in merged.columns
        else [],
    )
    f_cgce2 = st.multiselect(
        "CGCE Nível 2:",
        sorted(merged["cgce2_desc"].dropna().unique())
        if "cgce2_desc" in merged.columns
        else [],
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
            <div class="stat-label">Produtos com RCA Médio ≥ 1</div>
        </div>
    </div>
    """
  st.markdown(html_stats, unsafe_allow_html=True)

  st.subheader("📊 Distribuição de Exportação x Competitividade por Setor")
  group_opt = st.selectbox(
      "Agrupar Visualização por:",
      [
          "CUCI Grupo",
          "ISIC Divisão",
          "ISIC Seção",
          "CGCE Nível 1",
          "CGCE Nível 2",
      ],
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
            RCA_Medio=("rca", "mean"),
            RSCA_Medio=("rsca", "mean"),
            N_Produtos=("sh6_cod", "nunique"),
        )
        .reset_index()
        .sort_values("Valor_FOB", ascending=False)
    )

    col_sec1, col_sec2 = st.columns(2)

    with col_sec1:
      fig_sec_rca = px.bar(
          agg_sector.head(15),
          x="Valor_FOB",
          y=selected_col,
          orientation="h",
          color="RCA_Medio",
          color_continuous_scale=["#3b82f6", "#10b981"],
          title=f"Top 15 Setores por Valor e RCA Médio ({group_opt})",
          labels={
              "Valor_FOB": "Valor FOB (US$)",
              selected_col: "Setor",
              "RCA_Medio": "RCA Médio",
          },
          template="plotly_white",
      )
      fig_sec_rca.update_layout(
          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
      )
      st.plotly_chart(fig_sec_rca, use_container_width=True)

    with col_sec2:
      fig_sec_rsca = px.bar(
          agg_sector.head(15),
          x="Valor_FOB",
          y=selected_col,
          orientation="h",
          color="RSCA_Medio",
          color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
          title=f"Top 15 Setores por Valor e RSCA Médio ({group_opt})",
          labels={
              "Valor_FOB": "Valor FOB (US$)",
              selected_col: "Setor",
              "RSCA_Medio": "RSCA Médio",
          },
          template="plotly_white",
      )
      fig_sec_rsca.update_layout(
          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
      )
      st.plotly_chart(fig_sec_rsca, use_container_width=True)

    st.dataframe(
        agg_sector,
        column_config={
            "Valor_FOB": st.column_config.NumberColumn(
                "Valor FOB (US$)", format="$ %,.2f"
            ),
            "RCA_Medio": st.column_config.NumberColumn(
                "RCA Médio", format="%.2f"
            ),
            "RSCA_Medio": st.column_config.NumberColumn(
                "RSCA Médio", format="%.2f"
            ),
            "N_Produtos": st.column_config.NumberColumn(
                "Produtos SH6", format="%d"
            ),
        },
    )
  else:
    st.info(f"A coluna '{group_opt}' não foi encontrada na base carregada.")


# --- PÁGINA 3: POTENCIAL DE DIVERSIFICAÇÃO POR ESTADO (UF) ---
def page_state_diversification():
  st.title("🗺️ Potencial de Diversificação Exportadora por Estado (UF)")
  st.caption(
      "Cruzamento subnacional para identificar produtos com alta"
      " competitividade nacional (RCA ≥ 1) subaproveitados nos estados."
  )

  if (
      "comexstat_uf" not in st.session_state
      or "comtrade_tidy" not in st.session_state
  ):
    st.warning(
        "⚠️ É necessário carregar a planilha do ComexStat por Estado (UF) e o"
        " UN Comtrade."
    )
    return

  cs_uf = st.session_state["comexstat_uf"]
  ct = st.session_state["comtrade_tidy"]

  if (
      "ano" not in cs_uf.columns
      or "uf" not in cs_uf.columns
      or "sh6_cod" not in cs_uf.columns
  ):
    st.error(
        "A base do ComexStat por Estado precisa conter colunas de Ano, UF e"
        " Código SH6."
    )
    return

  anos_uf = sorted(int(a) for a in cs_uf["ano"].dropna().unique().tolist())
  selected_year = st.selectbox(
      "Ano de Análise Subnacional:", anos_uf, index=len(anos_uf) - 1
  )

  ct_metrics = compute_comtrade_metrics(ct, (selected_year,))
  if ct_metrics.empty:
    st.warning("Não há dados do Comtrade para o ano selecionado.")
    return

  df_potencial = compute_state_diversification_potentials(
      cs_uf, ct_metrics, selected_year
  )

  if df_potencial.empty:
    st.error(
        "Não foi possível calcular o potencial com os dados fornecidos."
        " Verifique se o Brasil aparece como reporter no Comtrade e se há"
        " produtos com RCA ≥ 1 no ano selecionado."
    )
    return

  st.subheader(
      "🧭 Radar de Oportunidades Estaduais (Matriz de Vantagem & Presença)"
  )

  n_alto_potencial = df_potencial[
      (df_potencial["potencial_score"] > 1.0) & (~df_potencial["ja_exportado"])
  ]["sh6_cod"].nunique()
  n_vantagem_nacional = df_potencial["sh6_cod"].nunique()
  val_mercado_oportunidade = (
      df_potencial[
          (df_potencial["potencial_score"] > 1.0)
          & (~df_potencial["ja_exportado"])
      ]
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
                <div class="qc-desc">Combinações Estado x Produto em que o Brasil possui vantagem comparativa global (RCA ≥ 1), mas o estado ainda não exporta o produto.</div>
            </div>
            <div class="qc-bottom">
                <div>
                    <div class="qc-count">{n_alto_potencial:,}</div>
                    <div class="qc-count-label">produtos-chave (média nacional)</div>
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
                <div class="qc-desc">Total de produtos no portfólio brasileiro com alto índice de especialização e inserção no comércio internacional, avaliados em todos os estados.</div>
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

  st.subheader(
      "🏆 Ranking de Estados por Score de Potencial de Diversificação"
  )

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
      labels={
          "Score_Potencial_Total": "Score de Potencial",
          "uf": "Estado",
          "Exportacao_Atual_FOB": "Exportação Atual (US$)",
      },
      template="plotly_white",
  )
  fig_uf_rank.update_layout(
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      yaxis=dict(categoryorder="total ascending"),
  )
  st.plotly_chart(fig_uf_rank, use_container_width=True)

  st.dataframe(
      rank_uf,
      column_config={
          "Score_Potencial_Total": st.column_config.NumberColumn(
              "Score de Potencial Total", format="%.2f"
          ),
          "Exportacao_Atual_FOB": st.column_config.NumberColumn(
              "Exportação Atual (US$)", format="$ %,.2f"
          ),
          "Produtos_Nao_Explorados": st.column_config.NumberColumn(
              "Produtos Não Explorados", format="%d"
          ),
      },
  )

  st.divider()

  st.subheader("🔍 Detalhamento e Segmentação por Estado (UF)")

  col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
  with col_sel1:
    uf_target = st.selectbox(
        "Selecione o Estado (UF):", sorted(df_potencial["uf"].unique())
    )

  df_uf_filtered = df_potencial[df_potencial["uf"] == uf_target]

  with col_sel2:
    cuci_options = (
        sorted(df_uf_filtered["cuci_desc"].dropna().unique())
        if "cuci_desc" in df_uf_filtered.columns
        else []
    )
    cuci_target = st.multiselect("Filtrar por CUCI Grupo:", cuci_options)
  with col_sel3:
    sh6_options = sorted(df_uf_filtered["sh6_cod"].dropna().unique())
    sh6_target = st.multiselect("Filtrar por SH6:", sh6_options)
  with col_sel4:
    only_new = st.checkbox(
        "Somente produtos NÃO exportados pelo estado", value=False
    )

  df_uf_seg = df_uf_filtered.copy()
  if cuci_target and "cuci_desc" in df_uf_seg.columns:
    df_uf_seg = df_uf_seg[df_uf_seg["cuci_desc"].isin(cuci_target)]
  if sh6_target:
    df_uf_seg = df_uf_seg[df_uf_seg["sh6_cod"].isin(sh6_target)]
  if only_new:
    df_uf_seg = df_uf_seg[~df_uf_seg["ja_exportado"]]

  st.markdown(f"### Oportunidades Prioritárias para **{uf_target}**")

  top_n = df_uf_seg.sort_values("potencial_score", ascending=False).head(15)
  if not top_n.empty:
    label_col = (
        "sh6_desc"
        if "sh6_desc" in top_n.columns and top_n["sh6_desc"].notna().any()
        else "sh6_cod"
    )
    color_col = (
        "cuci_desc"
        if "cuci_desc" in top_n.columns and top_n["cuci_desc"].notna().any()
        else None
    )
    fig_top = px.bar(
        top_n,
        x="potencial_score",
        y=label_col,
        orientation="h",
        color=color_col,
        title=f"Top produtos por potencial de diversificação — {uf_target}",
        labels={
            "potencial_score": "Score de Potencial",
            label_col: "Produto (SH6)",
            "cuci_desc": "CUCI Grupo",
        },
        template="plotly_white",
    )
    fig_top.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_top, use_container_width=True)

  disp_cols = [
      "sh6_cod",
      "sh6_desc",
      "cuci_desc",
      "rca",
      "rsca",
      "valor_fob",
      "share_local",
      "potencial_score",
      "ja_exportado",
  ]
  available_disp_cols = [c for c in disp_cols if c in df_uf_seg.columns]

  st.dataframe(
      df_uf_seg[available_disp_cols]
      .sort_values("potencial_score", ascending=False)
      .head(50),
      column_config={
          "rca": st.column_config.NumberColumn("RCA Médio", format="%.2f"),
          "rsca": st.column_config.NumberColumn("RSCA Médio", format="%.2f"),
          "valor_fob": st.column_config.NumberColumn(
              "Valor FOB (US$)", format="$ %,.2f"
          ),
          "share_local": st.column_config.NumberColumn(
              "Participação Local", format="%.2%%"
          ),
          "potencial_score": st.column_config.NumberColumn(
              "Score Potencial", format="%.2f"
          ),
      },
      height=420,
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
