# -*- coding: utf-8 -*-
"""
==============================================================================
 SISTEMA DE ANÁLISE DE DIVERSIFICAÇÃO DAS EXPORTAÇÕES BRASILEIRAS
 (Cupertino Executive Edition) — v7 (refatorado)
==============================================================================

Principais mudanças desta refatoração em relação à v6:

1. METODOLOGIA — RCA GLOBAL vs RCA BILATERAL
   A v6 usava, em TODAS as páginas, uma "RCA bilateral" calculada mercado a
   mercado (país x parceiro) e depois tirava a MÉDIA dessas RCAs bilaterais
   para representar a "vantagem comparativa nacional" do Brasil. Isso é
   estatisticamente frágil: RCA é uma razão não-limitada (pode explodir para
   valores muito altos quando o denominador é pequeno) e sua média entre
   parceiros distintos não equivale à RCA clássica de Balassa, além de mudar
   de acordo com quais parceiros o usuário incluiu no arquivo.

   Esta versão separa os dois conceitos:
   - RCA GLOBAL (Balassa clássico): Brasil (ou outro reporter) vs Mundo,
     calculada a partir das linhas em que partner = "World" na base do
     Comtrade. Usada como a métrica "oficial" de vantagem comparativa do
     país nas páginas 2 e 3.
   - RCA BILATERAL: mantida na página 1, pois ali o objetivo é justamente
     comparar a inserção de um país em mercados específicos — isso é
     análise de mercado, não vantagem comparativa nacional.
   Quando a base carregada não contém partner = "World", o app avisa o
   usuário e cai para a média bilateral como aproximação (comportamento da
   v6), deixando isso explícito na tela em vez de silencioso.

2. SCORE DE POTENCIAL DE DIVERSIFICAÇÃO (página 3)
   Antes: potencial_score = RCA * (1 - share_local) — usa RCA "cru", que não
   tem limite superior e pode fazer alguns produtos dominarem o ranking só
   por causa de um RCA de 300, por exemplo.
   Agora: potencial_score = RSCA_clipped[0,1] * (1 - share_local) — usa o
   RSCA (limitado entre -1 e +1, já filtrado para produtos com RCA ≥ 1, ou
   seja RSCA ≥ 0), o que produz um ranking mais robusto e comparável.

3. PERFORMANCE
   O cálculo de RCA/RSCA bilateral foi reescrito de um laço Python
   (ano × parceiro, com pivot_table a cada iteração) para operações
   vetorizadas de pandas (groupby + merge), preservando exatamente a mesma
   fórmula, mas de forma mais rápida e legível.

4. CORREÇÃO DE BUG DE FORMATAÇÃO
   A coluna "Participação Local" usava format="%.2%%%", uma string de
   formatação inválida. Agora a participação é convertida para pontos
   percentuais antes de ser exibida, com format="%.1f%%".

5. DRY / DESIGN
   As dezenas de blocos de HTML repetidos para os cartões de métricas (KPIs)
   foram substituídas por uma função utilitária `render_metric_cards`. Os
   tokens de design (cores, espaçamento) foram mantidos e levemente
   ajustados; foi adicionado um selo de metodologia (Global vs Bilateral) e
   um expander com as fórmulas em cada página analítica.
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
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def is_world_label(value) -> bool:
    nv = normalize_text(value)
    return nv in ("world", "mundo") or nv.startswith("world") or nv.startswith("mundo")


def detect_world_label(values) -> str | None:
    for v in values:
        if is_world_label(v):
            return v
    return None


BRAZIL_NAME_KEYS = ("brazil", "brasil", "bra")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divisão vetorizada tolerante a zero/NaN/inf, retornando 0.0 nesses casos."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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

    # IMPORTANTE: os filtros de "Parceiros" e "Reporters" abaixo afetam
    # apenas a página de RCA/RSCA bilateral. Para a RCA global (Balassa)
    # funcionar corretamente é necessário que o arquivo original também
    # contenha, para os reporters de interesse, linhas com partner = "World".
    # Se o usuário filtrar parceiros e isso excluir "World", a RCA global
    # deixa de estar disponível e o app cai automaticamente para a média
    # bilateral (ver get_country_metrics).
    if selected_partners:
        prt_str_list = [str(p).lower() for p in selected_partners]
        world_partners_in_data = [p for p in d["partner"].dropna().unique() if is_world_label(p)]
        keep_list = set(prt_str_list) | {str(p).lower() for p in world_partners_in_data}
        d = d[d["partner"].astype(str).str.lower().isin(keep_list)]

    if selected_reporters:
        rep_str_list = [str(r).lower() for r in selected_reporters]
        world_reporters_in_data = [r for r in d["reporter"].dropna().unique() if is_world_label(r)]
        keep_list = set(rep_str_list) | {str(r).lower() for r in world_reporters_in_data}
        d = d[d["reporter"].astype(str).str.lower().isin(keep_list)]

    agg = d.groupby(["ano", "reporter", "partner", "sh6", "sh6_desc"], as_index=False)["valor"].sum()
    return agg


# ==============================================================================
# 3. CÁLCULO DE MÉTRICAS DE VANTAGEM COMPARATIVA
# ==============================================================================
#
#   RCA (Balassa, 1965):    RCA_ij = (X_ij / X_i) / (X_wj / X_w)
#   RSCA (Laursen, 1998):   RSCA_ij = (RCA_ij - 1) / (RCA_ij + 1)   ∈ [-1, +1]
#
#   Onde, na versão GLOBAL (clássica):
#     X_ij = exportações do país i para o Mundo, produto j
#     X_i  = exportações totais do país i para o Mundo
#     X_wj = exportações mundiais do produto j
#     X_w  = exportações mundiais totais
#
#   Na versão BILATERAL (mercado a mercado, usada na página 1):
#     X_ij = exportações do reporter i para um parceiro p, produto j
#     X_i  = exportações totais do reporter i para esse mesmo parceiro p
#     X_wj = importações totais do parceiro p, produto j (a partir do Mundo)
#     X_w  = importações totais do parceiro p (a partir do Mundo)
#   Ou seja, mede a inserção do reporter i na pauta de importações de um
#   parceiro específico — não é uma vantagem comparativa "global".
# ==============================================================================

BILATERAL_METRICS_COLUMNS = [
    "ano", "reporter", "partner", "sh6", "sh6_desc", "valor",
    "mundo_valor", "rca_partner", "rsca_partner",
]


@st.cache_data(show_spinner="Calculando RCA/RSCA bilateral por parceiro...")
def compute_bilateral_metrics(comtrade_tidy: pd.DataFrame, years: tuple) -> pd.DataFrame:
    """RCA/RSCA de cada reporter em cada mercado (parceiro), vetorizado."""
    years = tuple(sorted(set(int(y) for y in years)))
    df = comtrade_tidy[comtrade_tidy["ano"].isin(years)].copy()
    if df.empty:
        return pd.DataFrame(columns=BILATERAL_METRICS_COLUMNS)

    world_label = detect_world_label(df["reporter"].unique().tolist())

    if world_label is not None:
        world_rows = df[df["reporter"] == world_label]
        reporters_df = df[df["reporter"] != world_label].copy()
        world_totals = (
            world_rows.groupby(["ano", "partner", "sh6"], as_index=False)["valor"]
            .sum().rename(columns={"valor": "mundo_valor"})
        )
    else:
        # Sem reporter "World" explícito: usa a soma de todos os reporters
        # presentes como proxy do total importado pelo parceiro (mesmo
        # comportamento de fallback da versão anterior).
        reporters_df = df.copy()
        world_totals = (
            df.groupby(["ano", "partner", "sh6"], as_index=False)["valor"]
            .sum().rename(columns={"valor": "mundo_valor"})
        )

    if reporters_df.empty:
        return pd.DataFrame(columns=BILATERAL_METRICS_COLUMNS)

    reporters_df = reporters_df.merge(world_totals, on=["ano", "partner", "sh6"], how="left")
    reporters_df["mundo_valor"] = reporters_df["mundo_valor"].fillna(0.0)

    reporter_totals = reporters_df.groupby(["ano", "partner", "reporter"])["valor"].transform("sum")

    world_partner_totals = (
        world_totals.groupby(["ano", "partner"], as_index=False)["mundo_valor"]
        .sum().rename(columns={"mundo_valor": "X_w"})
    )
    reporters_df = reporters_df.merge(world_partner_totals, on=["ano", "partner"], how="left")
    reporters_df["X_w"] = reporters_df["X_w"].fillna(0.0)

    share_pais = _safe_divide(reporters_df["valor"], reporter_totals)
    share_mundo = _safe_divide(reporters_df["mundo_valor"], reporters_df["X_w"])

    reporters_df["rca_partner"] = _safe_divide(share_pais, share_mundo)
    reporters_df["rsca_partner"] = (reporters_df["rca_partner"] - 1) / (reporters_df["rca_partner"] + 1)

    return reporters_df[BILATERAL_METRICS_COLUMNS]


@st.cache_data(show_spinner="Calculando RCA global (Balassa clássico)...")
def compute_global_rca(comtrade_tidy: pd.DataFrame, years: tuple) -> pd.DataFrame:
    """RCA/RSCA clássicos (reporter vs Mundo), a partir de linhas partner = 'World'.

    Retorna DataFrame vazio se a base não contiver o parceiro "World".
    """
    years = tuple(sorted(set(int(y) for y in years)))
    df = comtrade_tidy[comtrade_tidy["ano"].isin(years)].copy()
    empty = pd.DataFrame(columns=["ano", "reporter", "sh6", "sh6_desc", "valor", "mundo_valor", "rca_global", "rsca_global"])
    if df.empty:
        return empty

    world_label = detect_world_label(df["partner"].unique().tolist())
    if world_label is None:
        return empty

    gdf = df[df["partner"] == world_label].copy()
    if gdf.empty:
        return empty

    gdf = gdf.groupby(["ano", "reporter", "sh6", "sh6_desc"], as_index=False)["valor"].sum()

    reporter_totals = gdf.groupby(["ano", "reporter"])["valor"].transform("sum")
    product_totals = gdf.groupby(["ano", "sh6"])["valor"].transform("sum")
    year_totals = gdf.groupby(["ano"])["valor"].transform("sum")

    share_pais = _safe_divide(gdf["valor"], reporter_totals)
    share_mundo = _safe_divide(product_totals, year_totals)

    gdf["mundo_valor"] = product_totals
    gdf["rca_global"] = _safe_divide(share_pais, share_mundo)
    gdf["rsca_global"] = (gdf["rca_global"] - 1) / (gdf["rca_global"] + 1)

    return gdf


def get_country_metrics(
    comtrade_tidy: pd.DataFrame,
    year: int,
    bilateral_fallback: pd.DataFrame | None = None,
    target_name_keys: tuple = BRAZIL_NAME_KEYS,
) -> tuple[pd.DataFrame, str]:
    """Retorna (df, metodologia) com colunas sh6, sh6_desc, rca, rsca, mundo_valor
    para o país-alvo (Brasil por padrão) no ano informado.

    metodologia ∈ {"global", "bilateral_media", "indisponivel"}
    - "global": RCA clássico de Balassa (reporter vs Mundo). Preferido.
    - "bilateral_media": fallback — média das RCAs bilaterais do país contra
      os parceiros presentes na base (aproximação, usada apenas quando não
      há partner = "World" nos dados carregados).
    - "indisponivel": não foi possível localizar o país-alvo em nenhuma base.
    """
    global_df = compute_global_rca(comtrade_tidy, (year,))
    if not global_df.empty:
        country = next(
            (r for r in global_df["reporter"].dropna().unique() if normalize_text(r) in target_name_keys),
            None,
        )
        if country is not None:
            sub = (
                global_df[global_df["reporter"] == country]
                .groupby(["sh6", "sh6_desc"], as_index=False)
                .agg(rca=("rca_global", "mean"), rsca=("rsca_global", "mean"), mundo_valor=("mundo_valor", "mean"))
            )
            return sub, "global"

    if bilateral_fallback is not None and not bilateral_fallback.empty:
        country = next(
            (r for r in bilateral_fallback["reporter"].dropna().unique() if normalize_text(r) in target_name_keys),
            None,
        )
        if country is not None:
            sub = (
                bilateral_fallback[bilateral_fallback["reporter"] == country]
                .groupby(["sh6", "sh6_desc"], as_index=False)
                .agg(rca=("rca_partner", "mean"), rsca=("rsca_partner", "mean"), mundo_valor=("mundo_valor", "sum"))
            )
            return sub, "bilateral_media"

    return pd.DataFrame(columns=["sh6", "sh6_desc", "rca", "rsca", "mundo_valor"]), "indisponivel"


def render_methodology_badge(methodology: str):
    if methodology == "global":
        st.caption("✅ **Metodologia:** RCA Global (Balassa clássico) — Brasil vs Mundo, a partir de linhas partner = \"World\".")
    elif methodology == "bilateral_media":
        st.warning(
            "⚠️ A base carregada não contém o parceiro **\"World\"**, então a RCA global não pôde ser calculada. "
            "Usando, como aproximação, a **média das RCAs bilaterais** do Brasil contra os parceiros presentes no arquivo. "
            "Para maior precisão, inclua na extração do Comtrade as linhas com partner = \"World\"."
        )
    else:
        st.error("Não foi possível localizar o Brasil como reporter na base do UN Comtrade carregada.")


def render_methodology_expander():
    with st.expander("ℹ️ Metodologia e fórmulas utilizadas"):
        st.markdown(
            """
**RCA — Vantagem Comparativa Revelada (Balassa, 1965)**

$$RCA_{ij} = \\dfrac{X_{ij} / X_{i}}{X_{wj} / X_{w}}$$

- $X_{ij}$: exportações do produto *j* pelo país/parceiro *i*
- $X_{i}$: exportações totais do país/parceiro *i*
- $X_{wj}$: exportações mundiais do produto *j*
- $X_{w}$: exportações mundiais totais

$RCA > 1$ indica vantagem comparativa revelada no produto.

**RSCA — RCA Simétrico (Laursen, 1998)**

$$RSCA_{ij} = \\dfrac{RCA_{ij} - 1}{RCA_{ij} + 1} \\in [-1, +1]$$

Limita o índice a um intervalo simétrico, tornando-o comparável entre
produtos e adequado para médias e rankings. $RSCA > 0$ equivale a $RCA > 1$.

**RCA Global vs. RCA Bilateral**

- *Global* (páginas "Cruzamento Nacional" e "Potencial por Estado"): compara
  as exportações do Brasil para o **Mundo** com as exportações mundiais do
  produto — é a vantagem comparativa do país como um todo.
- *Bilateral* (página "RCA/RSCA Global (UN Comtrade)"): compara as
  exportações de um reporter para **um parceiro específico** com as
  importações totais desse parceiro — mede o quão bem o produto está
  inserido naquele mercado específico, não a vantagem comparativa nacional.

**Score de Potencial de Diversificação (página "Potencial por Estado")**

$$\\text{score} = \\text{RSCA}^{+} \\times (1 - \\text{participação local})$$

onde $\\text{RSCA}^{+}$ é o RSCA nacional do produto (limitado a [0, 1], já
que só entram produtos com $RCA \\ge 1$) e *participação local* é o quanto o
produto já representa na pauta do estado. Produtos com alta vantagem
comparativa nacional e baixa presença na pauta estadual pontuam mais alto.
            """
        )


def compute_state_diversification_potentials(
    comexstat_uf: pd.DataFrame, advantage_df: pd.DataFrame, year: int
) -> pd.DataFrame:
    """advantage_df: colunas sh6, sh6_desc, rca, rsca, mundo_valor — já filtrado
    para produtos com vantagem comparativa nacional (rca >= 1)."""
    if "ano" in comexstat_uf.columns:
        base_uf = comexstat_uf[comexstat_uf["ano"] == year].copy()
    else:
        base_uf = comexstat_uf.copy()

    required_cols = {"uf", "sh6_cod", "valor_fob"}
    if base_uf.empty or not required_cols.issubset(base_uf.columns) or advantage_df.empty:
        return pd.DataFrame()

    adv = advantage_df.rename(columns={"sh6": "sh6_cod"})

    desc_cols = [c for c in COMEXSTAT_DESC_COLUMNS if c in base_uf.columns and c != "sh6_desc"]
    if desc_cols:
        desc_map = base_uf.groupby("sh6_cod")[desc_cols].first()
    else:
        desc_map = pd.DataFrame(index=pd.Index([], name="sh6_cod"))

    ufs = sorted(base_uf["uf"].dropna().unique().tolist())
    sh6_advantage = adv["sh6_cod"].dropna().unique().tolist()
    if not ufs or not sh6_advantage:
        return pd.DataFrame()

    grid = pd.MultiIndex.from_product([ufs, sh6_advantage], names=["uf", "sh6_cod"]).to_frame(index=False)

    uf_totals = base_uf.groupby("uf")["valor_fob"].sum()
    grid["uf_total"] = grid["uf"].map(uf_totals).fillna(0.0)

    actual_exports = base_uf.groupby(["uf", "sh6_cod"], as_index=False)["valor_fob"].sum()
    grid = grid.merge(actual_exports, on=["uf", "sh6_cod"], how="left")
    grid["valor_fob"] = grid["valor_fob"].fillna(0.0)
    grid["share_local"] = np.where(grid["uf_total"] > 0, grid["valor_fob"] / grid["uf_total"], 0.0)

    grid = grid.merge(adv[["sh6_cod", "sh6_desc", "rca", "rsca", "mundo_valor"]], on="sh6_cod", how="left")

    if not desc_map.empty:
        grid = grid.merge(desc_map, on="sh6_cod", how="left")

    # RSCA já é limitado a [-1, +1]; como só entram produtos com rca >= 1
    # (rsca >= 0), o clip abaixo é apenas uma salvaguarda numérica.
    rsca_weight = grid["rsca"].clip(lower=0.0, upper=1.0)
    grid["potencial_score"] = rsca_weight * (1 - grid["share_local"])
    grid["ja_exportado"] = grid["valor_fob"] > 0

    return grid.sort_values("potencial_score", ascending=False)


# ==============================================================================
# 4. DESIGN EXEC — CSS E COMPONENTES REUTILIZÁVEIS
# ==============================================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        :root{--bg:#f5f7fb;--card:#fff;--border:#e5e7eb;--border2:#d5dbe5;--text:#101828;--muted:#667085;
              --blue:#2563eb;--green:#059669;--red:#dc2626;--amber:#d97706;--purple:#7c3aed;
              --shadow:0 1px 2px rgba(16,24,40,.03),0 8px 24px rgba(16,24,40,.045)}
        html,body,[class*="st-"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Plus Jakarta Sans",sans-serif}
        .stApp{background:var(--bg);color:var(--text)}
        .block-container{max-width:1480px;padding-top:2rem;padding-bottom:3.5rem}
        h1{font-weight:800!important;letter-spacing:-.045em!important;margin-bottom:.15rem!important} h2,h3{font-weight:750!important;letter-spacing:-.025em!important}
        /* SIDEBAR */
        [data-testid="stSidebar"]{background:#fbfcfe!important;border-right:1px solid var(--border)!important}
        [data-testid="stSidebar"]>div:first-child{padding:1.25rem 1rem 1.5rem}
        [data-testid="stSidebar"] hr{border-color:var(--border)!important;margin:1rem 0}
        [data-testid="stSidebar"] [data-testid="stRadio"]>label{font-size:.72rem!important;font-weight:800!important;color:#475467!important;text-transform:uppercase;letter-spacing:.06em}
        [data-testid="stSidebar"] [role="radiogroup"]{gap:.2rem}
        [data-testid="stSidebar"] [role="radiogroup"] label{border:1px solid transparent;border-radius:10px;padding:.48rem .55rem;transition:.15s ease}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#f1f5f9;border-color:var(--border)}
        .sidebar-section-title{font-size:.72rem;font-weight:800;letter-spacing:.065em;text-transform:uppercase;color:#344054;margin:.85rem 0 .3rem}
        .sidebar-file-title{font-size:.77rem;font-weight:750;color:#344054;margin:.65rem 0 .2rem}
        [data-testid="stSidebar"] [data-testid="stFileUploader"]{margin:0 0 .55rem}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section{background:#fff!important;border:1px dashed #c7d0dc!important;border-radius:13px!important;padding:.72rem!important;min-height:78px;box-shadow:0 1px 2px rgba(16,24,40,.025)!important;transition:.15s ease}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover{background:#fcfdff!important;border-color:#94a3b8!important;box-shadow:0 4px 14px rgba(16,24,40,.05)!important}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section>div{gap:.35rem!important}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button{background:#111827!important;color:#fff!important;border:0!important;border-radius:8px!important;min-height:34px!important;padding:0 .8rem!important;font-size:.74rem!important;font-weight:750!important;box-shadow:0 1px 2px rgba(16,24,40,.12)!important}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover{background:#1f2937!important}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] small{color:#98a2b3!important;font-size:.66rem!important}
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]{border:1px solid var(--border)!important;border-radius:9px!important;background:#fff!important}
        [data-testid="stSidebar"] .streamlit-expanderHeader{background:#f8fafc!important;border:1px solid var(--border)!important;border-radius:11px!important;color:#344054!important;font-weight:700!important;font-size:.77rem!important}
        [data-testid="stSidebar"] .streamlit-expanderContent{background:#fff!important;border:1px solid var(--border)!important;border-top:0!important;border-radius:0 0 11px 11px!important}
        /* CONTROLS */
        [data-baseweb="select"]>div,[data-baseweb="input"]>div{border:1px solid var(--border2)!important;border-radius:9px!important;background:#fff!important;box-shadow:none!important}
        [data-baseweb="select"]>div:hover,[data-baseweb="input"]>div:hover{border-color:#aab4c2!important}
        [data-baseweb="select"]>div:focus-within,[data-baseweb="input"]>div:focus-within{border-color:#93c5fd!important;box-shadow:0 0 0 3px rgba(37,99,235,.08)!important}
        [data-testid="stWidgetLabel"] p{font-size:.76rem!important;font-weight:650!important;color:#475467!important}
        [data-testid="stVerticalBlockBorderWrapper"]{background:#fff!important;border:1px solid var(--border)!important;border-radius:15px!important;box-shadow:var(--shadow)!important}
        .filter-card-title{font-size:.72rem;font-weight:800;color:#344054;text-transform:uppercase;letter-spacing:.065em;margin:.05rem 0 .7rem}
        .checkbox-container{min-height:42px;display:flex;align-items:center;padding-top:1.55rem}.checkbox-container .stCheckbox{margin-bottom:0!important}
        /* KPI */
        .metric-card{position:relative;background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1rem 1.05rem;min-height:108px;margin-bottom:1rem;box-shadow:var(--shadow);overflow:hidden}
        .metric-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#cbd5e1}
        .metric-card .label{font-size:.67rem;line-height:1.25;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.055em;margin-bottom:.5rem}
        .metric-card .value-container{display:flex;align-items:flex-end;justify-content:space-between;gap:.45rem}
        .metric-card .value{font-size:1.68rem;line-height:1;font-weight:800;color:var(--text);letter-spacing:-.035em;white-space:nowrap}
        .metric-card .badge{display:inline-flex;align-items:center;justify-content:center;padding:.25rem .48rem;border-radius:999px;font-size:.59rem;line-height:1.15;font-weight:800;white-space:nowrap}
        .badge-green{background:#ecfdf5;color:var(--green)} .badge-blue{background:#eff6ff;color:var(--blue)} .badge-amber{background:#fffbeb;color:var(--amber)} .badge-red{background:#fef2f2;color:var(--red)} .badge-purple{background:#f5f3ff;color:var(--purple)}
        .metric-positive:before{background:var(--green)} .metric-negative:before{background:var(--red)} .metric-blue:before{background:var(--blue)} .metric-amber:before{background:var(--amber)} .metric-purple:before{background:var(--purple)}
        /* BUTTONS */
        .stButton>button{border-radius:9px!important;min-height:38px!important;font-weight:750!important;font-size:.77rem!important;border:1px solid var(--border2)!important;background:#fff!important;color:#344054!important;box-shadow:0 1px 2px rgba(16,24,40,.04)!important}
        .stButton>button:hover{background:#f8fafc!important;border-color:#98a2b3!important;color:#111827!important}
        .stButton>button[kind="primary"]{background:#111827!important;color:#fff!important;border-color:#111827!important}.stButton>button[kind="primary"]:hover{background:#1f2937!important;border-color:#1f2937!important}
        [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden;box-shadow:0 1px 2px rgba(16,24,40,.025)}
        hr{border-color:var(--border)!important}
        </style>
        """,
        unsafe_allow_html=True,
    )


PASTEL_COLORS = {
    "green_main": "#10b981",
    "blue_main": "#3b82f6",
    "amber_main": "#f59e0b",
    "red_main": "#ef4444",
    "purple_main": "#7c3aed",
}


def render_metric_cards(cards: list[dict]):
    """Renderiza uma linha de cartões de KPI, substituindo os blocos de HTML
    repetidos da versão anterior.

    Cada item de `cards` aceita:
      label (str), value (str), badge_text (str),
      badge_class (str, ex.: "badge-green"), variant (str, ex.: "metric-positive")
    """
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            variant = card.get("variant", "")
            badge_class = card.get("badge_class", "badge-blue")
            badge_text = card.get("badge_text", "")
            st.markdown(
                f"""
                <div class="metric-card {variant}">
                    <div class="label">{card['label']}</div>
                    <div class="value-container">
                        <div class="value">{card['value']}</div>
                        <span class="badge {badge_class}">{badge_text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 5. SIDEBAR & ESTADO DA APLICAÇÃO
# ==============================================================================

inject_custom_css()

with st.sidebar:
    st.title("🌎 Navegação")
    st.caption("Sistema de análise de diversificação das exportações brasileiras")

    PAGE = st.radio(
        "Módulo",
        [
            "📊 RCA/RSCA Bilateral (UN Comtrade)",
            "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)",
            "🗺️ Potencial de Diversificação por Estado (UF)",
        ],
    )

    st.divider()
    st.markdown('<div class="sidebar-section-title">Carga de dados</div>', unsafe_allow_html=True)
    st.caption("Carregue as bases necessárias para habilitar cada módulo.")

    st.markdown('<div class="sidebar-file-title">01 · UN Comtrade</div>', unsafe_allow_html=True)
    file_comtrade = st.file_uploader(
        "Arquivo UN Comtrade",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        label_visibility="collapsed",
        help="CSV, XLSX, XLS, Parquet ou JSON. Para habilitar a RCA Global, inclua linhas com partner = 'World'.",
    )

    st.markdown('<div class="sidebar-file-title">02 · ComexStat Brasil — Nacional</div>', unsafe_allow_html=True)
    file_comexstat = st.file_uploader(
        "Arquivo ComexStat Nacional",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Base nacional de exportações brasileiras por SH6.",
    )

    st.markdown('<div class="sidebar-file-title">03 · ComexStat Brasil — UF</div>', unsafe_allow_html=True)
    file_comexstat_uf = st.file_uploader(
        "Arquivo ComexStat por UF",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Base de exportações por estado e SH6.",
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

    with st.sidebar.expander("⚙️ Configurações UN Comtrade", expanded=False):
        st.caption("Mapeamento dos campos e filtros opcionais da base carregada.")

        value_candidates = [
            c for c in cols
            if any(v in normalize_text(c) for v in ["value", "val", "fob", "cif", "primaryvalue"])
        ]
        default_val_idx = cols.index(value_candidates[0]) if value_candidates else 0
        c_val = st.selectbox("Campo de Valor", cols, index=default_val_idx)

        if col_prt in df_ct.columns:
            partners_list = sorted(df_ct[col_prt].dropna().astype(str).unique().tolist())
            sel_partners = st.multiselect(
                "Parceiros",
                partners_list,
                default=[],
                help="Deixe vazio para considerar todos os parceiros. 'World' é sempre mantido automaticamente.",
            )
        else:
            sel_partners = []

        if col_rep in df_ct.columns:
            reporters_list = sorted(df_ct[col_rep].dropna().astype(str).unique().tolist())
            sel_reporters = st.multiselect(
                "Reporters",
                reporters_list,
                default=[],
                help="Deixe vazio para considerar todos os reporters. 'World' é sempre mantido automaticamente.",
            )
        else:
            sel_reporters = []

        missing_cols = [c for c in [col_yr, col_rep, col_prt, col_sh6] if c not in df_ct.columns]

        if missing_cols:
            st.error(f"Colunas ausentes: {', '.join(missing_cols)}")
        else:
            if "comtrade_tidy" not in st.session_state:
                try:
                    st.session_state["comtrade_tidy"] = standardize_comtrade(
                        df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
                    )
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

            if st.button("Aplicar / Re-processar", type="primary", use_container_width=True):
                try:
                    tidy_ct = standardize_comtrade(
                        df_ct, col_yr, col_rep, col_prt, col_sh6, col_desc, c_val, sel_partners, sel_reporters
                    )
                    st.session_state["comtrade_tidy"] = tidy_ct
                    st.success("Comtrade processado!")
                except Exception as exc:
                    st.error(f"Falha ao processar: {exc}")

# ==============================================================================
# 6. MÓDULOS DA APLICAÇÃO
# ==============================================================================

# --- PÁGINA 1: UN COMTRADE RCA / RSCA BILATERAL ---
def page_comtrade_global():
    st.title("📊 Análise de RCA e RSCA Bilateral por Parceiro")
    st.caption("Inteligência comercial executiva · Inserção em mercados específicos · Balassa & Laursen")
    render_methodology_expander()

    if "comtrade_tidy" not in st.session_state:
        st.info("👈 Por favor, carregue e processe o arquivo do UN Comtrade na barra lateral.")
        return

    tidy = st.session_state["comtrade_tidy"]
    anos = sorted(int(a) for a in tidy["ano"].dropna().unique().tolist())
    if not anos:
        st.error("A base processada não contém anos válidos.")
        return

    with st.container(border=True):
        st.markdown('<div class="filter-card-title">🔍 Filtros Principais</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            start_year = st.selectbox("Ano de Início", anos, index=0)
        with col_b:
            end_year = st.selectbox("Ano Final", anos, index=len(anos) - 1)

        years_range = [y for y in anos if start_year <= y <= end_year] or [start_year]
        df_metrics = compute_bilateral_metrics(tidy, tuple(years_range))

        if df_metrics.empty:
            st.warning("Não há dados suficientes para o intervalo de anos selecionado.")
            return

        f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 1])
        with f_col1:
            reps = st.multiselect("Reporter (País):", sorted(df_metrics["reporter"].unique()))
        with f_col2:
            prts = st.multiselect("Partner (Parceiro):", sorted(df_metrics["partner"].unique()))
        with f_col3:
            sh6s = st.multiselect("SH6:", sorted(df_metrics["sh6"].unique()))
        with f_col4:
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
            only_advantage = st.checkbox("Apenas RCA por Partner ≥ 1")
            st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df_metrics.copy()
    if reps:
        filtered_df = filtered_df[filtered_df["reporter"].isin(reps)]
    if prts:
        filtered_df = filtered_df[filtered_df["partner"].isin(prts)]
    if sh6s:
        filtered_df = filtered_df[filtered_df["sh6"].isin(sh6s)]
    if only_advantage:
        filtered_df = filtered_df[filtered_df["rca_partner"] >= 1.0]

    render_metric_cards([
        {
            "label": "Registros Analisados",
            "value": f"{len(filtered_df):,}",
            "badge_text": "Tidy Base",
            "badge_class": "badge-blue",
        },
        {
            "label": "Média do RCA Bilateral",
            "value": format_num(filtered_df["rca_partner"].mean(), 2),
            "badge_text": "Balassa",
            "badge_class": "badge-green",
        },
        {
            "label": "Média do RSCA Bilateral",
            "value": format_num(filtered_df["rsca_partner"].mean(), 2),
            "badge_text": "Laursen",
            "badge_class": "badge-amber",
        },
    ])

    st.subheader("Resultados Detalhados com Rótulo Dinâmico por Parceiro")

    display_df = filtered_df.copy()
    partner_title = prts[0] if len(prts) == 1 else "Parceiro Selecionado"

    rca_col_name = f"RCA ({partner_title})"
    rsca_col_name = f"RSCA ({partner_title})"

    display_df = display_df.rename(columns={
        "rca_partner": rca_col_name,
        "rsca_partner": rsca_col_name,
    })

    st.dataframe(
        display_df,
        column_config={
            "valor": st.column_config.NumberColumn("Valor (US$)", format="$ %,.2f"),
            "mundo_valor": st.column_config.NumberColumn("Importado pelo Parceiro (US$)", format="$ %,.2f"),
            rca_col_name: st.column_config.NumberColumn(rca_col_name, format="%.4f"),
            rsca_col_name: st.column_config.NumberColumn(rsca_col_name, format="%.4f"),
            "ano": st.column_config.NumberColumn("Ano", format="%d"),
            "partner": "Parceiro Comercial",
            "reporter": "País Declarante",
        },
        height=380,
    )

    st.divider()

    st.subheader("📈 Análise Comparativa Executiva")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### **Índice RCA Bilateral (Balassa)**")
        fig_rca = px.histogram(
            filtered_df, x="rca_partner", color="partner",
            hover_data=["sh6_desc", "reporter"],
            title="Distribuição do RCA por Partner",
            labels={"rca_partner": "Índice RCA", "partner": "Parceiro"},
            template="plotly_white",
            nbins=30,
        )
        fig_rca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_rca.add_vline(x=1.0, line_dash="dash", line_color=PASTEL_COLORS["blue_main"])
        st.plotly_chart(fig_rca, use_container_width=True)

    with chart_col2:
        st.markdown("#### **Índice RSCA Bilateral (Laursen Simétrico)**")
        fig_rsca = px.box(
            filtered_df, x="partner", y="rsca_partner", color="partner",
            hover_data=["sh6_desc", "reporter"],
            title="Amplitude do RSCA Simétrico (-1 a +1)",
            labels={"rsca_partner": "Índice RSCA", "partner": "Parceiro"},
            template="plotly_white",
        )
        fig_rsca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig_rsca.add_hline(y=0, line_dash="dash", line_color=PASTEL_COLORS["red_main"])
        st.plotly_chart(fig_rsca, use_container_width=True)

    if len(years_range) > 1:
        st.divider()
        st.subheader("📉 Evolução Temporal de Métricas Bilaterais")

        c1, c2, c3 = st.columns(3)
        with c1:
            trend_reporter = st.selectbox("País:", sorted(df_metrics["reporter"].unique()), key="trend_rep")
        with c2:
            trend_partner = st.selectbox(
                "Parceiro:", sorted(df_metrics[df_metrics["reporter"] == trend_reporter]["partner"].unique()),
                key="trend_prt",
            )
        opts_sh6 = sorted(df_metrics[(df_metrics["reporter"] == trend_reporter) & (df_metrics["partner"] == trend_partner)]["sh6"].unique())
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
                    trend_df, x="ano", y="rca_partner", markers=True,
                    title=f"Evolução RCA — {trend_reporter} x {trend_partner}",
                    labels={"rca_partner": f"RCA ({trend_partner})", "ano": "Ano"},
                    template="plotly_white",
                )
                fig_trend_rca.update_traces(line_color=PASTEL_COLORS["blue_main"])
                fig_trend_rca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_trend_rca.add_hline(y=1.0, line_dash="dot", line_color=PASTEL_COLORS["amber_main"])
                st.plotly_chart(fig_trend_rca, use_container_width=True)

            with t_col2:
                fig_trend_rsca = px.line(
                    trend_df, x="ano", y="rsca_partner", markers=True,
                    title=f"Evolução RSCA — {trend_reporter} x {trend_partner}",
                    labels={"rsca_partner": f"RSCA ({trend_partner})", "ano": "Ano"},
                    template="plotly_white",
                )
                fig_trend_rsca.update_traces(line_color=PASTEL_COLORS["green_main"])
                fig_trend_rsca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_trend_rsca.add_hline(y=0.0, line_dash="dot", line_color=PASTEL_COLORS["red_main"])
                st.plotly_chart(fig_trend_rsca, use_container_width=True)


# --- PÁGINA 2: CRUZAMENTO BRASIL COMEXSTAT X COMTRADE ---
def page_comexstat_cross():
    st.title("🇧🇷 Cruzamento Pauta Brasil x Competitividade Global")
    st.caption("Inteligência comercial executiva · Pauta brasileira versus vantagem comparativa global")
    render_methodology_expander()

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

    bilateral_fallback = compute_bilateral_metrics(ct, (selected_year,))
    country_metrics, methodology = get_country_metrics(ct, selected_year, bilateral_fallback=bilateral_fallback)
    render_methodology_badge(methodology)

    if country_metrics.empty:
        return

    merged = pd.merge(
        cs[cs["ano"] == selected_year],
        country_metrics.rename(columns={"sh6": "sh6_cod"}),
        on="sh6_cod", how="left",
    )
    merged["rca"] = merged["rca"].fillna(0)
    merged["rsca"] = merged["rsca"].fillna(-1)

    with st.container(border=True):
        st.markdown('<div class="filter-card-title">🔍 Filtros de Segmentação</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            f_sh6 = st.multiselect("SH6", sorted(merged["sh6_cod"].dropna().unique()))
            f_cuci = st.multiselect("CUCI Grupo", sorted(merged["cuci_desc"].dropna().unique()) if "cuci_desc" in merged.columns else [])
        with col2:
            f_isic_div = st.multiselect("ISIC Divisão", sorted(merged["isic_div_desc"].dropna().unique()) if "isic_div_desc" in merged.columns else [])
            f_isic_sec = st.multiselect("ISIC Seção", sorted(merged["isic_sec_desc"].dropna().unique()) if "isic_sec_desc" in merged.columns else [])
        with col3:
            f_cgce1 = st.multiselect("CGCE Nível 1", sorted(merged["cgce1_desc"].dropna().unique()) if "cgce1_desc" in merged.columns else [])
            f_cgce2 = st.multiselect("CGCE Nível 2", sorted(merged["cgce2_desc"].dropna().unique()) if "cgce2_desc" in merged.columns else [])

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

    # O RCA tem ponto de corte em 1. O RSCA é a versão simétrica do RCA e
    # está matematicamente limitado ao intervalo [-1, +1]; portanto,
    # vantagem simultânea é representada por RCA > 1 e RSCA > 0.
    sh6_rca_rsca_alto = df_f[(df_f["rca"] > 1.0) & (df_f["rsca"] > 0.0)]["sh6_cod"].nunique()
    sh6_rca_rsca_baixo = df_f[(df_f["rca"] < 1.0) & (df_f["rsca"] < 0.0)]["sh6_cod"].nunique()

    metodologia_badge = "Global (Balassa)" if methodology == "global" else "Bilateral (média)"

    render_metric_cards([
        {"label": "Valor Exportado (FOB)", "value": format_usd(val_tot), "badge_text": "ComexStat", "badge_class": "badge-blue"},
        {"label": "Produtos Monitorados", "value": f"{df_f['sh6_cod'].nunique():,}", "badge_text": "SH6", "badge_class": "badge-amber"},
        {"label": "Produtos Competitivos", "value": f"{produtos_vantagem:,}", "badge_text": "RCA ≥ 1.0", "badge_class": "badge-green"},
        {
            "label": "SH6 com Vantagem Comparativa", "value": f"{sh6_rca_rsca_alto:,}",
            "badge_text": "RCA > 1 · RSCA > 0", "badge_class": "badge-green", "variant": "metric-positive",
        },
        {
            "label": "SH6 com Desvantagem Comparativa", "value": f"{sh6_rca_rsca_baixo:,}",
            "badge_text": "RCA < 1 · RSCA < 0", "badge_class": "badge-red", "variant": "metric-negative",
        },
    ])

    st.caption(
        f"Metodologia da RCA/RSCA nesta página: **{metodologia_badge}**. "
        "O RSCA varia entre −1 e +1; por isso, o corte equivalente ao RCA > 1 é RSCA > 0 (e RCA < 1 corresponde a RSCA < 0)."
    )

    st.subheader("📊 Distribuição de Exportação por Setor")
    group_opt = st.selectbox("Agrupar Visualização por:", ["CUCI Grupo", "ISIC Divisão", "ISIC Seção", "CGCE Nível 1", "CGCE Nível 2"])

    col_map = {
        "CUCI Grupo": "cuci_desc",
        "ISIC Divisão": "isic_div_desc",
        "ISIC Seção": "isic_sec_desc",
        "CGCE Nível 1": "cgce1_desc",
        "CGCE Nível 2": "cgce2_desc",
    }
    selected_col = col_map[group_opt]

    if selected_col in df_f.columns:
        agg_sector = df_f.groupby(selected_col).agg(
            Valor_FOB=("valor_fob", "sum"),
            RCA_Medio=("rca", "mean"),
            RSCA_Medio=("rsca", "mean"),
            N_Produtos=("sh6_cod", "nunique"),
        ).reset_index().sort_values("Valor_FOB", ascending=False)

        col_sec1, col_sec2 = st.columns(2)

        with col_sec1:
            fig_sec_rca = px.bar(
                agg_sector.head(15), x="Valor_FOB", y=selected_col, orientation="h",
                color="RCA_Medio", color_continuous_scale=["#3b82f6", "#10b981"],
                title=f"Top 15 Setores por Valor e RCA Médio ({group_opt})",
                labels={"Valor_FOB": "Valor FOB (US$)", selected_col: "Setor", "RCA_Medio": "RCA Médio"},
                template="plotly_white",
            )
            fig_sec_rca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec_rca, use_container_width=True)

        with col_sec2:
            fig_sec_rsca = px.bar(
                agg_sector.head(15), x="Valor_FOB", y=selected_col, orientation="h",
                color="RSCA_Medio", color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                title=f"Top 15 Setores por Valor e RSCA Médio ({group_opt})",
                labels={"Valor_FOB": "Valor FOB (US$)", selected_col: "Setor", "RSCA_Medio": "RSCA Médio"},
                template="plotly_white",
            )
            fig_sec_rsca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec_rsca, use_container_width=True)

        st.dataframe(
            agg_sector,
            column_config={
                "Valor_FOB": st.column_config.NumberColumn("Valor FOB (US$)", format="$ %,.2f"),
                "RCA_Medio": st.column_config.NumberColumn("RCA Médio", format="%.2f"),
                "RSCA_Medio": st.column_config.NumberColumn("RSCA Médio", format="%.2f"),
                "N_Produtos": st.column_config.NumberColumn("Produtos SH6", format="%d"),
            },
        )
    else:
        st.info(f"A coluna '{group_opt}' não foi encontrada na base carregada.")


# --- PÁGINA 3: POTENCIAL DE DIVERSIFICAÇÃO POR ESTADO (UF) ---
def page_state_diversification():
    st.title("🗺️ Potencial de Diversificação por Estado (UF)")
    st.caption("Inteligência subnacional · Produtos estratégicos subaproveitados e oportunidades de diversificação")
    render_methodology_expander()

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

    bilateral_fallback = compute_bilateral_metrics(ct, (selected_year,))
    country_metrics, methodology = get_country_metrics(ct, selected_year, bilateral_fallback=bilateral_fallback)
    render_methodology_badge(methodology)

    if country_metrics.empty:
        return

    advantage_df = country_metrics[country_metrics["rca"] >= 1.0]
    if advantage_df.empty:
        st.warning("Nenhum produto com RCA ≥ 1 foi encontrado para o Brasil no ano selecionado.")
        return

    df_potencial = compute_state_diversification_potentials(cs_uf, advantage_df, selected_year)

    if df_potencial.empty:
        st.error(
            "Não foi possível calcular o potencial com os dados fornecidos. Verifique se os códigos SH6 "
            "das duas bases (ComexStat UF e Comtrade) são compatíveis."
        )
        return

    n_alto_potencial = df_potencial[(df_potencial["potencial_score"] > 0.5) & (~df_potencial["ja_exportado"])]["sh6_cod"].nunique()
    n_vantagem_nacional = df_potencial["sh6_cod"].nunique()
    val_mercado_oportunidade = (
        df_potencial[(df_potencial["potencial_score"] > 0.5) & (~df_potencial["ja_exportado"])]
        .drop_duplicates("sh6_cod")["mundo_valor"].sum()
    )

    render_metric_cards([
        {
            "label": "Oportunidades Locais de Alta Prioridade", "value": f"{n_alto_potencial:,}",
            "badge_text": f"Demanda {format_usd(val_mercado_oportunidade)}", "badge_class": "badge-green",
        },
        {
            "label": "Produtos no Portfólio Nacional Competitivo", "value": f"{n_vantagem_nacional:,}",
            "badge_text": "RCA ≥ 1.0", "badge_class": "badge-blue",
        },
    ])
    st.caption(
        "Score de potencial = RSCA nacional (limitado a [0,1]) × (1 − participação do produto na pauta do estado). "
        "Prioridade alta = score acima de 0,5."
    )

    st.subheader("🏆 Ranking Subnacional por Score de Potencial")

    rank_uf = df_potencial.groupby("uf").agg(
        Score_Potencial_Total=("potencial_score", "sum"),
        Produtos_Nao_Explorados=("ja_exportado", lambda s: int((~s).sum())),
        Exportacao_Atual_FOB=("uf_total", "first"),
    ).reset_index().sort_values("Score_Potencial_Total", ascending=False)

    fig_uf_rank = px.bar(
        rank_uf, x="Score_Potencial_Total", y="uf", orientation="h",
        color="Exportacao_Atual_FOB", color_continuous_scale=["#eff6ff", "#3b82f6", "#1d4ed8"],
        title="Estados com Maior Potencial de Diversificação",
        labels={"Score_Potencial_Total": "Score de Potencial", "uf": "Estado", "Exportacao_Atual_FOB": "Exportação Atual (US$)"},
        template="plotly_white",
    )
    fig_uf_rank.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_uf_rank, use_container_width=True)

    st.dataframe(
        rank_uf,
        column_config={
            "Score_Potencial_Total": st.column_config.NumberColumn("Score de Potencial Total", format="%.2f"),
            "Exportacao_Atual_FOB": st.column_config.NumberColumn("Exportação Atual (US$)", format="$ %,.2f"),
            "Produtos_Nao_Explorados": st.column_config.NumberColumn("Produtos Não Explorados", format="%d"),
        },
    )

    st.divider()

    st.subheader("🔍 Detalhamento por Estado (UF)")

    with st.container(border=True):
        st.markdown('<div class="filter-card-title">🔍 Segmentação Subnacional</div>', unsafe_allow_html=True)

        col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
        with col_sel1:
            uf_target = st.selectbox("Selecione o Estado (UF)", sorted(df_potencial["uf"].unique()))

        df_uf_filtered = df_potencial[df_potencial["uf"] == uf_target]

        with col_sel2:
            cuci_options = sorted(df_uf_filtered["cuci_desc"].dropna().unique()) if "cuci_desc" in df_uf_filtered.columns else []
            cuci_target = st.multiselect("CUCI Grupo", cuci_options)
        with col_sel3:
            sh6_options = sorted(df_uf_filtered["sh6_cod"].dropna().unique())
            sh6_target = st.multiselect("SH6", sh6_options)
        with col_sel4:
            st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
            only_new = st.checkbox("Somente NÃO exportados pelo estado", value=False)
            st.markdown('</div>', unsafe_allow_html=True)

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
        label_col = "sh6_desc" if "sh6_desc" in top_n.columns and top_n["sh6_desc"].notna().any() else "sh6_cod"
        color_col = "cuci_desc" if "cuci_desc" in top_n.columns and top_n["cuci_desc"].notna().any() else None
        fig_top = px.bar(
            top_n, x="potencial_score", y=label_col, orientation="h",
            color=color_col,
            title=f"Top produtos por potencial de diversificação — {uf_target}",
            labels={"potencial_score": "Score de Potencial", label_col: "Produto (SH6)", "cuci_desc": "CUCI Grupo"},
            template="plotly_white",
        )
        fig_top.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_top, use_container_width=True)

    disp_cols = ["sh6_cod", "sh6_desc", "cuci_desc", "rca", "rsca", "valor_fob", "share_local", "potencial_score", "ja_exportado"]
    available_disp_cols = [c for c in disp_cols if c in df_uf_seg.columns]

    df_display = df_uf_seg[available_disp_cols].sort_values("potencial_score", ascending=False).head(50).copy()
    if "share_local" in df_display.columns:
        # Corrige o bug de formatação da v6 (format="%.2%%" era inválido):
        # convertemos a fração para pontos percentuais antes de exibir.
        df_display["share_local"] = df_display["share_local"] * 100

    st.dataframe(
        df_display,
        column_config={
            "sh6_cod": "SH6",
            "sh6_desc": "Descrição do Produto",
            "cuci_desc": "CUCI Grupo",
            "rca": st.column_config.NumberColumn("RCA Nacional", format="%.2f"),
            "rsca": st.column_config.NumberColumn("RSCA Nacional", format="%.2f"),
            "valor_fob": st.column_config.NumberColumn("Valor FOB Exportado pelo Estado (US$)", format="$ %,.2f"),
            "share_local": st.column_config.NumberColumn("Participação na Pauta Estadual", format="%.1f%%"),
            "potencial_score": st.column_config.NumberColumn("Score Potencial", format="%.2f"),
            "ja_exportado": "Já Exportado pelo Estado?",
        },
        height=420,
    )


# ==============================================================================
# 7. ROTEADOR DE PÁGINAS
# ==============================================================================

if PAGE == "📊 RCA/RSCA Bilateral (UN Comtrade)":
    page_comtrade_global()
elif PAGE == "🇧🇷 Cruzamento Nacional (ComexStat x Comtrade)":
    page_comexstat_cross()
elif PAGE == "🗺️ Potencial de Diversificação por Estado (UF)":
    page_state_diversification()
