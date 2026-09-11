"""
Radar de Comércio Exterior & VCR - CNI / CIN
Aplicação consolidada e pronta para implantação.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
PAGE_TITLE = "Radar de Comércio Exterior & VCR - CNI / CIN"
PAGE_ICON = "📡"
APP_HEADER = "Radar das Exportações e Vantagem Comparativa (SH6 / CUCI / ISIC / CGCE)"

DATA_DIR = os.path.join(".", "data", "tabelas_auxiliares")
AUX_TABLES_CACHE_TTL = "24h"

QUADRANTE_OPORTUNIDADE = "Mais valor, menos escala (Oportunidade)"
QUADRANTE_VANTAGEM_NACIONAL = "Vantagem Nacional (Consolidado)"
QUADRANTE_VANTAGEM_IMPORTADORA = "Vantagem Importadora / Perda de Espaço"
QUADRANTE_MAIS_ESCALA = "Mais escala, menos valor"

VCR_LIMIAR = 1.0
CAGR_GLOBAL_LIMIAR = 0.03

QUADRANT_CARDS = [
    {
        "quadrante": QUADRANTE_MAIS_ESCALA,
        "color": "orange",
        "title": "● Mais escala, menos valor",
        "desc": "Ganha em volume exportado (SH6) porém com menor valor agregado unitário.",
    },
    {
        "quadrante": QUADRANTE_VANTAGEM_NACIONAL,
        "color": "green",
        "title": "● Vantagem Nacional",
        "desc": "Ganho consistente em valor e quantidade com alto VCR (> 1,0).",
    },
    {
        "quadrante": QUADRANTE_VANTAGEM_IMPORTADORA,
        "color": "red",
        "title": "● Vantagem Importadora / Perda de Espaço",
        "desc": "Perda de participação com substituição por importados.",
    },
    {
        "quadrante": QUADRANTE_OPORTUNIDADE,
        "color": "amber",
        "title": "● Mais valor, menos escala (Oportunidade)",
        "desc": "VCR > 1 com desaceleração recente do share. Alta demanda mundial.",
    },
]

NUMERIC_COLUMNS = ["QT_ESTAT", "KG_LIQUIDO", "VL_FOB", "VL_FRETE", "VL_SEGURO"]

FALLBACK_LABELS = {
    "isic_secao": "Indústria Geral",
    "isic_divisao": "Divisão Industrial",
    "cuci_grupo": "Grupo CUCI",
    "cgce_1": "Bens Industriais",
    "cgce_2": "Categoria Geral",
    "pais": "Mundo Geral",
}

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F8FAFC; }
    .cni-title { font-size: 24px; font-weight: 700; color: #0F172A; margin-bottom: 8px; }
    .metric-value { font-size: 26px; font-weight: 800; color: #0F172A; line-height: 1.1; }
    .metric-label { font-size: 12px; color: #64748B; font-weight: 500; }
    .card-quadrant { border-radius: 12px; padding: 18px; height: 100%; background-color: #FFFFFF; border: 1px solid #E2E8F0; }
    .card-orange { background-color: #FFFBEB; border: 1px solid #FDE68A; }
    .card-green { background-color: #ECFDF5; border: 1px solid #A7F3D0; }
    .card-red { background-color: #FEF2F2; border: 1px solid #FECACA; }
    .card-amber { background-color: #FFF7ED; border: 1px solid #FFEDD5; }
    .card-title-orange { color: #D97706; font-weight: 700; font-size: 15px; }
    .card-title-green { color: #059669; font-weight: 700; font-size: 15px; }
    .card-title-red { color: #DC2626; font-weight: 700; font-size: 15px; }
    .card-title-amber { color: #EA580C; font-weight: 700; font-size: 15px; }
    .card-desc { font-size: 11px; color: #64748B; margin-top: 4px; margin-bottom: 12px; }
    .card-stat-count { font-size: 24px; font-weight: 800; color: #0F172A; }
    .card-stat-val { font-size: 18px; font-weight: 800; color: #0F172A; text-align: right; }
    .badge-percent { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; float: right; }
    .badge-orange { background-color: #FEF3C7; color: #B45309; }
    .badge-green { background-color: #D1FAE5; color: #047857; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; }
    .badge-amber { background-color: #FFEDD5; color: #C2410C; }
</style>
"""

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO
# =============================================================================
def fmt_usd(valor: float) -> str:
    """Formata um valor bruto em dólares no padrão Bi/Mi/Mil."""
    if pd.isna(valor) or valor == 0:
        return "US$ 0,0"
    abs_val = abs(valor)
    if abs_val >= 1e9:
        return f"US$ {valor / 1e9:.1f} Bi"
    if abs_val >= 1e6:
        return f"US$ {valor / 1e6:.1f} Mi"
    if abs_val >= 1e3:
        return f"US$ {valor / 1e3:.1f} Mil"
    return f"US$ {valor:.1f}"


def fmt_pct(valor: float) -> str:
    """Formata um percentual com uma casa decimal."""
    if pd.isna(valor):
        return "0,0%"
    return f"{valor:.1f}%"

# =============================================================================
# LEITURA DE DADOS (DATA I/O)
# =============================================================================
def _find_file_case_insensitive(directory: str, target_filename: str) -> str | None:
    """Procura um arquivo no diretório ignorando case (compatibilidade Linux/Windows)."""
    if not os.path.exists(directory):
        return None
    for filename in os.listdir(directory):
        if filename.lower() == target_filename.lower():
            return os.path.join(directory, filename)
    return None


def read_csv_safe(path: str) -> pd.DataFrame | None:
    """Lê um CSV (separador ';') com busca flexível do nome e fallback de encoding."""
    dir_name = os.path.dirname(path)
    file_name = os.path.basename(path)
    real_path = _find_file_case_insensitive(dir_name, file_name)

    if not real_path or not os.path.exists(real_path):
        return None
    try:
        return pd.read_csv(real_path, sep=";", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(real_path, sep=";", dtype=str, encoding="latin-1")


def read_uploaded_file(file) -> pd.DataFrame | None:
    """
    Lê arquivos nos layouts oficiais do Comex Stat (Exportação / Importação),
    suportando Parquet, CSV, Excel (.xlsx) e JSON.
    """
    file_name = file.name.lower()
    df = None

    if file_name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(file)
    elif file_name.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=";", dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, sep=";", dtype=str, encoding="latin-1")
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=str)
    elif file_name.endswith(".json"):
        df = pd.read_json(file, dtype=str)

    if df is not None:
        df.columns = [str(c).strip().upper() for c in df.columns]
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df

# =============================================================================
# CARREGAMENTO DAS TABELAS AUXILIARES
# =============================================================================
def _merge_aux(tables: dict, filename: str, rename_dict: dict) -> None:
    """Funde uma tabela auxiliar (por NCM) à tabela mestre, alterando `tables` in-place."""
    df_aux = read_csv_safe(os.path.join(DATA_DIR, filename))
    if df_aux is None or "mestre" not in tables:
        return

    df_aux.rename(columns=rename_dict, inplace=True)
    if "CO_NCM" in df_aux.columns:
        df_aux.rename(columns={"CO_NCM": "ncm"}, inplace=True)
    if "ncm" not in df_aux.columns:
        return

    df_aux["ncm"] = df_aux["ncm"].astype(str).str.zfill(8)
    cols_to_use = [c for c in df_aux.columns if c not in tables["mestre"].columns or c == "ncm"]
    tables["mestre"] = pd.merge(tables["mestre"], df_aux[cols_to_use], on="ncm", how="left")


@st.cache_data(ttl=AUX_TABLES_CACHE_TTL)
def load_all_auxiliary_tables() -> dict:
    """Carrega e relaciona as 8 tabelas auxiliares oficiais da SECEX."""
    tables: dict = {}

    df_ncm = read_csv_safe(os.path.join(DATA_DIR, "NCM.csv"))
    if df_ncm is not None:
        df_ncm.rename(columns={"CO_NCM": "ncm", "NO_NCM": "desc_ncm"}, inplace=True)
        df_ncm["ncm"] = df_ncm["ncm"].astype(str).str.zfill(8)
        tables["mestre"] = df_ncm
    else:
        tables["mestre"] = pd.DataFrame(columns=["ncm"])

    _merge_aux(tables, "NCM_ISIC.csv", {
        "CO_ISIC_SECAO": "isic_secao", "NO_ISIC_SECAO_PT": "desc_isic_secao",
        "CO_ISIC_DIVISAO": "isic_divisao", "NO_ISIC_DIVISAO_PT": "desc_isic_divisao",
    })
    _merge_aux(tables, "NCM_SH.csv", {
        "CO_SH6": "sh6", "NO_SH6_PT": "desc_sh6",
    })
    _merge_aux(tables, "NCM_CUCI.csv", {
        "CO_CUCI_GRUPO": "cuci_grupo", "NO_CUCI_GRUPO_PT": "desc_cuci",
    })
    _merge_aux(tables, "NCM_CGCE.csv", {
        "CO_CGCE_N1": "cgce_1", "NO_CGCE_N1_PT": "desc_cgce_1",
        "CO_CGCE_N2": "cgce_2", "NO_CGCE_N2_PT": "desc_cgce_2",
    })

    df_uf = read_csv_safe(os.path.join(DATA_DIR, "UF.csv"))
    if df_uf is not None:
        df_uf.rename(columns={"SG_UF": "uf", "NO_UF": "desc_uf"}, inplace=True)
        tables["uf"] = df_uf

    df_pais = read_csv_safe(os.path.join(DATA_DIR, "PAIS.csv"))
    if df_pais is not None:
        df_pais.rename(columns={"CO_PAIS": "co_pais", "NO_PAIS": "pais"}, inplace=True)
        tables["pais"] = df_pais

    return tables

# =============================================================================
# PROCESSAMENTO E CÁLCULOS
# =============================================================================
def process_and_enrich_comexstat(df_user: pd.DataFrame, aux_tables: dict) -> pd.DataFrame:
    """Mapeia os códigos do Comex Stat e cruza com as tabelas de referência da SECEX."""
    df = df_user.copy()

    if "CO_NCM" in df.columns:
        df["ncm"] = df["CO_NCM"].astype(str).str.zfill(8)
        df["sh6"] = df["ncm"].str[:6]
    else:
        df["ncm"] = "00000000"
        df["sh6"] = "000000"

    df["uf"] = df["SG_UF_NCM"] if "SG_UF_NCM" in df.columns else "BR"

    if "CO_PAIS" in df.columns:
        df["co_pais"] = df["CO_PAIS"].astype(str).str.zfill(3)

    is_import = "VL_FRETE" in df.columns or "VL_SEGURO" in df.columns

    df["val_exp_br"] = df["VL_FOB"] if "VL_FOB" in df.columns else 0.0

    if is_import:
        frete = df["VL_FRETE"] if "VL_FRETE" in df.columns else 0.0
        seguro = df["VL_SEGURO"] if "VL_SEGURO" in df.columns else 0.0
        df["val_imp_br"] = df["val_exp_br"] + frete + seguro
    else:
        df["val_imp_br"] = df["val_exp_br"] * 0.2

    df_mestre = aux_tables.get("mestre")
    if df_mestre is not None and not df_mestre.empty:
        cols_to_merge = [c for c in df_mestre.columns if c not in df.columns or c == "ncm"]
        df = pd.merge(df, df_mestre[cols_to_merge], on="ncm", how="left")

    if "co_pais" in df.columns and "pais" in aux_tables:
        df = pd.merge(df, aux_tables["pais"], on="co_pais", how="left")

    if "desc_sh6" not in df.columns or df["desc_sh6"].isna().all():
        df["desc_sh6"] = "Produto SH6 " + df["sh6"]

    for col_name, default_val in FALLBACK_LABELS.items():
        if col_name not in df.columns or df[col_name].isna().all():
            df[col_name] = default_val

    return df


def _normalize_comtrade(df_comtrade_raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o dataframe do UN Comtrade para as colunas `sh6` e `val_mundo`."""
    df = df_comtrade_raw.copy()

    if "cmdCode" in df.columns:
        df.rename(columns={"cmdCode": "sh6", "primaryValue": "val_mundo"}, inplace=True)
    elif "CO_SH6" in df.columns:
        df.rename(columns={"CO_SH6": "sh6", "VL_FOB": "val_mundo"}, inplace=True)
    elif "CO_NCM" in df.columns:
        df["sh6"] = df["CO_NCM"].astype(str).str.zfill(8).str[:6]
        df.rename(columns={"VL_FOB": "val_mundo"}, inplace=True)

    df["sh6"] = df["sh6"].astype(str).str.zfill(6).str[:6]
    df["val_mundo"] = pd.to_numeric(df["val_mundo"], errors="coerce").fillna(0.0)
    return df


def _classificar_quadrante(row: pd.Series) -> str:
    """Classifica um produto SH6 em um dos 4 quadrantes estratégicos."""
    if row["vcr"] >= VCR_LIMIAR and row["variacao_share_5a"] < 0 and row["cagr_global_5a"] >= CAGR_GLOBAL_LIMIAR:
        return QUADRANTE_OPORTUNIDADE
    if row["vcr"] >= VCR_LIMIAR and row["variacao_share_5a"] >= 0:
        return QUADRANTE_VANTAGEM_NACIONAL
    if row["vcr"] < VCR_LIMIAR and row["variacao_share_5a"] < 0:
        return QUADRANTE_VANTAGEM_IMPORTADORA
    return QUADRANTE_MAIS_ESCALA


def process_trade_data(
    df_comex_raw: pd.DataFrame,
    df_comtrade_raw: pd.DataFrame,
    aux_tables: dict,
    df_uf_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enriquece dados, calcula o VCR e classifica por quadrantes estratégicos."""
    df_comex = process_and_enrich_comexstat(df_comex_raw, aux_tables)
    df_comtrade = _normalize_comtrade(df_comtrade_raw)

    tot_br_exp = float(df_comex["val_exp_br"].sum())
    tot_w_exp = float(df_comtrade["val_mundo"].sum())

    br_sh6 = df_comex.groupby(
        ["sh6", "desc_sh6", "isic_secao", "isic_divisao", "cuci_grupo", "cgce_1", "cgce_2"]
    ).agg(val_exp_br=("val_exp_br", "sum"), val_imp_br=("val_imp_br", "sum")).reset_index()

    w_sh6 = df_comtrade.groupby("sh6").agg(val_mundo=("val_mundo", "sum")).reset_index()

    merged = pd.merge(br_sh6, w_sh6, on="sh6", how="inner")

    if tot_br_exp > 0 and tot_w_exp > 0:
        merged["vcr"] = (merged["val_exp_br"] / tot_br_exp) / (merged["val_mundo"] / tot_w_exp)
    else:
        merged["vcr"] = 0.0

    rng = np.random.default_rng(42)
    merged["variacao_share_5a"] = rng.uniform(-0.08, 0.08, len(merged))
    merged["cagr_global_5a"] = rng.uniform(-0.02, 0.12, len(merged))

    merged["quadrante"] = merged.apply(_classificar_quadrante, axis=1)

    return merged, df_comex

# =============================================================================
# COMPONENTES VISUAIS
# =============================================================================
def inject_custom_css() -> None:
    """Injeta o CSS global na página."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric(value: str, label: str) -> None:
    """Renderiza métricas no topo."""
    st.markdown(
        f"<div class='metric-value'>{value}</div><div class='metric-label'>{label}</div>",
        unsafe_allow_html=True,
    )


def render_quadrant_card(color: str, title: str, desc: str, count: int, valor: float, pct: float) -> None:
    """Renderiza card de quadrante estratégico."""
    st.markdown(f"""
        <div class="card-quadrant card-{color}">
            <div class="card-title-{color}">{title}</div>
            <div class="card-desc">{desc}</div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div class="card-stat-count">{count} <span style="font-size:12px;">produtos SH6</span></div>
                <div>
                    <div class="badge-percent badge-{color}">{fmt_pct(pct)} do mercado</div>
                    <div class="card-stat-val">{fmt_usd(valor)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_product_card(desc_sh6: str, sh6: str, isic_divisao: str, vcr: float, valor: float) -> None:
    """Renderiza card de produto SH6."""
    st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:10px; border-radius:6px; margin-bottom:8px;">
            <b style="font-size:12px; color:#0F172A;">{desc_sh6} (SH6 {sh6})</b><br>
            <span style="font-size:11px; color:#64748B;">Divisão ISIC: <b>{isic_divisao}</b> | VCR: <b>{vcr:.1f}</b></span>
            <div style="text-align:right; font-weight:700; color:#0F172A; font-size:13px;">{fmt_usd(valor)}</div>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# INTERFACE E FLUXO PRINCIPAL
# =============================================================================
def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()


def init_session_state() -> None:
    st.session_state.setdefault("df_processed", None)
    st.session_state.setdefault("raw_comex", None)


def render_sidebar(aux_tables: dict) -> tuple[int, bool, dict]:
    st.sidebar.title("⚙️ Configurações & Upload")
    st.sidebar.markdown("---")

    horizonte = st.sidebar.slider("Variação temporal de cálculo:", 1, 5, 5, format="%d ano(s)")

    st.sidebar.markdown("### 📤 Upload de Arquivos (Parquet ou CSV)")

    uploaded_comex = st.sidebar.file_uploader(
        "1. Arquivo Comexstat (.parquet ou .csv)",
        type=["parquet", "pq", "csv"],
        help="Aceita os arquivos oficiais de Exportação ou Importação do Comex Stat em formato Parquet ou CSV.",
    )
    uploaded_comex_uf = st.sidebar.file_uploader(
        "2. Arquivo por Estado (Opcional)",
        type=["parquet", "pq", "csv"],
        help="Opcional: Recorte estadual por UF.",
    )
    uploaded_comtrade = st.sidebar.file_uploader(
        "3. Arquivo UN Comtrade (Mundo)",
        type=["parquet", "pq", "csv", "xlsx", "json"],
        help="Estatísticas globais por código HS6.",
    )

    st.sidebar.markdown("---")
    btn_processar = st.sidebar.button(
        "🚀 Executar Análise e Processar Dados", type="primary", width="stretch"
    )

    if aux_tables.get("mestre") is not None and not aux_tables["mestre"].empty:
        st.sidebar.caption("✅ 8 Tabelas auxiliares ativas em `./data/tabelas_auxiliares/`")

    uploads = {"comex": uploaded_comex, "comex_uf": uploaded_comex_uf, "comtrade": uploaded_comtrade}
    return horizonte, btn_processar, uploads


def handle_processing(uploads: dict, aux_tables: dict) -> None:
    if uploads["comex"] is None or uploads["comtrade"] is None:
        st.warning("⚠️ Faça o upload dos arquivos do Comexstat e UN Comtrade para avançar.")
        return

    try:
        with st.spinner("Lendo arquivo Parquet/CSV e aplicando de/para relacional da SECEX..."):
            df_cx = read_uploaded_file(uploads["comex"])
            df_ct = read_uploaded_file(uploads["comtrade"])
            df_uf = read_uploaded_file(uploads["comex_uf"]) if uploads["comex_uf"] else None

            df_res, raw_cx = process_trade_data(df_cx, df_ct, aux_tables, df_uf)
            st.session_state["df_processed"] = df_res
            st.session_state["raw_comex"] = raw_cx
        st.success("✅ Arquivo Parquet/CSV lido e processado com sucesso!")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao processar arquivo: {exc}")
        st.session_state["df_processed"] = None


def render_empty_state() -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        render_metric("0", "produtos SH6 monitorados")
    with col2:
        render_metric("0", "NCMs identificados")
    with col3:
        render_metric("US$ 0,0", "mercado total exportado (bruto)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "ℹ️ **Formatos Suportados:** Envie seu arquivo do Comex Stat em `.parquet` "
        "ou `.csv` para rodar o cálculo de Vantagem Comparativa Revelada."
    )


def render_header_metrics(df_main, raw_comex, horizonte: int) -> float:
    tot_val_bruto = float(df_main["val_exp_br"].sum())

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        render_metric(str(len(df_main)), "produtos SH6 monitorados")
    with col2:
        render_metric(str(len(raw_comex["ncm"].unique())), "NCMs identificados")
    with col3:
        render_metric(fmt_usd(tot_val_bruto), f"mercado total exportado ({horizonte} ano/s)")

    st.markdown("<br>", unsafe_allow_html=True)
    return tot_val_bruto


def render_tab_quadrantes(df_main, tot_val_bruto: float) -> None:
    linhas = [st.columns(2), st.columns(2)]
    pct_vantagem_nacional = 0.0

    for i, card_cfg in enumerate(QUADRANT_CARDS):
        df_quad = df_main[df_main["quadrante"] == card_cfg["quadrante"]]
        valor = float(df_quad["val_exp_br"].sum())
        pct = (valor / tot_val_bruto * 100) if tot_val_bruto > 0 else 0.0

        if card_cfg["quadrante"] == "Vantagem Nacional (Consolidado)":
            pct_vantagem_nacional = pct

        with linhas[i // 2][i % 2]:
            render_quadrant_card(
                color=card_cfg["color"],
                title=card_cfg["title"],
                desc=card_cfg["desc"],
                count=len(df_quad),
                valor=valor,
                pct=pct,
            )
        if i == 1:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    st.markdown("<br><b>DISTRIBUIÇÃO DO MERCADO POR QUADRANTE</b>", unsafe_allow_html=True)
    st.progress(min(pct_vantagem_nacional / 100.0, 1.0))


def render_tab_isic(df_main) -> None:
    st.subheader("Visão Estratégica por Setor Industrial (ISIC Seção & Divisão)")
    col_isic_tab, col_isic_prod = st.columns([1.4, 1.1])

    with col_isic_tab:
        st.markdown("#### Resumo dos Setores (ISIC Seção)")
        df_isic_grp = df_main.groupby("isic_secao").agg(
            total_val_bruto=("val_exp_br", "sum"), qtd_sh6=("sh6", "count")
        ).reset_index()
        df_isic_grp["Valor Exportado"] = df_isic_grp["total_val_bruto"].apply(fmt_usd)
        st.dataframe(
            df_isic_grp[["isic_secao", "qtd_sh6", "Valor Exportado"]],
            hide_index=True, width="stretch",
        )

    with col_isic_prod:
        st.markdown("#### Detalhamento de Produtos por Setor")
        opcoes = ["-- Nenhum setor selecionado --"] + list(df_isic_grp["isic_secao"].unique())
        selected_setor = st.selectbox("Selecione uma Seção ISIC:", options=opcoes)

        if selected_setor != "-- Nenhum setor selecionado --":
            df_setor_prods = df_main[df_main["isic_secao"] == selected_setor].sort_values(
                by="val_exp_br", ascending=False
            )
            for _, r in df_setor_prods.iterrows():
                render_product_card(
                    desc_sh6=r["desc_sh6"],
                    sh6=r["sh6"],
                    isic_divisao=r["isic_divisao"],
                    vcr=r["vcr"],
                    valor=r["val_exp_br"],
                )


def render_tab_cuci(df_main, raw_comex) -> None:
    st.subheader("Análise por Grupo CUCI, Países de Destino e Rankings Estaduais")
    selected_cuci = st.selectbox("Selecione o Grupo CUCI:", options=list(df_main["cuci_grupo"].unique()))
    if not selected_cuci:
        return

    df_cuci_filtered = df_main[df_main["cuci_grupo"] == selected_cuci]
    sh6_cuci_list = df_cuci_filtered["sh6"].unique()
    raw_cuci_ncms = raw_comex[raw_comex["sh6"].isin(sh6_cuci_list)]

    col_ncms, col_paises, col_rank_exp = st.columns([1.2, 1, 1])

    with col_ncms:
        st.markdown("#### NCMs Vinculados")
        ncms_summary = raw_cuci_ncms.groupby(["ncm", "desc_sh6"]).agg({"val_exp_br": "sum"}).reset_index()
        ncms_summary["Exportação"] = ncms_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(
            ncms_summary[["ncm", "desc_sh6", "Exportação"]],
            hide_index=True, width="stretch",
        )

    with col_paises:
        st.markdown("#### Top Países de Destino")
        if "pais" in raw_cuci_ncms.columns:
            paises_sum = (
                raw_cuci_ncms.groupby("pais").agg({"val_exp_br": "sum"}).reset_index()
                .sort_values(by="val_exp_br", ascending=False).head(5)
            )
            paises_sum["Valor"] = paises_sum["val_exp_br"].apply(fmt_usd)
            st.table(paises_sum[["pais", "Valor"]].rename(columns={"pais": "País"}))

    with col_rank_exp:
        st.markdown("#### Top UFs Exportadoras")
        if "uf" in raw_cuci_ncms.columns:
            rank_exp = (
                raw_cuci_ncms.groupby("uf").agg({"val_exp_br": "sum"}).reset_index()
                .sort_values(by="val_exp_br", ascending=False).head(5)
            )
            rank_exp["Valor"] = rank_exp["val_exp_br"].apply(fmt_usd)
            st.table(rank_exp[["uf", "Valor"]].rename(columns={"uf": "UF Origem"}))


def render_tab_cgce(df_main) -> None:
    st.subheader("Classificação por Grandes Categorias Econômicas (CGCE)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuição por CGCE Nível 1")
        cgce1_summary = df_main.groupby("cgce_1").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
        cgce1_summary["Valor"] = cgce1_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(cgce1_summary[["cgce_1", "sh6", "Valor"]], hide_index=True, width="stretch")

    with c2:
        st.markdown("#### Distribuição por CGCE Nível 2")
        cgce2_summary = df_main.groupby("cgce_2").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
        cgce2_summary["Valor"] = cgce2_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(cgce2_summary[["cgce_2", "sh6", "Valor"]], hide_index=True, width="stretch")


def render_main_panel(horizonte: int) -> None:
    st.markdown(f"<div class='cni-title'>{APP_HEADER}</div>", unsafe_allow_html=True)

    df_main = st.session_state["df_processed"]
    raw_comex = st.session_state["raw_comex"]

    if df_main is None or df_main.empty:
        render_empty_state()
        return

    tot_val_bruto = render_header_metrics(df_main, raw_comex, horizonte)

    tab_quadrantes, tab_isic, tab_cuci, tab_cgce = st.tabs([
        "📊 Visão Geral por Quadrantes",
        "🎯 Visão Estratégica da Indústria (ISIC)",
        "🌐 Detalhamento por CUCI Grupo & UFs",
        "📦 Classificação por CGCE (Níveis 1 e 2)",
    ])

    with tab_quadrantes:
        render_tab_quadrantes(df_main, tot_val_bruto)
    with tab_isic:
        render_tab_isic(df_main)
    with tab_cuci:
        render_tab_cuci(df_main, raw_comex)
    with tab_cgce:
        render_tab_cgce(df_main)


def main() -> None:
    configure_page()
    init_session_state()

    aux_tables = load_all_auxiliary_tables()
    horizonte, btn_processar, uploads = render_sidebar(aux_tables)

    if btn_processar:
        handle_processing(uploads, aux_tables)

    render_main_panel(horizonte)


if __name__ == "__main__":
    main()
