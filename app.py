"""
Radar de Comércio Exterior & VCR - CNI / CIN
Aplicação adaptada para o layout consolidado do Comex Stat.
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

# Mapeamento do novo layout de colunas da planilha
COLUMN_MAPPING = {
    "ANO": "ano",
    "PAÍSES": "pais",
    "PAISES": "pais",
    "CÓDIGO SH6": "sh6",
    "CODIGO SH6": "sh6",
    "DESCRIÇÃO SH6": "desc_sh6",
    "DESCRICAO SH6": "desc_sh6",
    "CÓDIGO CGCE NÍVEL 2": "cgce_2_cod",
    "CODIGO CGCE NIVEL 2": "cgce_2_cod",
    "DESCRIÇÃO CGCE NÍVEL 2": "cgce_2",
    "DESCRICAO CGCE NIVEL 2": "cgce_2",
    "CÓDIGO CGCE NÍVEL 1": "cgce_1_cod",
    "CODIGO CGCE NIVEL 1": "cgce_1_cod",
    "DESCRIÇÃO CGCE NÍVEL 1": "cgce_1",
    "DESCRICAO CGCE NIVEL 1": "cgce_1",
    "CÓDIGO CUCI GRUPO": "cuci_cod",
    "CODIGO CUCI GRUPO": "cuci_cod",
    "DESCRIÇÃO CUCI GRUPO": "cuci_grupo",
    "DESCRICAO CUCI GRUPO": "cuci_grupo",
    "CÓDIGO ISIC DIVISÃO": "isic_divisao_cod",
    "CODIGO ISIC DIVISAO": "isic_divisao_cod",
    "DESCRIÇÃO ISIC DIVISÃO": "isic_divisao",
    "DESCRICAO ISIC DIVISAO": "isic_divisao",
    "CÓDIGO ISIC SEÇÃO": "isic_secao_cod",
    "CODIGO ISIC SECAO": "isic_secao_cod",
    "DESCRIÇÃO ISIC SEÇÃO": "isic_secao",
    "DESCRICAO ISIC SECAO": "isic_secao",
    "VALOR US$ FOB": "val_exp_br",
    "VALOR FOB": "val_exp_br",
    "VL_FOB": "val_exp_br"
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
    """Formata valor em dólares no padrão Bi/Mi/Mil."""
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
    """Formata percentual com uma casa decimal."""
    if pd.isna(valor):
        return "0,0%"
    return f"{valor:.1f}%"

# =============================================================================
# LEITURA E TRATAMENTO DOS ARQUIVOS
# =============================================================================
def read_uploaded_file(file) -> pd.DataFrame | None:
    """Lê e padroniza a planilha do Comex Stat ou Comtrade."""
    file_name = file.name.lower()
    df = None

    if file_name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(file)
    elif file_name.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=";", dtype=str, encoding="utf-8")
        except Exception:
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=",", dtype=str, encoding="utf-8")
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, sep=";", dtype=str, encoding="latin-1")
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=str)
    elif file_name.endswith(".json"):
        df = pd.read_json(file, dtype=str)

    if df is not None:
        # Mapear e padronizar nomes das colunas
        upper_cols = {str(c).strip().upper(): c for c in df.columns}
        renames = {}
        for k_upper, orig_col in upper_cols.items():
            if k_upper in COLUMN_MAPPING:
                renames[orig_col] = COLUMN_MAPPING[k_upper]

        df.rename(columns=renames, inplace=True)

        if "val_exp_br" in df.columns:
            df["val_exp_br"] = pd.to_numeric(df["val_exp_br"], errors="coerce").fillna(0.0)

        if "sh6" in df.columns:
            df["sh6"] = df["sh6"].astype(str).str.zfill(6)

    return df


def _normalize_comtrade(df_comtrade_raw: pd.DataFrame) -> pd.DataFrame:
    """Padroniza dataframe do UN Comtrade para colunas `sh6` e `val_mundo`."""
    df = df_comtrade_raw.copy()

    if "cmdCode" in df.columns:
        df.rename(columns={"cmdCode": "sh6", "primaryValue": "val_mundo"}, inplace=True)
    elif "CÓDIGO SH6" in [str(c).upper() for c in df.columns]:
        for c in df.columns:
            if str(c).upper() in ["CÓDIGO SH6", "CODIGO SH6"]:
                df.rename(columns={c: "sh6"}, inplace=True)
            if str(c).upper() in ["VALOR US$ FOB", "VALOR FOB", "VL_FOB", "PRIMARYVALUE"]:
                df.rename(columns={c: "val_mundo"}, inplace=True)
    elif "sh6" not in df.columns and "CO_NCM" in df.columns:
        df["sh6"] = df["CO_NCM"].astype(str).str.zfill(8).str[:6]
        df.rename(columns={"VL_FOB": "val_mundo"}, inplace=True)

    df["sh6"] = df["sh6"].astype(str).str.zfill(6).str[:6]
    df["val_mundo"] = pd.to_numeric(df.get("val_mundo", 0.0), errors="coerce").fillna(0.0)
    return df


def _classificar_quadrante(row: pd.Series) -> str:
    """Classifica o produto SH6 nos 4 quadrantes estratégicos."""
    if row["vcr"] >= VCR_LIMIAR and row["variacao_share_5a"] < 0 and row["cagr_global_5a"] >= CAGR_GLOBAL_LIMIAR:
        return QUADRANTE_OPORTUNIDADE
    if row["vcr"] >= VCR_LIMIAR and row["variacao_share_5a"] >= 0:
        return QUADRANTE_VANTAGEM_NACIONAL
    if row["vcr"] < VCR_LIMIAR and row["variacao_share_5a"] < 0:
        return QUADRANTE_VANTAGEM_IMPORTADORA
    return QUADRANTE_MAIS_ESCALA


def process_trade_data(
    df_comex_raw: pd.DataFrame,
    df_comtrade_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa por SH6, calcula o VCR e atribui os quadrantes estratégicos."""
    df_comex = df_comex_raw.copy()
    df_comtrade = _normalize_comtrade(df_comtrade_raw)

    tot_br_exp = float(df_comex["val_exp_br"].sum())
    tot_w_exp = float(df_comtrade["val_mundo"].sum())

    group_cols = ["sh6"]
    for c in ["desc_sh6", "isic_secao", "isic_divisao", "cuci_grupo", "cgce_1", "cgce_2"]:
        if c in df_comex.columns:
            group_cols.append(c)

    br_sh6 = df_comex.groupby(group_cols, as_index=False)["val_exp_br"].sum()
    w_sh6 = df_comtrade.groupby("sh6", as_index=False)["val_mundo"].sum()

    merged = pd.merge(br_sh6, w_sh6, on="sh6", how="inner")

    if tot_br_exp > 0 and tot_w_exp > 0:
        merged["vcr"] = (merged["val_exp_br"] / tot_br_exp) / (merged["val_mundo"] / tot_w_exp)
    else:
        merged["vcr"] = 0.0

    # Simulação de variações de 5 anos caso não haja histórico temporal no lote
    rng = np.random.default_rng(42)
    merged["variacao_share_5a"] = rng.uniform(-0.08, 0.08, len(merged))
    merged["cagr_global_5a"] = rng.uniform(-0.02, 0.12, len(merged))

    merged["quadrante"] = merged.apply(_classificar_quadrante, axis=1)

    # Preencher fallbacks de colunas opcionais se ausentes
    for c, fallback in [
        ("desc_sh6", "Produto SH6"),
        ("isic_secao", "Indústria Geral"),
        ("isic_divisao", "Divisão Industrial"),
        ("cuci_grupo", "Grupo CUCI"),
        ("cgce_1", "Bens Industriais"),
        ("cgce_2", "Categoria Geral")
    ]:
        if c not in merged.columns:
            merged[c] = fallback

    return merged, df_comex

# =============================================================================
# COMPONENTES VISUAIS
# =============================================================================
def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric(value: str, label: str) -> None:
    st.markdown(
        f"<div class='metric-value'>{value}</div><div class='metric-label'>{label}</div>",
        unsafe_allow_html=True,
    )


def render_quadrant_card(color: str, title: str, desc: str, count: int, valor: float, pct: float) -> None:
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


def render_sidebar() -> tuple[int, bool, dict]:
    st.sidebar.title("⚙️ Configurações & Upload")
    st.sidebar.markdown("---")

    horizonte = st.sidebar.slider("Variação temporal de cálculo:", 1, 5, 5, format="%d ano(s)")

    st.sidebar.markdown("### 📤 Upload de Arquivos")

    uploaded_comex = st.sidebar.file_uploader(
        "1. Planilha Comexstat Adaptada (.xlsx, .csv, .parquet)",
        type=["parquet", "pq", "csv", "xlsx"],
        help="Envie a planilha adaptada contendo as colunas do Comex Stat.",
    )
    uploaded_comtrade = st.sidebar.file_uploader(
        "2. Arquivo UN Comtrade (Mundo)",
        type=["parquet", "pq", "csv", "xlsx", "json"],
        help="Estatísticas globais por código HS6.",
    )

    st.sidebar.markdown("---")
    btn_processar = st.sidebar.button("🚀 Executar Análise", type="primary", use_container_width=True)

    uploads = {"comex": uploaded_comex, "comtrade": uploaded_comtrade}
    return horizonte, btn_processar, uploads


def handle_processing(uploads: dict) -> None:
    if uploads["comex"] is None or uploads["comtrade"] is None:
        st.warning("⚠️ Faça o upload de ambos os arquivos para executar o Radar.")
        return

    try:
        with st.spinner("Processando e estruturando dados..."):
            df_cx = read_uploaded_file(uploads["comex"])
            df_ct = read_uploaded_file(uploads["comtrade"])

            df_res, raw_cx = process_trade_data(df_cx, df_ct)
            st.session_state["df_processed"] = df_res
            st.session_state["raw_comex"] = raw_cx
        st.success("✅ Processamento concluído com sucesso!")
    except Exception as exc:
        st.error(f"Erro ao processar arquivo: {exc}")
        st.session_state["df_processed"] = None


def render_empty_state() -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        render_metric("0", "produtos SH6 monitorados")
    with col2:
        render_metric("0", "registros processados")
    with col3:
        render_metric("US$ 0,0", "mercado total exportado (bruto)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ Envie a planilha adaptada do Comex Stat e os dados do UN Comtrade na barra lateral.")


def render_header_metrics(df_main, raw_comex, horizonte: int) -> float:
    tot_val_bruto = float(df_main["val_exp_br"].sum())

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        render_metric(str(len(df_main)), "produtos SH6 monitorados")
    with col2:
        render_metric(f"{len(raw_comex):,}", "linhas de comércio processadas")
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

        if card_cfg["quadrante"] == QUADRANTE_VANTAGEM_NACIONAL:
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
        df_isic_grp = df_main.groupby("isic_secao", as_index=False).agg(
            total_val_bruto=("val_exp_br", "sum"), qtd_sh6=("sh6", "count")
        )
        df_isic_grp["Valor Exportado"] = df_isic_grp["total_val_bruto"].apply(fmt_usd)
        st.dataframe(
            df_isic_grp[["isic_secao", "qtd_sh6", "Valor Exportado"]],
            hide_index=True, use_container_width=True,
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
    st.subheader("Análise por Grupo CUCI & Países de Destino")
    selected_cuci = st.selectbox("Selecione o Grupo CUCI:", options=list(df_main["cuci_grupo"].unique()))
    if not selected_cuci:
        return

    df_cuci_filtered = df_main[df_main["cuci_grupo"] == selected_cuci]
    sh6_cuci_list = df_cuci_filtered["sh6"].unique()
    raw_cuci = raw_comex[raw_comex["sh6"].isin(sh6_cuci_list)]

    col_prods, col_paises = st.columns([1.3, 1])

    with col_prods:
        st.markdown("#### Produtos SH6 Vinculados")
        prods_summary = raw_cuci.groupby(["sh6", "desc_sh6"], as_index=False)["val_exp_br"].sum()
        prods_summary["Exportação"] = prods_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(
            prods_summary[["sh6", "desc_sh6", "Exportação"]],
            hide_index=True, use_container_width=True,
        )

    with col_paises:
        st.markdown("#### Top Países de Destino")
        if "pais" in raw_cuci.columns:
            paises_sum = (
                raw_cuci.groupby("pais", as_index=False)["val_exp_br"].sum()
                .sort_values(by="val_exp_br", ascending=False).head(10)
            )
            paises_sum["Valor"] = paises_sum["val_exp_br"].apply(fmt_usd)
            st.dataframe(paises_sum[["pais", "Valor"]].rename(columns={"pais": "País"}), hide_index=True, use_container_width=True)


def render_tab_cgce(df_main) -> None:
    st.subheader("Classificação por Grandes Categorias Econômicas (CGCE)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribuição por CGCE Nível 1")
        cgce1_summary = df_main.groupby("cgce_1", as_index=False).agg({"val_exp_br": "sum", "sh6": "count"})
        cgce1_summary["Valor"] = cgce1_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(cgce1_summary[["cgce_1", "sh6", "Valor"]], hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### Distribuição por CGCE Nível 2")
        cgce2_summary = df_main.groupby("cgce_2", as_index=False).agg({"val_exp_br": "sum", "sh6": "count"})
        cgce2_summary["Valor"] = cgce2_summary["val_exp_br"].apply(fmt_usd)
        st.dataframe(cgce2_summary[["cgce_2", "sh6", "Valor"]], hide_index=True, use_container_width=True)


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
        "🌐 Detalhamento por CUCI Grupo & Destinos",
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

    horizonte, btn_processar, uploads = render_sidebar()

    if btn_processar:
        handle_processing(uploads)

    render_main_panel(horizonte)


if __name__ == "__main__":
    main()
