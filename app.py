"""
Radar de Comércio Exterior & VCR / RCA - CNI / CIN
Aplicação para análise comparativa dos dados do Comex Stat e estatísticas globais.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
PAGE_TITLE = "Radar de Comércio Exterior & RCA - CNI / CIN"
PAGE_ICON = "📡"
APP_HEADER = "Radar de Inteligência Comercial e Vantagem Comparativa (RCA)"

# Países membros da União Europeia (UE-27)
EU_27_COUNTRIES = [
    "Alemanha", "Áustria", "Bélgica", "Bulgária", "Chipre", "Croácia",
    "Dinamarca", "Eslováquia", "Eslovênia", "Espanha", "Estônia", "Finlândia",
    "França", "Grécia", "Hungria", "Irlanda", "Itália", "Letônia",
    "Lituânia", "Luxemburgo", "Malta", "Países Baixos", "Polônia", "Portugal",
    "República Tcheca", "Romênia", "Suécia"
]

COLUMN_MAPPING = {
    "ANO": "ano",
    "REPORTER": "reporter",
    "PAÍS REPORTER": "reporter",
    "PAIS REPORTER": "reporter",
    "PARTNER": "partner",
    "PAÍS PARTNER": "partner",
    "PAIS PARTNER": "partner",
    "PAÍSES": "partner",
    "PAISES": "partner",
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
    "VALOR US$ FOB": "val_fob",
    "VALOR FOB": "val_fob",
    "VL_FOB": "val_fob",
    "PRIMARYVALUE": "val_fob"
}

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #F8FAFC; }
    .cni-title { font-size: 24px; font-weight: 800; color: #0F172A; margin-bottom: 4px; }
    .cni-subtitle { font-size: 13px; color: #64748B; margin-bottom: 16px; }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .metric-value { font-size: 22px; font-weight: 800; color: #0F172A; line-height: 1.2; }
    .metric-label { font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

    .product-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .product-header { font-size: 14px; font-weight: 700; color: #1E293B; }
    .product-sub { font-size: 11px; color: #64748B; margin-bottom: 10px; }
    .badge-rca {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-rca-high { background-color: #DCFCE7; color: #15803D; }
    .badge-rca-low { background-color: #FEE2E2; color: #B91C1C; }
    .badge-leader { background-color: #FEF3C7; color: #B45309; }
</style>
"""

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO E CÁLCULO
# =============================================================================
def fmt_usd(valor: float) -> str:
    """Formata valor monetário no padrão abreviado."""
    if pd.isna(valor) or valor == 0:
        return "US$ 0,0"
    abs_val = abs(valor)
    if abs_val >= 1e9:
        return f"US$ {valor / 1e9:.2f} Bi"
    if abs_val >= 1e6:
        return f"US$ {valor / 1e6:.2f} Mi"
    if abs_val >= 1e3:
        return f"US$ {valor / 1e3:.2f} Mil"
    return f"US$ {valor:.1f}"


def fmt_pct(valor: float) -> str:
    """Formata valor percentual."""
    if pd.isna(valor):
        return "0,0%"
    return f"{valor:.2f}%"


def calc_cagr(start_val: float, end_val: float, num_years: int) -> float:
    """Calcula a taxa de crescimento anual composta (CAGR)."""
    if start_val <= 0 or end_val <= 0 or num_years <= 0:
        return 0.0
    return ((end_val / start_val) ** (1 / num_years)) - 1

# =============================================================================
# LEITURA E PADRONIZAÇÃO DE DADOS
# =============================================================================
def read_uploaded_file(file) -> pd.DataFrame | None:
    """Lê arquivos e padroniza colunas do Comex Stat."""
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

    if df is not None:
        upper_cols = {str(c).strip().upper(): c for c in df.columns}
        renames = {orig: COLUMN_MAPPING[k] for k, orig in upper_cols.items() if k in COLUMN_MAPPING}
        df.rename(columns=renames, inplace=True)

        if "val_fob" in df.columns:
            df["val_fob"] = pd.to_numeric(df["val_fob"], errors="coerce").fillna(0.0)

        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)

        if "sh6" in df.columns:
            df["sh6"] = df["sh6"].astype(str).str.zfill(6)

        if "reporter" not in df.columns:
            df["reporter"] = "World"
        if "partner" not in df.columns:
            df["partner"] = "World"

        df["partner"] = df["partner"].replace({"Brasil": "Brazil", "BRASIL": "Brazil"})
        df["reporter"] = df["reporter"].replace({"Brasil": "Brazil", "BRASIL": "Brazil"})

        # Preencher fallbacks de categorias se não existirem no arquivo
        for col, default in [
            ("desc_sh6", "Sem Descrição"),
            ("isic_secao", "Indústria Geral"),
            ("isic_divisao", "Divisão Geral"),
            ("cuci_grupo", "Grupo CUCI"),
            ("cgce_1", "CGCE Nível 1"),
            ("cgce_2", "CGCE Nível 2")
        ]:
            if col not in df.columns:
                df[col] = default

    return df

# =============================================================================
# MOTOR DE PROCESSAMENTO COMPARATIVO
# =============================================================================
def process_trade_metrics(
    df_raw: pd.DataFrame,
    selected_reporters: list[str],
    partner_filter: str
) -> pd.DataFrame:
    """Processa a comparação temporal, RCA e classificações internacionais."""
    df = df_raw.copy()

    # 1. Agregação / Filtro de Reporters
    if "União Europeia (UE-27)" in selected_reporters:
        df_eu = df[df["reporter"].isin(EU_27_COUNTRIES)].copy()
        df_eu["reporter"] = "União Europeia (UE-27)"
        if "All" in selected_reporters or len(selected_reporters) > 1:
            df_others = df[~df["reporter"].isin(EU_27_COUNTRIES) & df["reporter"].isin(selected_reporters)]
            df = pd.concat([df_eu, df_others], ignore_index=True)
        else:
            df = df_eu
    elif "All" not in selected_reporters:
        df = df[df["reporter"].isin(selected_reporters)]

    # 2. Filtro de Partner
    if partner_filter != "All":
        df = df[df["partner"].isin([partner_filter, "World"])]

    if df.empty:
        return pd.DataFrame()

    anos = sorted([a for a in df["ano"].unique() if a > 0])
    if not anos:
        return pd.DataFrame()

    first_year, last_year = anos[0], anos[-1]
    num_years = max(last_year - first_year, 1)

    meta_cols = ["sh6", "desc_sh6", "isic_secao", "isic_divisao", "cuci_grupo", "cgce_1", "cgce_2"]

    # Pivot de dados por produto, parceiro e ano
    piv = df.pivot_table(
        index=meta_cols,
        columns=["partner", "ano"],
        values="val_fob",
        aggfunc="sum",
        fill_value=0.0
    )

    records = []
    tot_br_last = df[(df["partner"] == "Brazil") & (df["ano"] == last_year)]["val_fob"].sum()
    tot_world_last = df[(df["partner"] == "World") & (df["ano"] == last_year)]["val_fob"].sum()

    for idx, row in piv.iterrows():
        sh6, desc, isic_sec, isic_div, cuci, cgce1, cgce2 = idx

        w_first = row.get(("World", first_year), 0.0)
        w_last = row.get(("World", last_year), 0.0)
        br_first = row.get(("Brazil", first_year), 0.0)
        br_last = row.get(("Brazil", last_year), 0.0)

        cagr_world = calc_cagr(w_first, w_last, num_years)
        cagr_br = calc_cagr(br_first, br_last, num_years)

        share_first = (br_first / w_first) if w_first > 0 else 0.0
        share_last = (br_last / w_last) if w_last > 0 else 0.0
        delta_share = share_last - share_first

        # VCR / RCA (Índice de Balassa)
        if tot_br_last > 0 and tot_world_last > 0 and w_last > 0:
            rca = (br_last / tot_br_last) / (w_last / tot_world_last)
        else:
            rca = 0.0

        records.append({
            "sh6": sh6,
            "desc_sh6": desc,
            "isic_secao": isic_sec,
            "isic_divisao": isic_div,
            "cuci_grupo": cuci,
            "cgce_1": cgce1,
            "cgce_2": cgce2,
            "val_world_last": w_last,
            "val_br_last": br_last,
            "share_br_last": share_last * 100,
            "delta_share": delta_share * 100,
            "cagr_world": cagr_world * 100,
            "cagr_br": cagr_br * 100,
            "maior_crescimento": cagr_br > cagr_world,
            "rca": rca,
        })

    df_res = pd.DataFrame(records)
    return df_res.sort_values(by="val_world_last", ascending=False)

# =============================================================================
# INTERFACE E PAINÉIS
# =============================================================================
def configure_page():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(df_raw: pd.DataFrame | None):
    st.sidebar.title("⚙️ Filtros Comex Stat")
    st.sidebar.markdown("---")

    if df_raw is None or df_raw.empty:
        return ["All"], "Brazil"

    available_reporters = sorted(df_raw["reporter"].dropna().unique().tolist())
    has_eu = any(country in available_reporters for country in EU_27_COUNTRIES)

    reporter_options = ["All"]
    if has_eu:
        reporter_options.append("União Europeia (UE-27)")
    reporter_options.extend(available_reporters)

    st.sidebar.markdown("### 🌎 Seleção de Reporters")
    selected_reporters = st.sidebar.multiselect(
        "Selecione os mercados importadores:",
        options=reporter_options,
        default=["All"]
    )

    st.sidebar.markdown("### 🤝 Parceiro Comercial (Partner)")
    partner_filter = st.sidebar.radio(
        "Filtrar origem das exportações:",
        options=["Brazil", "World", "All"],
        index=0
    )

    return selected_reporters, partner_filter


def render_product_card(row: pd.Series):
    """Exibe os dados detalhados por produto SH6."""
    rca_badge = (
        f"<span class='badge-rca badge-rca-high'>RCA Competitivo ({row['rca']:.2f})</span>"
        if row["rca"] >= 1.0
        else f"<span class='badge-rca badge-rca-low'>RCA Desfavorável ({row['rca']:.2f})</span>"
    )

    leader_badge = (
        "<span class='badge-rca badge-leader'>🚀 Brasil Cresceu Acima do Mundo</span>"
        if row["maior_crescimento"]
        else ""
    )

    st.markdown(f"""
    <div class="product-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
            <div class="product-header">{row['desc_sh6']} <span style="color:#64748B;">(SH6 {row['sh6']})</span></div>
            <div>{rca_badge} {leader_badge}</div>
        </div>
        <div class="product-sub">
            ISIC Seção: <b>{row['isic_secao']}</b> | ISIC Divisão: <b>{row['isic_divisao']}</b> | CUCI: <b>{row['cuci_grupo']}</b> | CGCE: <b>{row['cgce_1']}</b>
        </div>
        <hr style="margin: 8px 0; border:0; border-top:1px solid #F1F5F9;">
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;">
            <div>
                <div class="metric-label">Importação Global</div>
                <div style="font-weight:700; font-size:14px;">{fmt_usd(row['val_world_last'])}</div>
            </div>
            <div>
                <div class="metric-label">Importação do Brasil</div>
                <div style="font-weight:700; font-size:14px; color:#0284C7;">{fmt_usd(row['val_br_last'])}</div>
            </div>
            <div>
                <div class="metric-label">Share Brasil</div>
                <div style="font-weight:700; font-size:14px;">{fmt_pct(row['share_br_last'])}</div>
            </div>
            <div>
                <div class="metric-label">CAGR Mundo vs BR</div>
                <div style="font-weight:700; font-size:12px;">
                    Mundo: {fmt_pct(row['cagr_world'])} | <span style="color:#059669;">BR: {fmt_pct(row['cagr_br'])}</span>
                </div>
            </div>
            <div>
                <div class="metric-label">Δ Share BR</div>
                <div style="font-weight:700; font-size:14px; color:{'#059669' if row['delta_share'] >= 0 else '#DC2626'};">
                    {'+' if row['delta_share'] > 0 else ''}{fmt_pct(row['delta_share'])}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    configure_page()
    st.markdown(f"<div class='cni-title'>{APP_HEADER}</div>", unsafe_allow_html=True)
    st.markdown("<div class='cni-subtitle'>Análise Comparativa de Dados do Comex Stat e Indicadores de Vantagem Comparativa</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Envie a planilha do Comex Stat (.parquet, .csv ou .xlsx)",
        type=["parquet", "pq", "csv", "xlsx"]
    )

    if uploaded_file is None:
        st.info("ℹ️ Insira um arquivo para carregar o painel comparativo.")
        return

    df_raw = read_uploaded_file(uploaded_file)
    if df_raw is None or df_raw.empty:
        st.error("⚠️ Não foi possível ler o arquivo enviado. Verifique a estrutura e tente novamente.")
        return

    selected_reporters, partner_filter = render_sidebar(df_raw)

    with st.spinner("Processando métricas e classificações internacionais..."):
        df_metrics = process_trade_metrics(df_raw, selected_reporters, partner_filter)

    if df_metrics.empty:
        st.warning("⚠️ Nenhum registro localizado para a combinação de filtros selecionada.")
        return

    # Métrica do topo
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Produtos Monitorados</div>
            <div class="metric-value">{len(df_metrics)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mercado Global Total</div>
            <div class="metric-value">{fmt_usd(df_metrics["val_world_last"].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Exportações do Brasil</div>
            <div class="metric-value">{fmt_usd(df_metrics["val_br_last"].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        tot_w = df_metrics["val_world_last"].sum()
        tot_b = df_metrics["val_br_last"].sum()
        share = (tot_b / tot_w * 100) if tot_w > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Share do Brasil</div>
            <div class="metric-value">{fmt_pct(share)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Navegação por Categorias e Classificações
    tab_prods, tab_isic, tab_cuci, tab_cgce, tab_table = st.tabs([
        "📦 Produtos (SH6)",
        "🎯 ISIC (Seção & Divisão)",
        "🌐 CUCI (Grupo)",
        "📦 CGCE (Níveis 1 e 2)",
        "📊 Tabela Consolidada"
    ])

    with tab_prods:
        st.subheader("Desempenho por Produto SH6")
        search = st.text_input("🔍 Filtrar código SH6 ou descrição:", "")
        df_view = df_metrics.copy()
        if search:
            df_view = df_view[
                df_view["desc_sh6"].str.contains(search, case=False, na=False) |
                df_view["sh6"].str.contains(search, case=False, na=False)
            ]
        for _, row in df_view.iterrows():
            render_product_card(row)

    with tab_isic:
        st.subheader("Agregação por Classificação ISIC")
        isic_grp = df_metrics.groupby(["isic_secao", "isic_divisao"], as_index=False).agg({
            "val_world_last": "sum",
            "val_br_last": "sum",
            "sh6": "count"
        })
        isic_grp["Share Brasil"] = (isic_grp["val_br_last"] / isic_grp["val_world_last"] * 100).fillna(0).apply(fmt_pct)
        isic_grp["val_world_last"] = isic_grp["val_world_last"].apply(fmt_usd)
        isic_grp["val_br_last"] = isic_grp["val_br_last"].apply(fmt_usd)
        st.dataframe(isic_grp.rename(columns={
            "isic_secao": "ISIC Seção",
            "isic_divisao": "ISIC Divisão",
            "sh6": "Qtd Produtos",
            "val_world_last": "Mundo",
            "val_br_last": "Brasil"
        }), use_container_width=True, hide_index=True)

    with tab_cuci:
        st.subheader("Agregação por Grupo CUCI")
        cuci_grp = df_metrics.groupby("cuci_grupo", as_index=False).agg({
            "val_world_last": "sum",
            "val_br_last": "sum",
            "sh6": "count"
        })
        cuci_grp["Share Brasil"] = (cuci_grp["val_br_last"] / cuci_grp["val_world_last"] * 100).fillna(0).apply(fmt_pct)
        cuci_grp["val_world_last"] = cuci_grp["val_world_last"].apply(fmt_usd)
        cuci_grp["val_br_last"] = cuci_grp["val_br_last"].apply(fmt_usd)
        st.dataframe(cuci_grp.rename(columns={
            "cuci_grupo": "Grupo CUCI",
            "sh6": "Qtd Produtos",
            "val_world_last": "Mundo",
            "val_br_last": "Brasil"
        }), use_container_width=True, hide_index=True)

    with tab_cgce:
        st.subheader("Agregação por Categoria CGCE")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### CGCE Nível 1")
            cgce1 = df_metrics.groupby("cgce_1", as_index=False).agg({"val_br_last": "sum", "val_world_last": "sum"})
            cgce1["val_world_last"] = cgce1["val_world_last"].apply(fmt_usd)
            cgce1["val_br_last"] = cgce1["val_br_last"].apply(fmt_usd)
            st.dataframe(cgce1, use_container_width=True, hide_index=True)
        with col_c2:
            st.markdown("#### CGCE Nível 2")
            cgce2 = df_metrics.groupby("cgce_2", as_index=False).agg({"val_br_last": "sum", "val_world_last": "sum"})
            cgce2["val_world_last"] = cgce2["val_world_last"].apply(fmt_usd)
            cgce2["val_br_last"] = cgce2["val_br_last"].apply(fmt_usd)
            st.dataframe(cgce2, use_container_width=True, hide_index=True)

    with tab_table:
        st.subheader("Matriz Geral de Indicadores")
        out_df = df_metrics.copy()
        out_df["val_world_last"] = out_df["val_world_last"].apply(fmt_usd)
        out_df["val_br_last"] = out_df["val_br_last"].apply(fmt_usd)
        out_df["share_br_last"] = out_df["share_br_last"].apply(fmt_pct)
        out_df["delta_share"] = out_df["delta_share"].apply(fmt_pct)
        out_df["cagr_world"] = out_df["cagr_world"].apply(fmt_pct)
        out_df["cagr_br"] = out_df["cagr_br"].apply(fmt_pct)
        out_df["rca"] = out_df["rca"].round(2)

        st.dataframe(out_df.rename(columns={
            "sh6": "SH6",
            "desc_sh6": "Descrição",
            "isic_secao": "ISIC Seção",
            "isic_divisao": "ISIC Divisão",
            "cuci_grupo": "CUCI Grupo",
            "cgce_1": "CGCE N1",
            "cgce_2": "CGCE N2",
            "val_world_last": "Mundo (Últ. Ano)",
            "val_br_last": "Brasil (Últ. Ano)",
            "share_br_last": "Share BR",
            "delta_share": "Δ Share BR",
            "cagr_world": "CAGR Mundo",
            "cagr_br": "CAGR BR",
            "maior_crescimento": "BR Superou Mundo?",
            "rca": "RCA / VCR"
        }), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
