"""
Radar de Comércio Exterior & VCR / RCA - CNI / CIN
Aplicação para análise de importações globais e desempenho das exportações brasileiras.
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
    "DESCRIÇÃO CGCE NÍVEL 2": "cgce_2",
    "CÓDIGO CGCE NÍVEL 1": "cgce_1_cod",
    "DESCRIÇÃO CGCE NÍVEL 1": "cgce_1",
    "CÓDIGO CUCI GRUPO": "cuci_cod",
    "DESCRIÇÃO CUCI GRUPO": "cuci_grupo",
    "CÓDIGO ISIC DIVISÃO": "isic_divisao_cod",
    "DESCRIÇÃO ISIC DIVISÃO": "isic_divisao",
    "CÓDIGO ISIC SEÇÃO": "isic_secao_cod",
    "DESCRIÇÃO ISIC SEÇÃO": "isic_secao",
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
    
    /* Metricas em cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .metric-value { font-size: 22px; font-weight: 800; color: #0F172A; line-height: 1.2; }
    .metric-label { font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

    /* Product Cards Visual */
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
# FUNÇÕES DE CÁLCULO FINANCEIRO E ESTATÍSTICO
# =============================================================================
def fmt_usd(valor: float) -> str:
    """Formata valor em dólares (Bi, Mi, Mil ou Unidade)."""
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
    """Formata percentual."""
    if pd.isna(valor):
        return "0,0%"
    return f"{valor:.2f}%"


def calc_cagr(start_val: float, end_val: float, num_years: int) -> float:
    """Calcula a Taxa de Crescimento Anual Composta (CAGR)."""
    if start_val <= 0 or end_val <= 0 or num_years <= 0:
        return 0.0
    return ((end_val / start_val) ** (1 / num_years)) - 1

# =============================================================================
# CARGA E TRATAMENTO DE DADOS
# =============================================================================
def read_uploaded_file(file) -> pd.DataFrame | None:
    """Carrega e padroniza a planilha de comércio exterior."""
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
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype(int)

        if "sh6" in df.columns:
            df["sh6"] = df["sh6"].astype(str).str.zfill(6)

        # Tratar Reporter/Partner genéricos caso não declarados
        if "reporter" not in df.columns:
            df["reporter"] = "World"
        if "partner" not in df.columns:
            df["partner"] = "World"

        # Padronização textual de nomes comuns de parceiros/reporters
        df["partner"] = df["partner"].replace({"Brasil": "Brazil", "BRASIL": "Brazil"})
        df["reporter"] = df["reporter"].replace({"Brasil": "Brazil", "BRASIL": "Brazil"})

    return df

# =============================================================================
# PROCESSAMENTO DE INDICADORES E RCA
# =============================================================================
def process_trade_metrics(
    df_raw: pd.DataFrame,
    selected_reporters: list[str],
    partner_filter: str
) -> pd.DataFrame:
    """Executa as agregações, cálculo de CAGR, variação de share e RCA."""
    df = df_raw.copy()

    # 1. Filtrar Reporters
    if "União Europeia (UE-27)" in selected_reporters:
        df_reporters = df[df["reporter"].isin(EU_27_COUNTRIES)].copy()
        df_reporters["reporter"] = "União Europeia (UE-27)"
        if "All" in selected_reporters or len(selected_reporters) > 1:
            df_others = df[~df["reporter"].isin(EU_27_COUNTRIES) & df["reporter"].isin(selected_reporters)]
            df = pd.concat([df_reporters, df_others], ignore_index=True)
        else:
            df = df_reporters
    elif "All" not in selected_reporters:
        df = df[df["reporter"].isin(selected_reporters)]

    # 2. Filtrar Partners (Focus em Brazil e World)
    if partner_filter != "All":
        df = df[df["partner"].isin([partner_filter, "World"])]

    if df.empty:
        return pd.DataFrame()

    anos = sorted(df["ano"].unique())
    first_year, last_year = anos[0], anos[-1]
    num_years = max(last_year - first_year, 1)

    # Identificar colunas descritivas de produto
    meta_cols = ["sh6"]
    for c in ["desc_sh6", "isic_secao", "isic_divisao", "cuci_grupo", "cgce_1", "cgce_2"]:
        if c in df.columns:
            meta_cols.append(c)

    # Agrupamentos para o cálculo do RCA e Shares
    # Total de Importações do Mundo e do Brasil por Produto e Ano
    piv = df.pivot_table(
        index=meta_cols,
        columns=["partner", "ano"],
        values="val_fob",
        aggfunc="sum",
        fill_value=0.0
    )

    records = []
    
    # Totais Globais de Importação do Período para o RCA
    tot_br_all_prods_last = df[df["partner"] == "Brazil"][df["ano"] == last_year]["val_fob"].sum()
    tot_world_all_prods_last = df[df["partner"] == "World"][df["ano"] == last_year]["val_fob"].sum()

    for idx, row in piv.iterrows():
        sh6 = idx[0] if isinstance(idx, tuple) else idx
        desc = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else "Produto SH6 " + str(sh6)
        
        # Valores de Importação do Mundo
        w_first = row.get(("World", first_year), 0.0)
        w_last = row.get(("World", last_year), 0.0)
        
        # Valores de Importação Vindos do Brasil
        br_first = row.get(("Brazil", first_year), 0.0)
        br_last = row.get(("Brazil", last_year), 0.0)

        # CAGR
        cagr_world = calc_cagr(w_first, w_last, num_years)
        cagr_br = calc_cagr(br_first, br_last, num_years)

        # Participação do Brasil
        share_first = (br_first / w_first) if w_first > 0 else 0.0
        share_last = (br_last / w_last) if w_last > 0 else 0.0
        delta_share = share_last - share_first

        # Brasil teve o maior crescimento? (Comparativo CAGR Brasil vs CAGR World)
        brasil_maior_crescimento = cagr_br > cagr_world

        # Cálculo do VCR / RCA (Balassa Index) no último ano
        if tot_br_all_prods_last > 0 and tot_world_all_prods_last > 0 and w_last > 0:
            rca = (br_last / tot_br_all_prods_last) / (w_last / tot_world_all_prods_last)
        else:
            rca = 0.0

        records.append({
            "sh6": sh6,
            "desc_sh6": desc,
            "val_world_last": w_last,
            "val_br_last": br_last,
            "share_br_last": share_last * 100,
            "delta_share": delta_share * 100,
            "cagr_world": cagr_world * 100,
            "cagr_br": cagr_br * 100,
            "maior_crescimento": brasil_maior_crescimento,
            "rca": rca,
            "isic_secao": idx[2] if isinstance(idx, tuple) and len(idx) > 2 else "Indústria",
        })

    df_res = pd.DataFrame(records)
    return df_res.sort_values(by="val_world_last", ascending=False)

# =============================================================================
# INTERFACE DE USUÁRIO
# =============================================================================
def configure_page():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(df_raw: pd.DataFrame | None):
    st.sidebar.title("⚙️ Filtros & Seleção")
    st.sidebar.markdown("---")

    if df_raw is None or df_raw.empty:
        return ["All"], "Brazil", None

    # Lista de Reporters disponíveis no arquivo
    available_reporters = sorted(df_raw["reporter"].dropna().unique().tolist())
    
    # Adicionar opção de União Europeia caso haja países membros no dataset
    has_eu = any(country in available_reporters for country in EU_27_COUNTRIES)
    reporter_options = ["All"]
    if has_eu:
        reporter_options.append("União Europeia (UE-27)")
    reporter_options.extend(available_reporters)

    st.sidebar.markdown("### 🌎 Seleção de Reporters (Importador)")
    selected_reporters = st.sidebar.multiselect(
        "Filtre os países importadores:",
        options=reporter_options,
        default=["All"],
        help="Escolha países individuais, 'All' para todos, ou agregue a UE-27."
    )

    st.sidebar.markdown("### 🤝 Parceiro Comercial (Partner)")
    partner_filter = st.sidebar.radio(
        "Selecione a origem das exportações:",
        options=["Brazil", "World", "All"],
        index=0,
        help="Defina se o foco da análise será o Brasil ou o Mercado Global."
    )

    st.sidebar.markdown("---")
    return selected_reporters, partner_filter


def render_product_card_detailed(row: pd.Series):
    """Renderização aprimorada em cards para cada produto SH6."""
    rca_badge = (
        f"<span class='badge-rca badge-rca-high'>RCA Competitivo ({row['rca']:.2f})</span>"
        if row["rca"] >= 1.0
        else f"<span class='badge-rca badge-rca-low'>RCA Desfavorável ({row['rca']:.2f})</span>"
    )
    
    leader_badge = (
        "<span class='badge-rca badge-leader'>🚀 Brasil Cresceu Acima da Média Mundial</span>"
        if row["maior_crescimento"]
        else ""
    )

    st.markdown(f"""
    <div class="product-box">
        <div style="display:flex; justify-shadow:space-between; align-items:center; margin-bottom: 8px;">
            <div class="product-header">{row['desc_sh6']} <span style="color:#64748B;">(SH6 {row['sh6']})</span></div>
            <div>{rca_badge} {leader_badge}</div>
        </div>
        <div class="product-sub">Setor / Seção: <b>{row['isic_secao']}</b></div>
        <hr style="margin: 8px 0; border:0; border-top:1px solid #F1F5F9;">
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: left;">
            <div>
                <div class="metric-label">Importação Global</div>
                <div style="font-weight:700; font-size:15px;">{fmt_usd(row['val_world_last'])}</div>
            </div>
            <div>
                <div class="metric-label">Importação do Brasil</div>
                <div style="font-weight:700; font-size:15px; color:#0284C7;">{fmt_usd(row['val_br_last'])}</div>
            </div>
            <div>
                <div class="metric-label">Share Brasil (Últ. Ano)</div>
                <div style="font-weight:700; font-size:15px;">{fmt_pct(row['share_br_last'])}</div>
            </div>
            <div>
                <div class="metric-label">CAGR Global vs Brasil</div>
                <div style="font-weight:700; font-size:13px;">
                    Mundo: {fmt_pct(row['cagr_world'])} | <span style="color:#059669;">BR: {fmt_pct(row['cagr_br'])}</span>
                </div>
            </div>
            <div>
                <div class="metric-label">Δ Share Brasil</div>
                <div style="font-weight:700; font-size:15px; color:{'#059669' if row['delta_share'] >= 0 else '#DC2626'};">
                    {'+' if row['delta_share'] > 0 else ''}{fmt_pct(row['delta_share'])}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    configure_page()
    st.markdown(f"<div class='cni-title'>{APP_HEADER}</div>", unsafe_allow_html=True)
    st.markdown("<div class='cni-subtitle'>Análise Estratégica de Vantagem Comparativa Revelada (RCA), CAGR e Market Share do Brasil</div>", unsafe_allow_html=True)

    # File Uploader
    uploaded_file = st.file_uploader(
        "📤 Envie a base de dados em formato Parquet, CSV ou Excel",
        type=["parquet", "pq", "csv", "xlsx"]
    )

    if uploaded_file is None:
        st.info("ℹ️ Faça o upload de um arquivo para iniciar o processamento das métricas.")
        return

    df_raw = read_uploaded_file(uploaded_file)
    if df_raw is None or df_raw.empty:
        st.error("⚠️ O arquivo enviado não contém dados válidos ou colunas reconhecidas.")
        return

    # Renderizar Filtros
    selected_reporters, partner_filter = render_sidebar(df_raw)

    # Processamento dos Dados
    with st.spinner("Processando indicadores de comércio exterior..."):
        df_metrics = process_trade_metrics(df_raw, selected_reporters, partner_filter)

    if df_metrics.empty:
        st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados.")
        return

    # Visão Consolidada de Topo
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Produtos Analisados</div>
            <div class="metric-value">{len(df_metrics)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        tot_world = df_metrics["val_world_last"].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Importação Global Total</div>
            <div class="metric-value">{fmt_usd(tot_world)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        tot_br = df_metrics["val_br_last"].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Importado do Brasil</div>
            <div class="metric-value">{fmt_usd(tot_br)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        share_medio = (tot_br / tot_world * 100) if tot_world > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Share Global do Brasil</div>
            <div class="metric-value">{fmt_pct(share_medio)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs de Exibição
    tab1, tab2 = st.tabs(["📦 Visualização Aprimorada por Produto", "📊 Tabela Consolidada de Dados"])

    with tab1:
        st.subheader("Desempenho por Produto SH6")
        
        # Filtro rápido por código/descrição
        search_term = st.text_input("🔍 Filtrar produto por código ou descrição:", "")
        df_filtered = df_metrics.copy()
        
        if search_term:
            df_filtered = df_filtered[
                df_filtered["desc_sh6"].str.contains(search_term, case=False, na=False) |
                df_filtered["sh6"].str.contains(search_term, case=False, na=False)
            ]

        for _, row in df_filtered.iterrows():
            render_product_card_detailed(row)

    with tab2:
        st.subheader("Matriz Geral de Indicadores de Comércio Exterior")
        
        # Formatando tabela para exportação
        display_df = df_metrics.copy()
        display_df["val_world_last"] = display_df["val_world_last"].apply(fmt_usd)
        display_df["val_br_last"] = display_df["val_br_last"].apply(fmt_usd)
        display_df["share_br_last"] = display_df["share_br_last"].apply(fmt_pct)
        display_df["delta_share"] = display_df["delta_share"].apply(fmt_pct)
        display_df["cagr_world"] = display_df["cagr_world"].apply(fmt_pct)
        display_df["cagr_br"] = display_df["cagr_br"].apply(fmt_pct)
        display_df["rca"] = display_df["rca"].round(2)

        st.dataframe(
            display_df[[
                "sh6", "desc_sh6", "val_world_last", "val_br_last",
                "share_br_last", "delta_share", "cagr_world", "cagr_br",
                "maior_crescimento", "rca"
            ]].rename(columns={
                "sh6": "Código SH6",
                "desc_sh6": "Descrição Produto",
                "val_world_last": "Imp. Mundo (Últ. Ano)",
                "val_br_last": "Imp. Brasil (Últ. Ano)",
                "share_br_last": "Share Brasil",
                "delta_share": "Δ Share BR",
                "cagr_world": "CAGR Mundo",
                "cagr_br": "CAGR Brasil",
                "maior_crescimento": "BR Liderou Crescimento?",
                "rca": "RCA / VCR"
            }),
            use_container_width=True,
            hide_index=True
        )


if __name__ == "__main__":
    main()
