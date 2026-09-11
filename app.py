import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Análise de Diversificação das Exportações BR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customização CSS leve
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MOTOR DE CARREGAMENTO E PROCESSAMENTO DE DADOS
# ==============================================================================
@st.cache_data(show_spinner="Carregando e processando arquivo...")
def load_uploaded_file(file):
    """Lê arquivos enviados nos formatos CSV, Parquet, JSON ou Excel."""
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith('.parquet'):
            return pd.read_parquet(file)
        elif name.endswith('.csv'):
            return pd.read_csv(file)
        elif name.endswith('.json'):
            return pd.read_json(file)
        elif name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error(f"Formato não suportado: {file.name}")
            return None
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file.name}: {e}")
        return None

def calc_cagr(start_val, end_val, periods):
    """Calcula a Taxa de Crescimento Anual Composta (CAGR)."""
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0 or periods <= 0:
        return np.nan
    return ((end_val / start_val) ** (1 / periods)) - 1

def generate_mock_comtrade():
    """Gera dados fictícios coerentes do UN Comtrade para demonstração."""
    anos = [2020, 2021, 2022, 2023, 2024]
    sh6_list = [
        ("090111", "Café não torrado, não descafeinado", "071", "Café e sucedâneos", "01", "Agricultura e Pecuária", "A", "Agricultura, Pecuária e Florestal", "111", "Bens de Consumo"),
        ("120190", "Soja, mesmo triturada", "222", "Sementes oleaginosas", "01", "Agricultura e Pecuária", "A", "Agricultura, Pecuária e Florestal", "121", "Insumos Intermediários"),
        ("260111", "Minérios de ferro não aglomerados", "281", "Minérios de ferro", "07", "Extração de Minerais Metálicos", "B", "Indústrias Extrativas", "121", "Insumos Intermediários"),
        ("270900", "Óleos brutos de petróleo", "333", "Petróleo bruto", "06", "Extração de Petróleo e Gás", "B", "Indústrias Extrativas", "310", "Combustíveis e Lubrificantes"),
        ("470321", "Pasta química de madeira (celulose)", "251", "Pasta de madeira", "17", "Fabricação de Celulose e Papel", "C", "Indústria de Transformação", "121", "Insumos Intermediários"),
        ("020230", "Carne bovina desossada congelada", "012", "Carne bovina", "10", "Fabricação de Produtos Alimentícios", "C", "Indústria de Transformação", "111", "Bens de Consumo"),
        ("880230", "Aviões e outras aeronaves > 15.000kg", "792", "Aeronaves e equipamentos", "30", "Fabricação de Outros Equipamentos de Transporte", "C", "Indústria de Transformação", "210", "Bens de Capital"),
        ("720711", "Produtos semimanufaturados de ferro/aço", "672", "Ferro/Aço em formas primárias", "24", "Metalurgia", "C", "Indústria de Transformação", "121", "Insumos Intermediários")
    ]
    
    rows = []
    np.random.seed(42)
    for ano in anos:
        for item in sh6_list:
            base_world = np.random.uniform(500, 5000) * (1 + (ano - 2020) * 0.05)
            base_br = base_world * np.random.uniform(0.05, 0.25)
            rows.append({
                "Ano": ano,
                "Código SH6": item[0],
                "Descrição SH6": item[1],
                "Código CUCI Grupo": item[2],
                "Descrição CUCI Grupo": item[3],
                "Código ISIC Divisão": item[4],
                "Descrição ISIC Divisão": item[5],
                "Código ISIC Seção": item[6],
                "Descrição ISIC Seção": item[7],
                "Código CGCE Nível 1": item[8],
                "Descrição CGCE Nível 1": item[9],
                "Valor World FOB": round(base_world * 1000, 2),
                "Valor BR FOB": round(base_br * 1000, 2)
            })
    return pd.DataFrame(rows)

def generate_mock_comexstat():
    """Gera dados fictícios do Comex Stat por Estado (UF) para demonstração."""
    anos = [2020, 2021, 2022, 2023, 2024]
    ufs = ["SP", "MG", "PR", "RS", "MT", "PA", "RJ", "TO"]
    sh6_codes = ["090111", "120190", "260111", "270900", "470321", "020230"]
    
    rows = []
    np.random.seed(100)
    for ano in anos:
        for sh in sh6_codes:
            for uf in np.random.choice(ufs, size=3, replace=False):
                rows.append({
                    "Ano": ano,
                    "UF": uf,
                    "Países": "Mundo",
                    "Código SH6": sh,
                    "Valor US$ FOB": round(np.random.uniform(10, 500) * 1000, 2)
                })
    return pd.DataFrame(rows)

@st.cache_data
def process_analytics(df):
    """Calcula métricas agregadas de competitividade (RCA, CAGR, Quadrantes)."""
    anos = sorted(df["Ano"].unique())
    t_ini, t_fim = anos[0], anos[-1]
    n_periodos = t_fim - t_ini

    # Agrupa por SH6 e metadados nos anos inicial e final
    group_cols = [
        "Código SH6", "Descrição SH6", "Código CUCI Grupo", "Descrição CUCI Grupo",
        "Código ISIC Divisão", "Descrição ISIC Divisão", "Código ISIC Seção", "Descrição ISIC Seção"
    ]
    
    df_ini = df[df["Ano"] == t_ini].groupby(group_cols)[["Valor World FOB", "Valor BR FOB"]].sum().reset_index()
    df_fim = df[df["Ano"] == t_fim].groupby(group_cols)[["Valor World FOB", "Valor BR FOB"]].sum().reset_index()

    merged = pd.merge(df_ini, df_fim, on=group_cols, suffixes=("_ini", "_fim"))

    # Totais Globais
    total_world_ini = merged["Valor World FOB_ini"].sum()
    total_world_fim = merged["Valor World FOB_fim"].sum()
    total_br_ini = merged["Valor BR FOB_ini"].sum()
    total_br_fim = merged["Valor BR FOB_fim"].sum()

    # RCA (Revealed Comparative Advantage)
    merged["RCA_ini"] = (merged["Valor BR FOB_ini"] / total_br_ini) / (merged["Valor World FOB_ini"] / total_world_ini)
    merged["RCA_fim"] = (merged["Valor BR FOB_fim"] / total_br_fim) / (merged["Valor World FOB_fim"] / total_world_fim)

    # Market Share
    merged["Share_BR_ini"] = merged["Valor BR FOB_ini"] / merged["Valor World FOB_ini"]
    merged["Share_BR_fim"] = merged["Valor BR FOB_fim"] / merged["Valor World FOB_fim"]
    merged["Var_Share_BR"] = merged["Share_BR_fim"] - merged["Share_BR_ini"]

    # CAGRs
    merged["CAGR_World"] = merged.apply(lambda r: calc_cagr(r["Valor World FOB_ini"], r["Valor World FOB_fim"], n_periodos), axis=1)
    merged["CAGR_BR"] = merged.apply(lambda r: calc_cagr(r["Valor BR FOB_ini"], r["Valor BR FOB_fim"], n_periodos), axis=1)

    # CAGR Médio Ponderado Global e do Brasil
    cagr_world_avg = calc_cagr(total_world_ini, total_world_fim, n_periodos)
    cagr_br_avg = calc_cagr(total_br_ini, total_br_fim, n_periodos)

    # Classificação em Quadrantes
    def get_quadrant(row):
        mkt_growth = row["CAGR_World"] > cagr_world_avg
        space_gain = row["Var_Share_BR"] > 0
        if mkt_growth and space_gain:
            return "Ganho de Espaço & Mercado Cresce (Oportunidade)"
        elif mkt_growth and not space_gain:
            return "Perda de Espaço & Mercado Cresce (Ameaça)"
        elif not mkt_growth and space_gain:
            return "Ganho de Espaço & Mercado Cai (Vulnerabilidade)"
        else:
            return "Perda de Espaço & Mercado Cai (Retirada)"

    merged["Quadrante"] = merged.apply(get_quadrant, axis=1)

    return merged, cagr_world_avg, cagr_br_avg, t_ini, t_fim


# ==============================================================================
# 3. INTERFACE LATERAL (UPLOAD DE DADOS E FILTROS GLOBAIS)
# ==============================================================================
st.sidebar.title("🛠️ Configurações & Dados")

st.sidebar.subheader("1. Base UN Comtrade")
comtrade_file = st.sidebar.file_uploader(
    "Upload UN Comtrade (CSV, Parquet, JSON, Excel)",
    type=["csv", "parquet", "json", "xlsx", "xls"],
    key="comtrade"
)

st.sidebar.subheader("2. Base Comex Stat (Opcional)")
comexstat_file = st.sidebar.file_uploader(
    "Upload Comex Stat por UF (CSV, Parquet, Excel)",
    type=["csv", "parquet", "xlsx", "xls"],
    key="comexstat"
)

# Carregamento efetivo ou mock
if comtrade_file is not None:
    df_raw = load_uploaded_file(comtrade_file)
else:
    st.sidebar.info("💡 Usando dados demonstrativos do UN Comtrade.")
    df_raw = generate_mock_comtrade()

if comexstat_file is not None:
    df_uf_raw = load_uploaded_file(comexstat_file)
else:
    df_uf_raw = generate_mock_comexstat()

# Processamento analítico
analytics_df, cagr_w_avg, cagr_b_avg, t_start, t_end = process_analytics(df_raw)


# ==============================================================================
# 4. PÁGINAS DO SISTEMA
# ==============================================================================

# --- PÁGINA 1: DASHBOARD EXECUTIVE & QUADRANTES ---
def page_dashboard():
    st.title("📊 Dashboard Executivo de Diversificação")
    st.caption(f"Análise comparativa do período de **{t_start} a {t_end}**")

    # KPIs Principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR Médio Global", f"{cagr_w_avg:.2%}")
    col2.metric("CAGR Total Brasil", f"{cagr_b_avg:.2%}")
    
    diff_cagr = cagr_b_avg - cagr_w_avg
    if diff_cagr > 0:
        col3.metric("Performance Relativa", f"+{diff_cagr:.2%}", delta="Acima do Mundo", delta_color="normal")
    else:
        col3.metric("Performance Relativa", f"{diff_cagr:.2%}", delta="Abaixo do Mundo", delta_color="inverse")

    n_oportunidades = len(analytics_df[analytics_df["Quadrante"] == "Ganho de Espaço & Mercado Cresce (Oportunidade)"])
    col4.metric("Oportunidades (SH6)", f"{n_oportunidades} de {len(analytics_df)}")

    st.markdown("---")

    # Matriz de Quadrantes
    st.subheader("📍 Matriz Estratégica de Posicionamento (Quadrantes de RCA)")
    
    fig = px.scatter(
        analytics_df,
        x="Var_Share_BR",
        y="CAGR_World",
        size="Valor BR FOB_fim",
        color="Quadrante",
        hover_name="Descrição SH6",
        hover_data=["Código SH6", "RCA_fim", "CAGR_BR"],
        labels={
            "Var_Share_BR": "Variação de Share do Brasil (Início vs Fim)",
            "CAGR_World": "CAGR do Mercado Global",
            "Valor BR FOB_fim": "Exportações BR (US$)"
        },
        color_discrete_map={
            "Ganho de Espaço & Mercado Cresce (Oportunidade)": "#2ea44f",
            "Perda de Espaço & Mercado Cresce (Ameaça)": "#cb2431",
            "Ganho de Espaço & Mercado Cai (Vulnerabilidade)": "#dbab09",
            "Retirada / Declínio": "#6a737d"
        },
        height=550
    )

    # Linhas de referência dos quadrantes
    fig.add_hline(y=cagr_w_avg, line_dash="dash", line_color="gray", annotation_text="CAGR Médio Global")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Share Neutro")

    st.plotly_chart(fig, use_container_width=True)

    # Análise Setorial por ISIC
    st.markdown("---")
    st.subheader("🎯 Oportunidades e Ameaças por Setor ISIC")

    tab1, tab2 = st.tabs(["ISIC Divisão", "ISIC Seção"])

    with tab1:
        isic_div = analytics_df.groupby(["Código ISIC Divisão", "Descrição ISIC Divisão"]).agg(
            Exp_BR_Fim=("Valor BR FOB_fim", "sum"),
            Var_Share_Media=("Var_Share_BR", "mean"),
            CAGR_World_Media=("CAGR_World", "mean")
        ).reset_index()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🟢 Maiores Potenciais (Ganho Médio de Share)**")
            st.dataframe(
                isic_div.sort_values(by="Var_Share_Media", ascending=False).head(5)[
                    ["Código ISIC Divisão", "Descrição ISIC Divisão", "Var_Share_Media", "Exp_BR_Fim"]
                ],
                use_container_width=True, hide_index=True
            )
        with col_b:
            st.markdown("**🔴 Maiores Ameaças (Perda Média de Share)**")
            st.dataframe(
                isic_div.sort_values(by="Var_Share_Media", ascending=True).head(5)[
                    ["Código ISIC Divisão", "Descrição ISIC Divisão", "Var_Share_Media", "Exp_BR_Fim"]
                ],
                use_container_width=True, hide_index=True
            )

    with tab2:
        isic_sec = analytics_df.groupby(["Código ISIC Seção", "Descrição ISIC Seção"]).agg(
            Exp_BR_Fim=("Valor BR FOB_fim", "sum"),
            Var_Share_Media=("Var_Share_BR", "mean"),
            CAGR_World_Media=("CAGR_World", "mean")
        ).reset_index()

        st.dataframe(
            isic_sec.sort_values(by="Exp_BR_Fim", ascending=False),
            use_container_width=True, hide_index=True
        )


# --- PÁGINA 2: TABELA ANALÍTICA SH6 ---
def page_sh6():
    st.title("📋 Tabela Analítica Completa por SH6")
    
    # Filtros Avançados
    col_f1, col_f2, col_f3 = st.columns(3)
    
    list_cuci = ["Todos"] + sorted(analytics_df["Descrição CUCI Grupo"].unique().tolist())
    list_isic_sec = ["Todos"] + sorted(analytics_df["Descrição ISIC Seção"].unique().tolist())
    list_isic_div = ["Todos"] + sorted(analytics_df["Descrição ISIC Divisão"].unique().tolist())

    sel_cuci = col_f1.selectbox("Filtrar por CUCI Grupo", list_cuci)
    sel_isic_sec = col_f2.selectbox("Filtrar por ISIC Seção", list_isic_sec)
    sel_isic_div = col_f3.selectbox("Filtrar por ISIC Divisão", list_isic_div)

    # Aplicação dos Filtros
    filtered = analytics_df.copy()
    if sel_cuci != "Todos":
        filtered = filtered[filtered["Descrição CUCI Grupo"] == sel_cuci]
    if sel_isic_sec != "Todos":
        filtered = filtered[filtered["Descrição ISIC Seção"] == sel_isic_sec]
    if sel_isic_div != "Todos":
        filtered = filtered[filtered["Descrição ISIC Divisão"] == sel_isic_div]

    # Seleção de Colunas para Exibição
    cols_display = [
        "Código SH6", "Descrição SH6", "Descrição CUCI Grupo", "Valor BR FOB_fim",
        "Share_BR_ini", "Share_BR_fim", "Var_Share_BR", "CAGR_World", "CAGR_BR", "RCA_fim", "Quadrante"
    ]
    
    df_view = filtered[cols_display].copy()
    
    # Formatação para Apresentação
    st.dataframe(
        df_view.style.format({
            "Valor BR FOB_fim": "US$ {:,.2f}",
            "Share_BR_ini": "{:.2%}",
            "Share_BR_fim": "{:.2%}",
            "Var_Share_BR": "{:+.2%}",
            "CAGR_World": "{:.2%}",
            "CAGR_BR": "{:.2%}",
            "RCA_fim": "{:.2f}"
        }),
        use_container_width=True,
        height=600
    )

    # Botão de Download Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_view.to_excel(writer, index=False, sheet_name='Analise_SH6')
    
    st.download_button(
        label="📥 Baixar Tabela Filtrada em Excel",
        data=output.getvalue(),
        file_name="analise_diversificacao_sh6.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# --- PÁGINA 3: VISÃO AGREGADA CUCI GRUPO & DRILL-DOWN ---
def page_cuci():
    st.title("📦 Agrupamento por CUCI Grupo & Drill-down")

    # Agrupamento por CUCI Grupo
    cuci_summary = analytics_df.groupby(["Código CUCI Grupo", "Descrição CUCI Grupo"]).agg(
        Total_BR_Fob=("Valor BR FOB_fim", "sum"),
        Total_World_Fob=("Valor World FOB_fim", "sum"),
        Qtd_SH6=("Código SH6", "count"),
        CAGR_World_Medio=("CAGR_World", "mean"),
        Var_Share_Medio=("Var_Share_BR", "mean")
    ).reset_index()

    cuci_summary["Share_BR_Grupo"] = cuci_summary["Total_BR_Fob"] / cuci_summary["Total_World_Fob"]

    st.subheader("Visão Geral por Grupo CUCI")
    st.dataframe(
        cuci_summary.style.format({
            "Total_BR_Fob": "US$ {:,.2f}",
            "Total_World_Fob": "US$ {:,.2f}",
            "Share_BR_Grupo": "{:.2%}",
            "CAGR_World_Medio": "{:.2%}",
            "Var_Share_Medio": "{:+.2%}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("🔍 Drill-down: Detalhar um Grupo CUCI")

    selected_cuci_code = st.selectbox(
        "Selecione o Grupo CUCI para ver os produtos SH6 e ISICs correlacionados:",
        options=cuci_summary["Código CUCI Grupo"] + " - " + cuci_summary["Descrição CUCI Grupo"]
    )

    if selected_cuci_code:
        code_only = selected_cuci_code.split(" - ")[0]
        sub_sh6 = analytics_df[analytics_df["Código CUCI Grupo"] == code_only]

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(f"**Produtos SH6 no Grupo {selected_cuci_code}**")
            st.dataframe(
                sub_sh6[[
                    "Código SH6", "Descrição SH6", "Descrição ISIC Divisão",
                    "Valor BR FOB_fim", "CAGR_BR", "RCA_fim", "Quadrante"
                ]].style.format({
                    "Valor BR FOB_fim": "US$ {:,.2f}",
                    "CAGR_BR": "{:.2%}",
                    "RCA_fim": "{:.2f}"
                }),
                use_container_width=True, hide_index=True
            )

        with col_right:
            st.markdown("**Distribuição por ISIC Divisão**")
            fig_pie = px.pie(
                sub_sh6,
                names="Descrição ISIC Divisão",
                values="Valor BR FOB_fim",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)


# --- PÁGINA 4: ESTADO EXPORTADOR (COMEX STAT) ---
def page_uf():
    st.title("🗺️ Análise por Estado Exportador (Comex Stat)")

    if df_uf_raw is None or df_uf_raw.empty:
        st.warning("⚠️ Nenhum dado de exportação estadual foi carregado.")
        return

    st.caption("Cruzamento das exportações estaduais com indicadores do UN Comtrade")

    # Mapeamento do arquivo enviado
    uf_df = df_uf_raw.copy()
    uf_df["Código SH6"] = uf_df["Código SH6"].astype(str).str.zfill(6)

    # Filtro por Estado
    ufs_disponiveis = ["Todos"] + sorted(uf_df["UF"].unique().tolist())
    sel_uf = st.selectbox("Selecione a Unidade da Federação (UF):", ufs_disponiveis)

    if sel_uf != "Todos":
        uf_df = uf_df[uf_df["UF"] == sel_uf]

    # Agrupa por UF e SH6 para consolidação do último ano disponível
    ano_max_uf = uf_df["Ano"].max()
    uf_grouped = uf_df[uf_df["Ano"] == ano_max_uf].groupby(["UF", "Código SH6"])["Valor US$ FOB"].sum().reset_index()

    # Mescla com os dados do UN Comtrade tratados
    merged_uf = pd.merge(
        uf_grouped,
        analytics_df[["Código SH6", "Descrição SH6", "Descrição CUCI Grupo", "Valor World FOB_fim", "CAGR_World", "RCA_fim", "Quadrante"]],
        on="Código SH6",
        how="inner"
    )

    # Cálculo do RCA Regional (Vantagem Comparativa da UF)
    tot_uf = merged_uf["Valor US$ FOB"].sum()
    tot_br = analytics_df["Valor BR FOB_fim"].sum()
    
    merged_uf["RCA_Regional"] = (merged_uf["Valor US$ FOB"] / tot_uf) / (analytics_df.set_index("Código SH6").loc[merged_uf["Código SH6"]]["Valor BR FOB_fim"].values / tot_br)

    # Métricas da UF
    col_u1, col_u2, col_u3 = st.columns(3)
    col_u1.metric(f"Exportações Totais ({sel_uf})", f"US$ {tot_uf:,.2f}")
    col_u2.metric("Produtos SH6 Exportados", len(merged_uf))
    col_u3.metric("Oportunidades Globais Cobertas", len(merged_uf[merged_uf["Quadrante"] == "Ganho de Espaço & Mercado Cresce (Oportunidade)"]))

    st.markdown("---")
    
    col_chart, col_rank = st.columns([1, 1])

    with col_chart:
        st.subheader("Maiores Pautador da UF por Valor FOB")
        fig_bar = px.bar(
            merged_uf.sort_values(by="Valor US$ FOB", ascending=False).head(10),
            x="Valor US$ FOB",
            y="Descrição SH6",
            orientation="h",
            color="Quadrante",
            height=450
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_rank:
        st.subheader("Especialização Regional (RCA Estaduais)")
        st.dataframe(
            merged_uf.sort_values(by="RCA_Regional", ascending=False)[
                ["Código SH6", "Descrição SH6", "Valor US$ FOB", "RCA_Regional", "Quadrante"]
            ].style.format({
                "Valor US$ FOB": "US$ {:,.2f}",
                "RCA_Regional": "{:.2f}"
            }),
            use_container_width=True, hide_index=True, height=450
        )


# ==============================================================================
# 5. ROTEAMENTO DAS PÁGINAS (ST.NAVIGATION)
# ==============================================================================
pg = st.navigation([
    st.Page(page_dashboard, title="Dashboard Geral", icon="📊"),
    st.Page(page_sh6, title="Detalhamento SH6", icon="📋"),
    st.Page(page_cuci, title="Visão CUCI Grupo", icon="📦"),
    st.Page(page_uf, title="Exportação por UF", icon="🗺️"),
])

pg.run()
