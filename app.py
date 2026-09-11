import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Matriz de Oportunidades Comerciais - Brasil",
    page_icon="📈",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SIMULAÇÃO E PROCESSAMENTO DE DADOS (COMEXSTAT / UN COMTRADE)
# -----------------------------------------------------------------------------
@st.cache_data
def load_trade_data():
    """
    Gera/carrega base simulada com a estrutura exata do UN Comtrade / Comex Stat
    contendo 117 produtos (SH4) para demonstrar a metodologia de filtros.
    """
    np.random.seed(42)
    sh4_codes = [f"SH{1000 + i}" for i in range(117)]
    products = [f"Produto SH4 - {code}" for code in sh4_codes]
    
    data = []
    for code, name in zip(sh4_codes, products):
        vcr_2016 = np.random.uniform(1.1, 5.5)
        vcr_atual = vcr_2016 + np.random.uniform(-1.5, 1.2)
        
        # Fatia do Brasil no mercado global
        share_2016 = np.random.uniform(0.02, 0.25)
        # Força 37 produtos a terem queda ou estagnação de share (Filtro 1)
        if len(data) < 37:
            share_atual = share_2016 * np.random.uniform(0.6, 0.98)
        else:
            share_atual = share_2016 * np.random.uniform(1.05, 1.5)
            
        # Crescimento da demanda mundial (CAGR Importações Globais)
        cagr_demanda_global = np.random.uniform(-0.02, 0.12)
        
        data.append({
            "codigo_sh4": code,
            "produto": name,
            "vcr_2016": round(vcr_2016, 2),
            "vcr_atual": round(vcr_atual, 2),
            "share_br_2016": round(share_2016, 4),
            "share_br_atual": round(share_atual, 4),
            "variacao_share": round(share_atual - share_2016, 4),
            "cagr_demanda_global": round(cagr_demanda_global, 4)
        })
        
    df = pd.DataFrame(data)
    return df

@st.cache_data
def load_country_gaps():
    """
    Base fictícia cruzando países e gaps de mercado para os produtos elegíveis.
    """
    paises = ["China", "Estados Unidos", "Alemanha", "Índia", "Japão", "México", "Voo Vietnam"]
    barreiras = {
        "China": "Exigências SPS / Licenciamento Não-Tarifário",
        "Estados Unidos": "Tarifas Adicionais (Seção 232) / Quotas",
        "Alemanha": "Certificação ESG / Regulamento de Desmatamento da UE (EUDR)",
        "Índia": "Tarifas de Importação Elevadas (MFN)",
        "Japão": "Padrões Sanitários Estritos / Normas Técnicas",
        "México": "Acordo Preferencial Pendente / Concorrência Regional",
        "Vietnam": "Logística e Barreiras Tarifárias Out-of-Quota"
    }
    return paises, barreiras

# Carregar Dados
df_produtos = load_trade_data()
paises_lista, dict_barreiras = load_country_gaps()

# -----------------------------------------------------------------------------
# INTERFACE DO USUÁRIO & FILTROS METODOLÓGICOS
# -----------------------------------------------------------------------------
st.title("📊 Plano e Matriz de Oportunidades Comerciais")
st.caption("Análise de Diversificação das Exportações Brasileiras via UN Comtrade & Comex Stat")

st.sidebar.header("Parâmetros do Metodologia")

# Filtro 1: Oferta
st.sidebar.subheader("Filtro 1: Oferta")
min_vcr = st.sidebar.number_input("VCR Mínimo (Balassa)", value=1.0, step=0.1)

df_f1 = df_produtos[
    (df_produtos["vcr_atual"] >= min_vcr) & 
    (df_produtos["variacao_share"] <= 0)
]

# Filtro 2: Demanda Externa
st.sidebar.subheader("Filtro 2: Demanda Externa")
media_cagr_global = df_produtos["cagr_demanda_global"].mean()
cagr_corte = st.sidebar.slider(
    "Corte do CAGR da Demanda Global (%)",
    min_value=float(df_produtos["cagr_demanda_global"].min() * 100),
    max_value=float(df_produtos["cagr_demanda_global"].max() * 100),
    value=float(media_cagr_global * 100)
) / 100

df_f2 = df_f1[df_f1["cagr_demanda_global"] >= cagr_corte]

# KPI Overview
col1, col2, col3 = st.columns(3)
col1.metric("1. Universo VCR > 1", f"{len(df_produtos[df_produtos['vcr_atual'] >= min_vcr])} Produtos")
col2.metric("2. Filtro Oferta (Perda de Share)", f"{len(df_f1)} Produtos")
col3.metric("3. Filtro Oportunidade (Demanda em Alta)", f"{len(df_f2)} Produtos")

st.markdown("---")

# -----------------------------------------------------------------------------
# SEÇÕES DO RELATÓRIO
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "2.1 Matriz Cruzada (VCR vs. Crescimento)", 
    "2.2 Priorização de Mercados-Alvo", 
    "2.3 Barreiras & Acesso a Mercado"
])

# -----------------------------------------------------------------------------
# TAB 1: MATRIZ CRUZADA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Matriz VCR vs. Crescimento de Mercado Mundial")
    st.write(
        "Cruzamento dos produtos de alto potencial brasileiro ($VCR > 1$) com a taxa de crescimento da demanda global. "
        "Destaque para a zona de **alta prioridade** (quadrante superior direito)."
    )
    
    fig = px.scatter(
        df_f1,
        x="vcr_atual",
        y="cagr_demanda_global",
        color=df_f1["cagr_demanda_global"] >= cagr_corte,
        color_discrete_map={True: "#00CC96", False: "#EF553B"},
        hover_data=["codigo_sh4", "produto", "variacao_share"],
        labels={
            "vcr_atual": "Vantagem Comparativa Revelada (VCR Atual)",
            "cagr_demanda_global": "Crescimento Demanda Global (CAGR)",
            "color": "Oportunidade Prioritária"
        },
        title="Posicionamento Estratégico de Produtos Elegíveis"
    )
    fig.add_hline(y=cagr_corte, line_dash="dash", line_color="gray", annotation_text="Corte de Demanda Global")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: PRIORIZAÇÃO DE MERCADOS-ALVO
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Top Produtos de Maior Valor Estratégico e Gaps de Mercado")
    
    top_n = st.slider("Selecione o número de produtos para priorizar:", 5, 10, 5)
    df_top = df_f2.sort_values(by="cagr_demanda_global", ascending=False).head(top_n)
    
    st.write(f"**Top {top_n} Produtos Selecionados (Filtro 3 - Gaps de Mercado):**")
    st.dataframe(
        df_top[["codigo_sh4", "produto", "vcr_atual", "variacao_share", "cagr_demanda_global"]],
        use_container_width=True
    )
    
    selected_prod = st.selectbox("Selecione um produto para visualizar os mercados potenciais:", df_top["produto"].unique())
    
    # Simulação de Matriz de Destinos por Produto
    np.random.seed(abs(hash(selected_prod)) % 1000)
    dest_data = []
    for pais in paises_lista[:5]:
        import_crec = np.random.uniform(0.04, 0.18)
        share_br_local = np.random.uniform(0.001, 0.04)
        dest_data.append({
            "País Destino": pais,
            "Crescimento Importação País (%)": f"{import_crec*100:.2f}%",
            "Market Share Atual do Brasil": f"{share_br_local*100:.2f}%",
            "Potencial de Penetração": "Alto (Gap)" if share_br_local < 0.02 else "Médio"
        })
    
    st.table(pd.DataFrame(dest_data))

# -----------------------------------------------------------------------------
# TAB 3: BARREIRAS E ESTRATÉGIA DE ACESSO
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Identificação de Barreiras de Acesso ao Mercado")
    st.write("Diagnóstico preliminar dos entraves que provocaram a perda de espaço das exportações brasileiras:")
    
    col_prod, col_pais = st.columns(2)
    with col_prod:
        prod_barreira = st.selectbox("Selecione o Produto:", df_f2["produto"].unique(), key="barr_prod")
    with col_pais:
        pais_barreira = st.selectbox("Selecione o Mercado-Alvo:", paises_lista, key="barr_pais")
        
    st.markdown("---")
    
    st.markdown(f"#### Diagnóstico para **{prod_barreira}** em **{pais_barreira}**")
    
    barreira_identificada = dict_barreiras.get(pais_barreira, "Exigências Tarifárias e Regulatórias Padrão")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.error(f"**Principal Barreira Mapeada:**\n\n{barreira_identificada}")
    with col_b:
        st.success(
            "**Recomendação Estratégica:**\n\n"
            "- Aceleração de acordos de equivalência de normas técnicas/sanitárias.\n"
            "- Ações de promoção comercial focadas através de missões da ApexBrasil.\n"
            "- Adaptação às exigências de rastreabilidade e sustentabilidade."
        )
