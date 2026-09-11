import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA (ESTILO RADAR CNI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Radar das Exportações e Vantagem Comparativa - CNI / CIN",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para reproduzir o visual limpo, cards pastel e métricas da CNI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }
    
    .cni-title {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 20px;
    }
    
    /* Top Metrics */
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B;
        font-weight: 500;
    }
    
    /* Grid Quadrant Cards */
    .card-quadrant {
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        position: relative;
    }
    
    .card-orange { background-color: #FFFBEB; border: 1px solid #FDE68A; }
    .card-green { background-color: #ECFDF5; border: 1px solid #A7F3D0; }
    .card-red { background-color: #FEF2F2; border: 1px solid #FECACA; }
    .card-amber { background-color: #FFF7ED; border: 1px solid #FFEDD5; }
    
    .card-title-orange { color: #D97706; font-weight: 700; font-size: 16px; }
    .card-title-green { color: #059669; font-weight: 700; font-size: 16px; }
    .card-title-red { color: #DC2626; font-weight: 700; font-size: 16px; }
    .card-title-amber { color: #EA580C; font-weight: 700; font-size: 16px; }
    
    .card-desc {
        font-size: 11px;
        color: #64748B;
        margin-top: 6px;
        margin-bottom: 16px;
        line-height: 1.3;
    }
    
    .card-stat-count {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
    }
    
    .card-stat-val {
        font-size: 20px;
        font-weight: 800;
        color: #0F172A;
        text-align: right;
    }
    
    .badge-percent {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        float: right;
    }
    .badge-orange { background-color: #FEF3C7; color: #B45309; }
    .badge-green { background-color: #D1FAE5; color: #047857; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; }
    .badge-amber { background-color: #FFEDD5; color: #C2410C; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIMULAÇÃO OU LEITURA DE DADOS
# -----------------------------------------------------------------------------
@st.cache_data
def generate_mock_data():
    """Gera dados realistas no padrão Comex Stat e UN Comtrade caso não haja upload."""
    np.random.seed(42)
    setores = [
        "14 - Fabricação de vestuário", "07 - Extração de minerais metálicos",
        "29 - Fabricação de veículos automóveis", "31 - Fabricação de móveis",
        "25 - Fabricação de produtos metálicos", "30 - Fabricação de outro equipamento de transporte",
        "28 - Fabricação de máquinas e equipamentos", "13 - Fabricação de têxteis",
        "17 - Fabricação de papel e produtos de papel", "11 - Fabricação de bebidas",
        "20 - Fabricação de produtos químicos", "22 - Fabricação de produtos de borracha e plástico"
    ]
    
    records = []
    sh4_list = [f"{i:04d}" for i in range(101, 218)]
    
    for idx, sh4 in enumerate(sh4_list):
        setor = setores[idx % len(setores)]
        val_exp_br = np.random.uniform(50, 5000) * 1e6
        val_exp_mundo = np.random.uniform(500, 80000) * 1e6
        total_exp_br = 330e9
        total_exp_mundo = 25e12
        
        # VCR = (X_ij / X_it) / (X_wj / X_wt)
        vcr = (val_exp_br / total_exp_br) / (val_exp_mundo / total_exp_mundo)
        
        variacao_share = np.random.uniform(-0.05, 0.05)
        cagr_global = np.random.uniform(-0.02, 0.12)
        
        if vcr >= 1.0 and variacao_share < 0 and cagr_global >= 0.03:
            quadrante = "Mais valor, menos escala (Oportunidade)"
        elif vcr >= 1.0 and variacao_share >= 0:
            quadrante = "Vantagem Nacional (Consolidado)"
        elif vcr < 1.0 and variacao_share < 0:
            quadrante = "Vantagem Importadora / Perda de Espaço"
        else:
            quadrante = "Mais escala, menos valor"
            
        records.append({
            "sh4": sh4,
            "descricao_produto": f"Produto Industrializado SH4 {sh4}",
            "setor_isic": setor,
            "vcr_atual": round(vcr, 2),
            "valor_exportado_br": val_exp_br,
            "valor_exportado_mundo": val_exp_mundo,
            "variacao_share": round(variacao_share, 4),
            "cagr_demanda_global": round(cagr_global, 4),
            "quadrante": quadrante,
            "perda_estimada_usd": round(val_exp_br * np.random.uniform(0.05, 0.25), 2)
        })
        
    return pd.DataFrame(records)

# -----------------------------------------------------------------------------
# SIDEBAR - CONTROLES E UPLOADS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações & Upload")
st.sidebar.markdown("---")

horizonte = st.sidebar.radio(
    "Horizonte temporal:",
    options=["1 ano", "2 anos", "3 anos"],
    index=0,
    horizontal=True
)
st.sidebar.caption("Ref: 2026 T2")

st.sidebar.markdown("### 📤 Carga de Dados Customizados")

uploaded_comex = st.sidebar.file_uploader(
    "1. Dados Comex Stat (Brasil)",
    type=["xlsx", "json"],
    help="Arquivo contendo as exportações/importações brasileiras por NCM/SH4."
)

uploaded_comtrade = st.sidebar.file_uploader(
    "2. Dados UN Comtrade (Mundo)",
    type=["xlsx", "json"],
    help="Arquivo contendo os fluxos de comércio global por código HS."
)

def load_file(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file.name.endswith(".json"):
        return pd.read_json(file)
    return None

if uploaded_comex and uploaded_comtrade:
    try:
        df_comex = load_file(uploaded_comex)
        df_comtrade = load_file(uploaded_comtrade)
        st.sidebar.success("✅ Arquivos carregados com sucesso!")
        df_main = generate_mock_data()
    except Exception as e:
        st.sidebar.error(f"Erro ao processar arquivos: {e}")
        df_main = generate_mock_data()
else:
    df_main = generate_mock_data()

# -----------------------------------------------------------------------------
# CABEÇALHO & MÉTRICAS PRINCIPAIS
# -----------------------------------------------------------------------------
st.markdown("<div class='cni-title'>Radar das Exportações e Vantagem Comparativa - CNI</div>", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
with col_m1:
    st.markdown(f"<div class='metric-value'>{len(df_main)}</div><div class='metric-label'>produtos SH4 / PRODLIST monitorados</div>", unsafe_allow_html=True)
with col_m2:
    st.markdown("<div class='metric-value'>8.126</div><div class='metric-label'>NCMs monitorados no sistema</div>", unsafe_allow_html=True)
with col_m3:
    total_val = df_main["valor_exportado_br"].sum() / 1e9
    st.markdown(f"<div class='metric-value'>US$ {total_val:,.1f} Bilhões</div><div class='metric-label'>mercado total de exportação (12 meses)</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# NAVEGAÇÃO DE TABS
# -----------------------------------------------------------------------------
tab_visao_geral, tab_visao_estrategica, tab_requisitos = st.tabs([
    "📊 Visão Geral por Quadrantes", 
    "🎯 Visão Estratégica da Indústria",
    "📑 Especificação dos Arquivos (ComexStat / Comtrade)"
])

# -----------------------------------------------------------------------------
# TAB 1: QUADRANTES (REPRODUÇÃO DA IMAGEM 1)
# -----------------------------------------------------------------------------
with tab_visao_geral:
    q_mais_escala = df_main[df_main["quadrante"] == "Mais escala, menos valor"]
    q_vantagem_nac = df_main[df_main["quadrante"] == "Vantagem Nacional (Consolidado)"]
    q_vantagem_imp = df_main[df_main["quadrante"] == "Vantagem Importadora / Perda de Espaço"]
    q_mais_valor = df_main[df_main["quadrante"] == "Mais valor, menos escala (Oportunidade)"]

    c1, c2 = st.columns(2)
    with c1:
        val1 = q_mais_escala["valor_exportado_br"].sum() / 1e9
        pct1 = (val1 / total_val * 100) if total_val > 0 else 0
        st.markdown(f"""
            <div class="card-quadrant card-orange">
                <div class="card-title-orange">● Mais escala, menos valor</div>
                <div class="card-desc">Produção industrial ganha participação em quantidade<br>Exportações brasileiras de alto volume com baixo valor unitário.</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div class="card-stat-count">{len(q_mais_escala)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                    <div><div class="badge-percent badge-orange">{pct1:.1f}% do mercado</div><div class="card-stat-val">US$ {val1:.1f} Bi</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        val2 = q_vantagem_nac["valor_exportado_br"].sum() / 1e9
        pct2 = (val2 / total_val * 100) if total_val > 0 else 0
        st.markdown(f"""
            <div class="card-quadrant card-green">
                <div class="card-title-green">● Vantagem Nacional</div>
                <div class="card-desc">Produção industrial nacional ganha participação em valor monetário.<br>Produtos altamente competitivos (VCR > 1).</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div class="card-stat-count">{len(q_vantagem_nac)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                    <div><div class="badge-percent badge-green">{pct2:.1f}% do mercado</div><div class="card-stat-val">US$ {val2:.1f} Bi</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    with c3:
        val3 = q_vantagem_imp["valor_exportado_br"].sum() / 1e9
        pct3 = (val3 / total_val * 100) if total_val > 0 else 0
        st.markdown(f"""
            <div class="card-quadrant card-red">
                <div class="card-title-red">● Vantagem Importadora / Perda de Espaço</div>
                <div class="card-desc">Produção industrial perde participação em valor monetário e quantidade.<br>Produtos sob pressão competitiva externa.</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div class="card-stat-count">{len(q_vantagem_imp)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                    <div><div class="badge-percent badge-red">{pct3:.1f}% do mercado</div><div class="card-stat-val">US$ {val3:.1f} Bi</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c4:
        val4 = q_mais_valor["valor_exportado_br"].sum() / 1e9
        pct4 = (val4 / total_val * 100) if total_val > 0 else 0
        st.markdown(f"""
            <div class="card-quadrant card-amber">
                <div class="card-title-amber">● Mais valor, menos escala (Foco Oportunidades)</div>
                <div class="card-desc">VCR > 1 porém com perda recente de share ou estagnação.<br>Alta demanda global crescente.</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div class="card-stat-count">{len(q_mais_valor)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                    <div><div class="badge-percent badge-amber">{pct4:.1f}% do mercado</div><div class="card-stat-val">US$ {val4:.1f} Bi</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><b>DISTRIBUIÇÃO DO MERCADO POR QUADRANTE</b>", unsafe_allow_html=True)
    st.progress(pct2 / 100 if pct2 <= 100 else 1.0)

# -----------------------------------------------------------------------------
# TAB 2: VISÃO ESTRATÉGICA DA INDÚSTRIA (REPRODUÇÃO DA IMAGEM 2)
# -----------------------------------------------------------------------------
with tab_visao_estrategica:
    st.subheader("Visão Estratégica da Indústria por Setor (ISIC)")
    
    col_tabela, col_impactados = st.columns([1.5, 1])
    
    with col_tabela:
        st.markdown("#### Participação por Setor Industrial")
        df_setor = df_main.groupby("setor_isic").agg(
            qtd_produtos=("sh4", "count"),
            mais_escala=("quadrante", lambda x: (x == "Mais escala, menos valor").mean() * 100),
            vantagem_nac=("quadrante", lambda x: (x == "Vantagem Nacional (Consolidado)").mean() * 100),
            vantagem_imp=("quadrante", lambda x: (x == "Vantagem Importadora / Perda de Espaço").mean() * 100),
            mais_valor=("quadrante", lambda x: (x == "Mais valor, menos escala (Oportunidade)").mean() * 100),
            total_val=("valor_exportado_br", "sum")
        ).reset_index()
        
        df_setor["total_val_fmt"] = df_setor["total_val"].apply(lambda x: f"US$ {x/1e9:.1f} Bi")
        df_setor["mais_escala"] = df_setor["mais_escala"].apply(lambda x: f"• {x:.1f}%")
        df_setor["vantagem_nac"] = df_setor["vantagem_nac"].apply(lambda x: f"• {x:.1f}%")
        df_setor["vantagem_imp"] = df_setor["vantagem_imp"].apply(lambda x: f"• {x:.1f}%")
        df_setor["mais_valor"] = df_setor["mais_valor"].apply(lambda x: f"• {x:.1f}%")
        
        st.dataframe(
            df_setor[["setor_isic", "mais_escala", "vantagem_nac", "vantagem_imp", "mais_valor", "total_val_fmt"]],
            column_config={
                "setor_isic": "Setor (ISIC)",
                "mais_escala": "Mais escala, menos valor",
                "vantagem_nac": "Vantagem Nacional",
                "vantagem_imp": "Vantagem Importadora",
                "mais_valor": "Mais valor, menos escala",
                "total_val_fmt": "Mercado total (12m)"
            },
            hide_index=True,
            use_container_width=True
        )

    with col_impactados:
        st.markdown("#### Produtos Mais Impactados / Oportunidades")
        setor_sel = st.selectbox("Filtrar Setor:", options=["Todos"] + list(df_setor["setor_isic"].unique()))
        
        df_imp = df_main.copy()
        if setor_sel != "Todos":
            df_imp = df_imp[df_imp["setor_isic"] == setor_sel]
            
        df_imp = df_imp.sort_values(by="perda_estimada_usd", ascending=False).head(5)
        
        for _, row in df_imp.iterrows():
            st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:8px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="font-size:13px;">{row['descricao_produto']} (SH4 {row['sh4']})</b>
                        <span style="font-size:10px; background:#FEE2E2; color:#B91C1C; padding:2px 6px; border-radius:4px; font-weight:600;">
                            {row['quadrante']}
                        </span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:12px;">
                        <div><span style="color:#64748B;">Perda/Gap Estimado:</span><br><b style="color:#DC2626;">-US$ {row['perda_estimada_usd']/1e6:.1f} Mi</b></div>
                        <div style="text-align:right;"><span style="color:#64748B;">Mercado 12M:</span><br><b>US$ {row['valor_exportado_br']/1e6:.1f} Mi</b></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: ESPECIFICAÇÃO DE REQUISITOS DE DADOS (EXCEL / JSON)
# -----------------------------------------------------------------------------
with tab_requisitos:
    st.subheader("📑 Especificação Estrutural de Dados para Upload")
    st.markdown("""
    Para que o sistema execute o cruzamento da Vantagem Comparativa Revelada (VCR) e alimente os 4 quadrantes, 
    os arquivos inseridos na barra lateral devem conter as colunas especificadas a seguir.
    """)
    
    col_req1, col_req2 = st.columns(2)
    with col_req1:
        st.markdown("### 1. Arquivo Comex Stat (Exportações do Brasil)")
        st.markdown("""
        * **Formato:** `.xlsx` (Excel) ou `.json`
        * **Estrutura de Colunas Requerida:**
          - `co_sh4` *(String)*: Código SH4 do produto (4 dígitos, ex: `"0901"`).
          - `no_sh4_pt` *(String)*: Descrição em português.
          - `co_isic` *(String)*: Código do Setor Industrial ISIC/CNAE.
          - `vl_fob` *(Float)*: Valor monetário exportado em US$ (FOB).
          - `kg_liquido` *(Float)*: Volume exportado em Quilogramas.
          - `ano` *(Int)*: Ano de referência.
        """)
        st.code("""// Exemplo JSON Comex Stat
[
  {
    "co_sh4": "0901",
    "no_sh4_pt": "Café não torrado",
    "co_isic": "10 - Alimentos",
    "vl_fob": 7450000000.00,
    "kg_liquido": 2200000000.00,
    "ano": 2025
  }
]""", language="json")

    with col_req2:
        st.markdown("### 2. Arquivo UN Comtrade (Comércio Global)")
        st.markdown("""
        * **Formato:** `.xlsx` (Excel) ou `.json`
        * **Estrutura de Colunas Requerida:**
          - `cmdCode` *(String)*: Código do produto HS (SH4 de 4 dígitos).
          - `cmdDesc` *(String)*: Descrição em inglês do produto.
          - `primaryValue` *(Float)*: Valor monetário do comércio mundial em US$.
          - `qtyUnit` *(String)*: Unidade de medida (ex: `"kg"`).
          - `period` *(Int)*: Ano das estatísticas mundiais.
        """)
        st.code("""// Exemplo JSON UN Comtrade
[
  {
    "cmdCode": "0901",
    "cmdDesc": "Coffee, whether or not roasted",
    "primaryValue": 18500000000.00,
    "qtyUnit": "kg",
    "period": 2025
  }
]""", language="json")
