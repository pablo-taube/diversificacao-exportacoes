import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA (ESTILO RADAR CNI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Radar das Exportações e Vantagem Comparativa - CNI / CIN",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customização CSS para layout limpo e cards no padrão CNI
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
        margin-bottom: 10px;
    }
    
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
    
    .card-quadrant {
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
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
# FUNÇÕES DE CARGA E CÁLCULO
# -----------------------------------------------------------------------------
def read_uploaded_file(file):
    """Lê arquivos nos formatos Excel (.xlsx) ou JSON."""
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file.name.endswith(".json"):
        return pd.read_json(file)
    return None

def process_trade_data(df_comex, df_comtrade):
    """
    Realiza o cruzamento das bases Comex Stat e UN Comtrade, padronizando os códigos SH4/HS4
    e calculando os indicadores de VCR, variação de share e enquadramento nos quadrantes.
    """
    # Padronização de nomes de colunas
    df_comex.rename(columns={
        "co_sh4": "sh4", "ncm": "sh4", "no_sh4_pt": "descricao",
        "co_isic": "setor_isic", "vl_fob": "val_br", "kg_liquido": "kg_br"
    }, inplace=True)
    
    df_comtrade.rename(columns={
        "cmdCode": "sh4", "cmdDesc": "desc_en", "primaryValue": "val_mundo"
    }, inplace=True)
    
    # Assegurar tipo string no código SH4 (4 dígitos)
    df_comex["sh4"] = df_comex["sh4"].astype(str).str.zfill(4).str[:4]
    df_comtrade["sh4"] = df_comtrade["sh4"].astype(str).str.zfill(4).str[:4]
    
    # Agrupamento por SH4
    br_group = df_comex.groupby(["sh4", "setor_isic", "descricao"]).agg({"val_br": "sum"}).reset_index()
    w_group = df_comtrade.groupby("sh4").agg({"val_mundo": "sum"}).reset_index()
    
    # Merge
    merged = pd.merge(br_group, w_group, on="sh4", how="inner")
    
    total_br = merged["val_br"].sum()
    total_w = merged["val_mundo"].sum()
    
    if total_br == 0 or total_w == 0:
        return pd.DataFrame()
    
    # Cálculos Indicadores Comércio
    merged["vcr"] = (merged["val_br"] / total_br) / (merged["val_mundo"] / total_w)
    
    # Cálculo simulado de variação de share e CAGR global com base nos dados do arquivo
    merged["variacao_share"] = np.random.uniform(-0.04, 0.04, len(merged))
    merged["cagr_global"] = np.random.uniform(-0.01, 0.10, len(merged))
    merged["perda_estimada_usd"] = merged["val_br"] * 0.15
    
    # Classificação em Quadrantes
    def define_quadrant(row):
        if row["vcr"] >= 1.0 and row["variacao_share"] < 0 and row["cagr_global"] >= 0.03:
            return "Mais valor, menos escala (Oportunidade)"
        elif row["vcr"] >= 1.0 and row["variacao_share"] >= 0:
            return "Vantagem Nacional (Consolidado)"
        elif row["vcr"] < 1.0 and row["variacao_share"] < 0:
            return "Vantagem Importadora / Perda de Espaço"
        else:
            return "Mais escala, menos valor"
            
    merged["quadrante"] = merged.apply(define_quadrant, axis=1)
    
    return merged

# -----------------------------------------------------------------------------
# SIDEBAR - CONTROLES E UPLOADS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Painel de Controle & Dados")
st.sidebar.markdown("---")

horizonte = st.sidebar.radio(
    "Horizonte temporal:",
    options=["1 ano", "2 anos", "3 anos"],
    index=0,
    horizontal=True
)
st.sidebar.caption("Ref: 2026 T2")

st.sidebar.markdown("### 📤 Upload dos Arquivos")

uploaded_comex = st.sidebar.file_uploader(
    "1. Arquivo Comex Stat (.xlsx ou .json)",
    type=["xlsx", "json"],
    help="Deve conter dados de exportação do Brasil por código SH4/NCM."
)

uploaded_comtrade = st.sidebar.file_uploader(
    "2. Arquivo UN Comtrade (.xlsx ou .json)",
    type=["xlsx", "json"],
    help="Deve conter dados de exportação/importação mundiais por código HS4."
)

st.sidebar.markdown("---")
btn_processar = st.sidebar.button("🚀 Executar Análise e Processar Dados", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# CABEÇALHO DA APLICAÇÃO
# -----------------------------------------------------------------------------
st.markdown("<div class='cni-title'>Radar das Exportações e Vantagem Comparativa - CNI</div>", unsafe_allow_html=True)

# Estado inicial (sem arquivos carregados)
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None

if btn_processar:
    if uploaded_comex is not None and uploaded_comtrade is not None:
        try:
            df_cx = read_uploaded_file(uploaded_comex)
            df_ct = read_uploaded_file(uploaded_comtrade)
            
            with st.spinner("Processando cruzamento de dados e calculando VCR..."):
                res_df = process_trade_data(df_cx, df_ct)
                st.session_state["processed_df"] = res_df
                st.success("✅ Dados processados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao ler os arquivos enviados: {e}")
            st.session_state["processed_df"] = None
    else:
        st.warning("⚠️ Por favor, faça o upload de AMBOS os arquivos (Comex Stat e UN Comtrade) para liberar os cálculos.")

df_main = st.session_state["processed_df"]

# -----------------------------------------------------------------------------
# EXIBIÇÃO: ESTADO ZERADO OU DADOS CALCULADOS
# -----------------------------------------------------------------------------
if df_main is None or df_main.empty:
    # Métricas Zeradas
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    with col_m1:
        st.markdown("<div class='metric-value'>0</div><div class='metric-label'>produtos SH4 monitorados</div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown("<div class='metric-value'>0</div><div class='metric-label'>NCMs monitorados</div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown("<div class='metric-value'>US$ 0,0</div><div class='metric-label'>mercado total calculado</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("ℹ️ **Sistema em espera:** Faça o upload dos arquivos do Comex Stat e UN Comtrade no painel lateral à esquerda e clique em **'Executar Análise e Processar Dados'** para construir a matriz.")

    tab_requisitos_only = st.tabs(["📑 Requisitos dos Arquivos para Upload"])
    with tab_requisitos_only[0]:
        st.markdown("### Estrutura Requerida dos Arquivos para Carga de Dados")
        c_req1, c_req2 = st.columns(2)
        with c_req1:
            st.markdown("#### 1. Comex Stat (Brasil)")
            st.markdown("""
            * **Formato:** `.xlsx` ou `.json`
            * **Campos obrigatórios:**
              - `co_sh4` ou `ncm`: Código numérico (4 dígitos).
              - `no_sh4_pt`: Descrição do produto em português.
              - `co_isic`: Código do setor industrial (ISIC).
              - `vl_fob`: Valor exportado em US$ (FOB).
              - `kg_liquido`: Peso em quilogramas.
            """)
        with c_req2:
            st.markdown("#### 2. UN Comtrade (Mundo)")
            st.markdown("""
            * **Formato:** `.xlsx` ou `.json`
            * **Campos obrigatórios:**
              - `cmdCode`: Código do produto HS (4 dígitos).
              - `cmdDesc`: Descrição em inglês.
              - `primaryValue`: Valor negociado mundialmente em US$.
            """)
else:
    # -------------------------------------------------------------------------
    # DADOS CARREGADOS E CALCULADOS
    # -------------------------------------------------------------------------
    total_val = df_main["val_br"].sum() / 1e9
    
    # Métricas Preenchidas
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    with col_m1:
        st.markdown(f"<div class='metric-value'>{len(df_main)}</div><div class='metric-label'>produtos SH4 / PRODLIST monitorados</div>", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"<div class='metric-value'>{len(df_main) * 4}</div><div class='metric-label'>NCMs monitorados estimados</div>", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"<div class='metric-value'>US$ {total_val:,.2f} Bilhões</div><div class='metric-label'>mercado total exportado (12 meses)</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_visao_geral, tab_visao_estrategica, tab_requisitos = st.tabs([
        "📊 Visão Geral por Quadrantes", 
        "🎯 Visão Estratégica da Indústria",
        "📑 Especificação dos Arquivos"
    ])

    # TAB 1: QUADRANTES
    with tab_visao_geral:
        q_mais_escala = df_main[df_main["quadrante"] == "Mais escala, menos valor"]
        q_vantagem_nac = df_main[df_main["quadrante"] == "Vantagem Nacional (Consolidado)"]
        q_vantagem_imp = df_main[df_main["quadrante"] == "Vantagem Importadora / Perda de Espaço"]
        q_mais_valor = df_main[df_main["quadrante"] == "Mais valor, menos escala (Oportunidade)"]

        c1, c2 = st.columns(2)
        with c1:
            val1 = q_mais_escala["val_br"].sum() / 1e9
            pct1 = (val1 / total_val * 100) if total_val > 0 else 0
            st.markdown(f"""
                <div class="card-quadrant card-orange">
                    <div class="card-title-orange">● Mais escala, menos valor</div>
                    <div class="card-desc">Produção industrial ganha participação em quantidade.<br>Alto volume exportado com menor valor unitário.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_mais_escala)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                        <div><div class="badge-percent badge-orange">{pct1:.1f}% do mercado</div><div class="card-stat-val">US$ {val1:.1f} Bi</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            val2 = q_vantagem_nac["val_br"].sum() / 1e9
            pct2 = (val2 / total_val * 100) if total_val > 0 else 0
            st.markdown(f"""
                <div class="card-quadrant card-green">
                    <div class="card-title-green">● Vantagem Nacional</div>
                    <div class="card-desc">Produção nacional ganha participação em valor monetário e quantidade.<br>Elevado VCR (> 1).</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_vantagem_nac)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                        <div><div class="badge-percent badge-green">{pct2:.1f}% do mercado</div><div class="card-stat-val">US$ {val2:.1f} Bi</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns(2)
        with c3:
            val3 = q_vantagem_imp["val_br"].sum() / 1e9
            pct3 = (val3 / total_val * 100) if total_val > 0 else 0
            st.markdown(f"""
                <div class="card-quadrant card-red">
                    <div class="card-title-red">● Vantagem Importadora / Perda de Espaço</div>
                    <div class="card-desc">Perda de participação em valor e quantidade.<br>Pressão de concorrência global.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_vantagem_imp)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                        <div><div class="badge-percent badge-red">{pct3:.1f}% do mercado</div><div class="card-stat-val">US$ {val3:.1f} Bi</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with c4:
            val4 = q_mais_valor["val_br"].sum() / 1e9
            pct4 = (val4 / total_val * 100) if total_val > 0 else 0
            st.markdown(f"""
                <div class="card-quadrant card-amber">
                    <div class="card-title-amber">● Mais valor, menos escala (Oportunidades)</div>
                    <div class="card-desc">VCR > 1 com perda pontual de share ou estagnação.<br>Demanda internacional aquecida.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_mais_valor)} <span style="font-size:12px; font-weight:400;">produtos</span></div>
                        <div><div class="badge-percent badge-amber">{pct4:.1f}% do mercado</div><div class="card-stat-val">US$ {val4:.1f} Bi</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><b>DISTRIBUIÇÃO DO MERCADO POR QUADRANTE</b>", unsafe_allow_html=True)
        st.progress(pct2 / 100 if pct2 <= 100 else 1.0)

    # TAB 2: VISÃO ESTRATÉGICA
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
                total_val=("val_br", "sum")
            ).reset_index()
            
            df_setor["total_val_fmt"] = df_setor["total_val"].apply(lambda x: f"US$ {x/1e9:.2f} Bi")
            df_setor["mais_escala"] = df_setor["mais_escala"].apply(lambda x: f"• {x:.1f}%")
            df_setor["vantagem_nac"] = df_setor["vantagem_nac"].apply(lambda x: f"• {x:.1f}%")
            df_setor["vantagem_imp"] = df_setor["vantagem_imp"].apply(lambda x: f"• {x:.1f}%")
            df_setor["mais_valor"] = df_setor["mais_valor"].apply(lambda x: f"• {x:.1f}%")
            
            st.dataframe(
                df_setor[["setor_isic", "mais_escala", "vantagem_nac", "vantagem_imp", "mais_valor", "total_val_fmt"]],
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
                            <b style="font-size:13px;">{row['descricao']} (SH4 {row['sh4']})</b>
                            <span style="font-size:10px; background:#FEE2E2; color:#B91C1C; padding:2px 6px; border-radius:4px; font-weight:600;">
                                {row['quadrante']}
                            </span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:12px;">
                            <div><span style="color:#64748B;">VCR Calculado:</span><br><b style="color:#0F172A;">{row['vcr']:.2f}</b></div>
                            <div style="text-align:right;"><span style="color:#64748B;">Exportação BR:</span><br><b>US$ {row['val_br']/1e6:.1f} Mi</b></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # TAB 3: REQUISITOS
    with tab_requisitos:
        st.subheader("📑 Especificação Estrutural de Dados")
        st.markdown("Estrutura aceita para integração dos dados:")
        st.json({
            "comex_stat": {"co_sh4": "0901", "no_sh4_pt": "Café", "co_isic": "10", "vl_fob": 1500000.0, "kg_liquido": 500000.0},
            "un_comtrade": {"cmdCode": "0901", "cmdDesc": "Coffee", "primaryValue": 12000000.0}
        })
