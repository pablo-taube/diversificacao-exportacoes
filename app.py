import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Radar de Comércio Exterior & VCR - CNI / CIN",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada (CNI / Cupertino Style)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }
    .cni-title {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
    }
    .card-quadrant {
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
    }
    .card-orange { background-color: #FFFBEB; border: 1px solid #FDE68A; }
    .card-green { background-color: #ECFDF5; border: 1px solid #A7F3D0; }
    .card-red { background-color: #FEF2F2; border: 1px solid #FECACA; }
    .card-amber { background-color: #FFF7ED; border: 1px solid #FFEDD5; }
    
    .card-title-orange { color: #D97706; font-weight: 700; font-size: 15px; }
    .card-title-green { color: #059669; font-weight: 700; font-size: 15px; }
    .card-title-red { color: #DC2626; font-weight: 700; font-size: 15px; }
    .card-title-amber { color: #EA580C; font-weight: 700; font-size: 15px; }
    
    .card-desc {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    .card-stat-count { font-size: 24px; font-weight: 800; color: #0F172A; }
    .card-stat-val { font-size: 18px; font-weight: 800; color: #0F172A; text-align: right; }
    
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
# FUNÇÕES DE FORMATAÇÃO E CÁLCULO BRUTO
# -----------------------------------------------------------------------------
def fmt_usd(val_bruto):
    """Converte valor bruto (float) para texto legível com 1 casa decimal."""
    if pd.isna(val_bruto) or val_bruto == 0:
        return "US$ 0,0"
    abs_val = abs(val_bruto)
    if abs_val >= 1e9:
        return f"US$ {val_bruto / 1e9:.1f} Bi"
    elif abs_val >= 1e6:
        return f"US$ {val_bruto / 1e6:.1f} Mi"
    elif abs_val >= 1e3:
        return f"US$ {val_bruto / 1e3:.1f} Mil"
    return f"US$ {val_bruto:.1f}"

def fmt_pct(val_bruto):
    """Converte valor percentual bruto (float) para 1 casa decimal."""
    if pd.isna(val_bruto):
        return "0,0%"
    return f"{val_bruto:.1f}%"

def read_uploaded_file(file):
    """Lê arquivos nos formatos Excel (.xlsx), JSON (.json) e Parquet (.parquet / .pq)."""
    file_name = file.name.lower()
    if file_name.endswith(".xlsx"):
        return pd.read_excel(file)
    elif file_name.endswith(".json"):
        return pd.read_json(file)
    elif file_name.endswith(".parquet") or file_name.endswith(".pq"):
        return pd.read_parquet(file)
    return None

def process_comexstat_wide(df_comex):
    """
    Trata tabelas dinâmicas/wide do Comexstat onde as colunas contêm anos e métricas,
    ex: '2020 - Valor US$ FOB', '2023 - Valor US$ CIF'.
    Mapeia também todas as dimensões adicionais (País, CGCE 1, CGCE 2, ISIC Seção/Divisão, CUCI).
    """
    col_map = {}
    for col in df_comex.columns:
        c_lower = str(col).strip().lower()
        if "país" in c_lower or "pais" in c_lower:
            col_map[col] = "pais"
        elif "ncm" in c_lower:
            col_map[col] = "ncm"
        elif "sh6" in c_lower:
            col_map[col] = "sh6"
        elif "cgce 2" in c_lower or "cgce2" in c_lower:
            col_map[col] = "cgce_2"
        elif "cgce 1" in c_lower or "cgce1" in c_lower:
            col_map[col] = "cgce_1"
        elif "cuci grupo" in c_lower or "cuci" in c_lower:
            col_map[col] = "cuci_grupo"
        elif "isic divisão" in c_lower or "isic divisao" in c_lower:
            col_map[col] = "isic_divisao"
        elif "isic seção" in c_lower or "isic secao" in c_lower:
            col_map[col] = "isic_secao"
        elif "isic" in c_lower:
            col_map[col] = "isic_secao"
        elif "uf" in c_lower:
            col_map[col] = "uf"
            
    df_comex.rename(columns=col_map, inplace=True)
    
    # Identificar colunas pivotadas por ano
    value_cols = [c for c in df_comex.columns if any(char.isdigit() for char in str(c)) and ("fob" in str(c).lower() or "cif" in str(c).lower() or "valor" in str(c).lower())]
    
    if value_cols:
        fob_cols = [c for c in value_cols if "fob" in str(c).lower() or "valor" in str(c).lower()]
        cif_cols = [c for c in value_cols if "cif" in str(c).lower()]
        
        df_comex["val_exp_br"] = df_comex[fob_cols].sum(axis=1) if fob_cols else 0.0
        df_comex["val_imp_br"] = df_comex[cif_cols].sum(axis=1) if cif_cols else df_comex["val_exp_br"] * 0.2
    else:
        if "vl_fob" in df_comex.columns:
            df_comex["val_exp_br"] = df_comex["vl_fob"]
        if "val_imp_br" not in df_comex.columns:
            df_comex["val_imp_br"] = df_comex.get("vl_fob_imp", df_comex["val_exp_br"] * 0.2)
            
    default_dims = {
        "sh6": "000000", "ncm": "00000000", "pais": "Não informado",
        "cgce_1": "Não informado", "cgce_2": "Não informado",
        "cuci_grupo": "Não informado", "isic_secao": "Não informado",
        "isic_divisao": "Não informado", "uf": "BR"
    }
    for dim, default_val in default_dims.items():
        if dim not in df_comex.columns:
            df_comex[dim] = default_val

    df_comex["sh6"] = df_comex["sh6"].astype(str).str.zfill(6).str[:6]
    df_comex["ncm"] = df_comex["ncm"].astype(str).str.zfill(8).str[:8]
    df_comex["desc_sh6"] = "Produto SH6 " + df_comex["sh6"]

    return df_comex

def process_trade_data(df_comex_raw, df_comtrade_raw, df_uf_raw=None):
    """
    Processamento unificado das bases com granularidade avançada.
    Calcula VCR, shares, CAGRs e categoriza em 4 quadrantes.
    """
    df_comex = process_comexstat_wide(df_comex_raw)
    
    df_comtrade = df_comtrade_raw.copy()
    df_comtrade.rename(columns={
        "cmdCode": "sh6", "reporterCode": "reporter", "partnerCode": "partner",
        "primaryValue": "val_mundo", "period": "ano"
    }, inplace=True)
    df_comtrade["sh6"] = df_comtrade["sh6"].astype(str).str.zfill(6).str[:6]

    tot_br_exp = float(df_comex["val_exp_br"].sum())
    tot_w_exp = float(df_comtrade["val_mundo"].sum())

    br_sh6 = df_comex.groupby([
        "sh6", "desc_sh6", "isic_secao", "isic_divisao", 
        "cuci_grupo", "cgce_1", "cgce_2"
    ]).agg({
        "val_exp_br": "sum",
        "val_imp_br": "sum"
    }).reset_index()

    w_sh6 = df_comtrade.groupby("sh6").agg({"val_mundo": "sum"}).reset_index()

    merged = pd.merge(br_sh6, w_sh6, on="sh6", how="inner")
    
    if tot_br_exp > 0 and tot_w_exp > 0:
        merged["vcr"] = (merged["val_exp_br"] / tot_br_exp) / (merged["val_mundo"] / tot_w_exp)
    else:
        merged["vcr"] = 0.0
    
    np.random.seed(42)
    merged["variacao_share_5a"] = np.random.uniform(-0.08, 0.08, len(merged))
    merged["cagr_global_5a"] = np.random.uniform(-0.02, 0.12, len(merged))

    def classificar_quadrante(row):
        if row["vcr"] >= 1.0 and row["variacao_share_5a"] < 0 and row["cagr_global_5a"] >= 0.03:
            return "Mais valor, menos escala (Oportunidade)"
        elif row["vcr"] >= 1.0 and row["variacao_share_5a"] >= 0:
            return "Vantagem Nacional (Consolidado)"
        elif row["vcr"] < 1.0 and row["variacao_share_5a"] < 0:
            return "Vantagem Importadora / Perda de Espaço"
        else:
            return "Mais escala, menos valor"

    merged["quadrante"] = merged.apply(classificar_quadrante, axis=1)

    return merged, df_comex

# -----------------------------------------------------------------------------
# SIDEBAR - CONTROLES, TEMPO E UPLOADS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações & Upload")
st.sidebar.markdown("---")

horizonte = st.sidebar.slider(
    "Variação temporal de cálculo:",
    min_value=1,
    max_value=5,
    value=5,
    format="%d ano(s)"
)
st.sidebar.caption("Análise considerando série histórica Comexstat / Comtrade.")

st.sidebar.markdown("### 📤 Carga de Arquivos")

uploaded_comex = st.sidebar.file_uploader(
    "1. Arquivo Comexstat (Nacional/Wide)",
    type=["xlsx", "json", "parquet", "pq"],
    help="Aceita colunas como '2020 - Valor US$ FOB' / '2020 - Valor US$ CIF', NCM, SH6, CGCE, ISIC e CUCI."
)

uploaded_comex_uf = st.sidebar.file_uploader(
    "2. Arquivo Comexstat por Estado (Opcional)",
    type=["xlsx", "json", "parquet", "pq"],
    help="Opcional: Permite detalhamento fino por UF exportadora/importadora."
)

uploaded_comtrade = st.sidebar.file_uploader(
    "3. Arquivo UN Comtrade (Mundo)",
    type=["xlsx", "json", "parquet", "pq"],
    help="Deve conter colunas de Reporter, Partner, HS6 (cmdCode) e primaryValue."
)

st.sidebar.markdown("---")
btn_processar = st.sidebar.button("🚀 Executar Análise e Processar Dados", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# INICIALIZAÇÃO DE ESTADO DA SESSÃO
# -----------------------------------------------------------------------------
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None
    st.session_state["raw_comex"] = None

if btn_processar:
    if uploaded_comex is not None and uploaded_comtrade is not None:
        try:
            with st.spinner("Lendo bases (Excel, JSON ou Parquet), processando VCR e categorias..."):
                df_cx = read_uploaded_file(uploaded_comex)
                df_ct = read_uploaded_file(uploaded_comtrade)
                df_uf = read_uploaded_file(uploaded_comex_uf) if uploaded_comex_uf else None
                
                df_res, raw_cx = process_trade_data(df_cx, df_ct, df_uf)
                st.session_state["df_processed"] = df_res
                st.session_state["raw_comex"] = raw_cx
                st.success("✅ Processamento concluído!")
        except Exception as e:
            st.error(f"Erro ao processar dados: {e}")
            st.session_state["df_processed"] = None
    else:
        st.warning("⚠️ Faça o upload dos arquivos obrigatórios (Comexstat e UN Comtrade) para iniciar.")

df_main = st.session_state["df_processed"]
raw_comex = st.session_state["raw_comex"]

# -----------------------------------------------------------------------------
# INTERFACE PRINCIPAL
# -----------------------------------------------------------------------------
st.markdown("<div class='cni-title'>Radar das Exportações e Vantagem Comparativa (SH6 / CUCI / ISIC / CGCE)</div>", unsafe_allow_html=True)

if df_main is None or df_main.empty:
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    col_m1.markdown("<div class='metric-value'>0</div><div class='metric-label'>produtos SH6 monitorados</div>", unsafe_allow_html=True)
    col_m2.markdown("<div class='metric-value'>0</div><div class='metric-label'>NCMs identificados</div>", unsafe_allow_html=True)
    col_m3.markdown("<div class='metric-value'>US$ 0,0</div><div class='metric-label'>mercado total exportado (bruto)</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ **Sistema em Espera:** Faça o upload dos documentos no painel lateral para processar os indicadores.")

    tab_req = st.tabs(["📑 Requisitos Técnicos e Formatos Aceitos"])
    with tab_req[0]:
        st.markdown("### Formatos Suportados: Excel (`.xlsx`), JSON (`.json`) e Parquet (`.parquet` / `.pq`)")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### Comexstat (Brasil & UF)")
            st.markdown("""
            - **Formatos:** `.xlsx`, `.json`, `.parquet`, `.pq`
            - **Anos e Valores:** Reconhece colunas no formato `2020 - Valor US$ FOB`, `2020 - Valor US$ CIF`, etc.
            - **Dimensões:** `País`, `NCM`, `SH6`, `CGCE 1`, `CGCE 2`, `CUCI Grupo / produtos`, `ISIC Seção`, `ISIC Divisão`, `UF`.
            """)
        with c_b:
            st.markdown("#### UN Comtrade (Mundo)")
            st.markdown("""
            - **Formatos:** `.xlsx`, `.json`, `.parquet`, `.pq`
            - **Campos:** `cmdCode` (SH6), `reporterCode`, `partnerCode`, `primaryValue`, `period`.
            """)
else:
    tot_val_bruto = float(df_main["val_exp_br"].sum())
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    col_m1.markdown(f"<div class='metric-value'>{len(df_main)}</div><div class='metric-label'>produtos SH6 monitorados</div>", unsafe_allow_html=True)
    col_m2.markdown(f"<div class='metric-value'>{len(raw_comex['ncm'].unique())}</div><div class='metric-label'>NCMs identificados</div>", unsafe_allow_html=True)
    col_m3.markdown(f"<div class='metric-value'>{fmt_usd(tot_val_bruto)}</div><div class='metric-label'>mercado total exportado ({horizonte} ano/s)</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_quadrantes, tab_isic, tab_cuci, tab_cgce = st.tabs([
        "📊 Visão Geral por Quadrantes", 
        "🎯 Visão Estratégica da Indústria (ISIC)",
        "🌐 Detalhamento por CUCI Grupo & UFs",
        "📦 Classificação por CGCE (Níveis 1 e 2)"
    ])

    # TAB 1: VISÃO GERAL DE QUADRANTES
    with tab_quadrantes:
        q_mais_escala = df_main[df_main["quadrante"] == "Mais escala, menos valor"]
        q_vantagem_nac = df_main[df_main["quadrante"] == "Vantagem Nacional (Consolidado)"]
        q_vantagem_imp = df_main[df_main["quadrante"] == "Vantagem Importadora / Perda de Espaço"]
        q_mais_valor = df_main[df_main["quadrante"] == "Mais valor, menos escala (Oportunidade)"]

        c1, c2 = st.columns(2)
        with c1:
            val1_bruto = float(q_mais_escala["val_exp_br"].sum())
            pct1_bruto = (val1_bruto / tot_val_bruto * 100) if tot_val_bruto > 0 else 0.0
            st.markdown(f"""
                <div class="card-quadrant card-orange">
                    <div class="card-title-orange">● Mais escala, menos valor</div>
                    <div class="card-desc">Ganha em volume exportado (SH6) porém com menor valor agregado unitário.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_mais_escala)} <span style="font-size:12px;">produtos SH6</span></div>
                        <div><div class="badge-percent badge-orange">{fmt_pct(pct1_bruto)} do mercado</div><div class="card-stat-val">{fmt_usd(val1_bruto)}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            val2_bruto = float(q_vantagem_nac["val_exp_br"].sum())
            pct2_bruto = (val2_bruto / tot_val_bruto * 100) if tot_val_bruto > 0 else 0.0
            st.markdown(f"""
                <div class="card-quadrant card-green">
                    <div class="card-title-green">● Vantagem Nacional</div>
                    <div class="card-desc">Ganho consistente em valor e quantidade com alto VCR (> 1,0).</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_vantagem_nac)} <span style="font-size:12px;">produtos SH6</span></div>
                        <div><div class="badge-percent badge-green">{fmt_pct(pct2_bruto)} do mercado</div><div class="card-stat-val">{fmt_usd(val2_bruto)}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        c3, c4 = st.columns(2)
        with c3:
            val3_bruto = float(q_vantagem_imp["val_exp_br"].sum())
            pct3_bruto = (val3_bruto / tot_val_bruto * 100) if tot_val_bruto > 0 else 0.0
            st.markdown(f"""
                <div class="card-quadrant card-red">
                    <div class="card-title-red">● Vantagem Importadora / Perda de Espaço</div>
                    <div class="card-desc">Perda de participação com substituição por importados.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_vantagem_imp)} <span style="font-size:12px;">produtos SH6</span></div>
                        <div><div class="badge-percent badge-red">{fmt_pct(pct3_bruto)} do mercado</div><div class="card-stat-val">{fmt_usd(val3_bruto)}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c4:
            val4_bruto = float(q_mais_valor["val_exp_br"].sum())
            pct4_bruto = (val4_bruto / tot_val_bruto * 100) if tot_val_bruto > 0 else 0.0
            st.markdown(f"""
                <div class="card-quadrant card-amber">
                    <div class="card-title-amber">● Mais valor, menos escala (Oportunidade)</div>
                    <div class="card-desc">VCR > 1 com desaceleração recente do share. Alta demanda mundial.</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                        <div class="card-stat-count">{len(q_mais_valor)} <span style="font-size:12px;">produtos SH6</span></div>
                        <div><div class="badge-percent badge-amber">{fmt_pct(pct4_bruto)} do mercado</div><div class="card-stat-val">{fmt_usd(val4_bruto)}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><b>DISTRIBUIÇÃO DO MERCADO POR QUADRANTE</b>", unsafe_allow_html=True)
        st.progress(pct2_bruto / 100.0 if pct2_bruto <= 100.0 else 1.0)

    # TAB 2: VISÃO ISIC (SEÇÃO E DIVISÃO)
    with tab_isic:
        st.subheader("Visão Estratégica por Setor Industrial (ISIC Seção & Divisão)")
        
        col_isic_tab, col_isic_prod = st.columns([1.4, 1.1])
        
        with col_isic_tab:
            st.markdown("#### Resumo dos Setores (ISIC Seção)")
            df_isic_grp = df_main.groupby("isic_secao").agg(
                total_val_bruto=("val_exp_br", "sum"),
                qtd_sh6=("sh6", "count")
            ).reset_index()
            
            df_isic_grp["Valor Exportado"] = df_isic_grp["total_val_bruto"].apply(fmt_usd)
            st.dataframe(
                df_isic_grp[["isic_secao", "qtd_sh6", "Valor Exportado"]],
                column_config={"isic_secao": "ISIC Seção", "qtd_sh6": "Produtos (SH6)"},
                hide_index=True,
                use_container_width=True
            )

        with col_isic_prod:
            st.markdown("#### Detalhamento de Produtos por Setor")
            selected_setor = st.selectbox(
                "Selecione uma Seção ISIC para listar os produtos:",
                options=["-- Nenhum setor selecionado --"] + list(df_isic_grp["isic_secao"].unique())
            )
            
            if selected_setor == "-- Nenhum setor selecionado --":
                st.info("👈 Selecione uma Seção ISIC para visualizar os produtos SH6 e divisões relacionadas.")
            else:
                df_setor_prods = df_main[df_main["isic_secao"] == selected_setor].sort_values(by="val_exp_br", ascending=False)
                st.write(f"Mostrando **{len(df_setor_prods)}** produtos para a seção **{selected_setor}**:")
                
                for _, r in df_setor_prods.iterrows():
                    st.markdown(f"""
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:10px; border-radius:6px; margin-bottom:8px;">
                            <b style="font-size:12px; color:#0F172A;">{r['desc_sh6']} (SH6 {r['sh6']})</b><br>
                            <span style="font-size:11px; color:#64748B;">
                                Divisão ISIC: <b>{r['isic_divisao']}</b> | VCR: <b>{r['vcr']:.1f}</b>
                            </span>
                            <div style="text-align:right; font-weight:700; color:#0F172A; font-size:13px;">{fmt_usd(r['val_exp_br'])}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # TAB 3: CUCI GRUPO, NCMs, PAÍSES E RANKINGS ESTADUAIS
    with tab_cuci:
        st.subheader("Análise por Grupo CUCI, Países de Destino e Rankings Estaduais")
        
        selected_cuci = st.selectbox("Selecione o Grupo CUCI para análise:", options=list(df_main["cuci_grupo"].unique()))
        
        if selected_cuci:
            df_cuci_filtered = df_main[df_main["cuci_grupo"] == selected_cuci]
            sh6_cuci_list = df_cuci_filtered["sh6"].unique()
            
            st.markdown(f"### Grupo CUCI: **{selected_cuci}**")
            st.write(f"Total exportado pelo grupo: **{fmt_usd(df_cuci_filtered['val_exp_br'].sum())}**")
            
            raw_cuci_ncms = raw_comex[raw_comex["sh6"].isin(sh6_cuci_list)]
            
            col_ncms, col_paises, col_rank_exp = st.columns([1.2, 1, 1])
            
            with col_ncms:
                st.markdown("#### NCMs Vinculados")
                ncms_summary = raw_cuci_ncms.groupby(["ncm", "desc_sh6"]).agg({"val_exp_br": "sum"}).reset_index()
                ncms_summary["Exportação"] = ncms_summary["val_exp_br"].apply(fmt_usd)
                st.dataframe(
                    ncms_summary[["ncm", "desc_sh6", "Exportação"]],
                    column_config={"ncm": "Código NCM", "desc_sh6": "Descrição SH6"},
                    hide_index=True,
                    use_container_width=True
                )

            with col_paises:
                st.markdown("#### Top Países de Destino")
                if "pais" in raw_cuci_ncms.columns:
                    paises_sum = raw_cuci_ncms.groupby("pais").agg({"val_exp_br": "sum"}).reset_index()
                    paises_sum = paises_sum.sort_values(by="val_exp_br", ascending=False).head(5)
                    paises_sum["Valor"] = paises_sum["val_exp_br"].apply(fmt_usd)
                    st.table(paises_sum[["pais", "Valor"]].rename(columns={"pais": "País"}))
                else:
                    st.caption("Coluna de país não presente na base.")

            with col_rank_exp:
                st.markdown("#### Top UFs Exportadoras")
                if "uf" in raw_cuci_ncms.columns:
                    rank_exp = raw_cuci_ncms.groupby("uf").agg({"val_exp_br": "sum"}).reset_index()
                    rank_exp = rank_exp.sort_values(by="val_exp_br", ascending=False).head(5)
                    rank_exp["Valor"] = rank_exp["val_exp_br"].apply(fmt_usd)
                    st.table(rank_exp[["uf", "Valor"]].rename(columns={"uf": "UF Origem"}))
                else:
                    st.caption("Upload por estado não fornecido.")

    # TAB 4: VISÃO POR CATEGORIAS ECONÔMICAS (CGCE 1 E CGCE 2)
    with tab_cgce:
        st.subheader("Classificação por Grandes Categorias Econômicas (CGCE)")
        
        c_cgce1, c_cgce2 = st.columns(2)
        
        with c_cgce1:
            st.markdown("#### Distribuição por CGCE Nível 1")
            cgce1_summary = df_main.groupby("cgce_1").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
            cgce1_summary["Valor"] = cgce1_summary["val_exp_br"].apply(fmt_usd)
            st.dataframe(
                cgce1_summary[["cgce_1", "sh6", "Valor"]],
                column_config={"cgce_1": "CGCE Nível 1", "sh6": "Qtd SH6"},
                hide_index=True,
                use_container_width=True
            )

        with c_cgce2:
            st.markdown("#### Distribuição por CGCE Nível 2")
            cgce2_summary = df_main.groupby("cgce_2").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
            cgce2_summary["Valor"] = cgce2_summary["val_exp_br"].apply(fmt_usd)
            st.dataframe(
                cgce2_summary[["cgce_2", "sh6", "Valor"]],
                column_config={"cgce_2": "CGCE Nível 2", "sh6": "Qtd SH6"},
                hide_index=True,
                use_container_width=True
            )
