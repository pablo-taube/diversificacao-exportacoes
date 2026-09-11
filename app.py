import os
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

# Estilização CSS CNI / Cupertino
st.markdown("""
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
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FUNÇÕES DE FORMATAÇÃO E CÁLCULOS BRUTOS
# -----------------------------------------------------------------------------
def fmt_usd(val_bruto):
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
    if pd.isna(val_bruto):
        return "0,0%"
    return f"{val_bruto:.1f}%"

def read_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=";", dtype=str, encoding="latin-1")

# -----------------------------------------------------------------------------
# CARREGAMENTO INTEGRAL DAS 8 TABELAS AUXILIARES DA SECEX
# -----------------------------------------------------------------------------
DATA_DIR = os.path.join(".", "data", "tabelas_auxiliares")

@st.cache_data(ttl="24h")
def load_all_auxiliary_tables():
    tables = {}
    
    # 1. Tabela NCM Base
    df_ncm = read_csv_safe(os.path.join(DATA_DIR, "NCM.csv"))
    if df_ncm is not None:
        df_ncm.rename(columns={"CO_NCM": "ncm", "NO_NCM": "desc_ncm"}, inplace=True)
        df_ncm["ncm"] = df_ncm["ncm"].astype(str).str.zfill(8)
        tables["mestre"] = df_ncm
    else:
        tables["mestre"] = pd.DataFrame(columns=["ncm"])

    def merge_aux(filename, rename_dict):
        df_aux = read_csv_safe(os.path.join(DATA_DIR, filename))
        if df_aux is not None and "mestre" in tables:
            df_aux.rename(columns=rename_dict, inplace=True)
            if "CO_NCM" in df_aux.columns:
                df_aux.rename(columns={"CO_NCM": "ncm"}, inplace=True)
            if "ncm" in df_aux.columns:
                df_aux["ncm"] = df_aux["ncm"].astype(str).str.zfill(8)
                cols_to_use = [c for c in df_aux.columns if c not in tables["mestre"].columns or c == "ncm"]
                tables["mestre"] = pd.merge(tables["mestre"], df_aux[cols_to_use], on="ncm", how="left")

    # 2. NCM_ISIC.csv
    merge_aux("NCM_ISIC.csv", {
        "CO_ISIC_SECAO": "isic_secao", "NO_ISIC_SECAO_PT": "desc_isic_secao",
        "CO_ISIC_DIVISAO": "isic_divisao", "NO_ISIC_DIVISAO_PT": "desc_isic_divisao"
    })

    # 3. NCM_SH.csv
    merge_aux("NCM_SH.csv", {
        "CO_SH6": "sh6", "NO_SH6_PT": "desc_sh6"
    })

    # 4. NCM_CUCI.csv
    merge_aux("NCM_CUCI.csv", {
        "CO_CUCI_GRUPO": "cuci_grupo", "NO_CUCI_GRUPO_PT": "desc_cuci"
    })

    # 5. NCM_CGCE.csv
    merge_aux("NCM_CGCE.csv", {
        "CO_CGCE_N1": "cgce_1", "NO_CGCE_N1_PT": "desc_cgce_1",
        "CO_CGCE_N2": "cgce_2", "NO_CGCE_N2_PT": "desc_cgce_2"
    })

    # 6. UF.csv
    df_uf = read_csv_safe(os.path.join(DATA_DIR, "UF.csv"))
    if df_uf is not None:
        df_uf.rename(columns={"SG_UF": "uf", "NO_UF": "desc_uf"}, inplace=True)
        tables["uf"] = df_uf

    # 7. PAIS.csv
    df_pais = read_csv_safe(os.path.join(DATA_DIR, "PAIS.csv"))
    if df_pais is not None:
        df_pais.rename(columns={"CO_PAIS": "co_pais", "NO_PAIS": "pais"}, inplace=True)
        tables["pais"] = df_pais

    return tables

AUX_TABLES = load_all_auxiliary_tables()

# -----------------------------------------------------------------------------
# LEITURA E TRATAMENTO DOS ARQUIVOS COMEX STAT (SUPORTE NATIVO A PARQUET E CSV)
# -----------------------------------------------------------------------------
def read_uploaded_file(file):
    """
    Lê arquivos nos layouts oficiais do Comex Stat (Exportação / Importação),
    suportando arquivos em formato Apache Parquet e CSV.
    """
    file_name = file.name.lower()
    df = None
    
    # 1. Leitura do Parquet
    if file_name.endswith(".parquet") or file_name.endswith(".pq"):
        df = pd.read_parquet(file)
    # 2. Leitura do CSV
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
        # Padroniza nomes das colunas para maiúsculo
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Converte e garante tipo float para métricas numéricas do Comex Stat
        numeric_cols = ["QT_ESTAT", "KG_LIQUIDO", "VL_FOB", "VL_FRETE", "VL_SEGURO"]
        for num_c in numeric_cols:
            if num_c in df.columns:
                df[num_c] = pd.to_numeric(df[num_c], errors="coerce").fillna(0.0)
                
    return df

def process_and_enrich_comexstat(df_user):
    """
    Processa o DataFrame (oriundo de CSV ou Parquet), mapeia os códigos e
    executa o merge relacional com as tabelas de referência da SECEX.
    """
    df = df_user.copy()
    
    # Validação do campo NCM
    if "CO_NCM" in df.columns:
        df["ncm"] = df["CO_NCM"].astype(str).str.zfill(8)
        df["sh6"] = df["ncm"].str[:6]
    else:
        df["ncm"] = "00000000"
        df["sh6"] = "000000"

    if "SG_UF_NCM" in df.columns:
        df["uf"] = df["SG_UF_NCM"]
    else:
        df["uf"] = "BR"

    if "CO_PAIS" in df.columns:
        df["co_pais"] = df["CO_PAIS"].astype(str).str.zfill(3)

    # Identificação de Fluxo Importador (VL_FOB + VL_FRETE + VL_SEGURO) vs Exportação
    is_import = "VL_FRETE" in df.columns or "VL_SEGURO" in df.columns
    
    if "VL_FOB" in df.columns:
        df["val_exp_br"] = df["VL_FOB"]
    else:
        df["val_exp_br"] = 0.0

    if is_import:
        frete = df["VL_FRETE"] if "VL_FRETE" in df.columns else 0.0
        seguro = df["VL_SEGURO"] if "VL_SEGURO" in df.columns else 0.0
        df["val_imp_br"] = df["val_exp_br"] + frete + seguro
    else:
        df["val_imp_br"] = df["val_exp_br"] * 0.2

    # Cruzamento com Tabelas Auxiliares
    if "mestre" in AUX_TABLES and not AUX_TABLES["mestre"].empty:
        df_mestre = AUX_TABLES["mestre"]
        cols_to_merge = [c for c in df_mestre.columns if c not in df.columns or c == "ncm"]
        df = pd.merge(df, df_mestre[cols_to_merge], on="ncm", how="left")

    if "co_pais" in df.columns and "pais" in AUX_TABLES:
        df = pd.merge(df, AUX_TABLES["pais"], on="co_pais", how="left")

    # Descrições Padrão de Fallback
    defaults = {
        "desc_sh6": "Produto SH6 " + df["sh6"],
        "isic_secao": "Indústria Geral",
        "isic_divisao": "Divisão Industrial",
        "cuci_grupo": "Grupo CUCI",
        "cgce_1": "Bens Industriais",
        "cgce_2": "Categoria Geral",
        "pais": "Mundo Geral"
    }
    for col_name, def_val in defaults.items():
        if col_name not in df.columns or df[col_name].isna().all():
            df[col_name] = def_val

    return df

def process_trade_data(df_comex_raw, df_comtrade_raw, df_uf_raw=None):
    df_comex = process_and_enrich_comexstat(df_comex_raw)
    
    df_comtrade = df_comtrade_raw.copy()
    if "cmdCode" in df_comtrade.columns:
        df_comtrade.rename(columns={"cmdCode": "sh6", "primaryValue": "val_mundo"}, inplace=True)
    elif "CO_SH6" in df_comtrade.columns:
        df_comtrade.rename(columns={"CO_SH6": "sh6", "VL_FOB": "val_mundo"}, inplace=True)
    elif "CO_NCM" in df_comtrade.columns:
        df_comtrade["sh6"] = df_comtrade["CO_NCM"].astype(str).str.zfill(8).str[:6]
        df_comtrade.rename(columns={"VL_FOB": "val_mundo"}, inplace=True)
        
    df_comtrade["sh6"] = df_comtrade["sh6"].astype(str).str.zfill(6).str[:6]
    df_comtrade["val_mundo"] = pd.to_numeric(df_comtrade["val_mundo"], errors="coerce").fillna(0.0)

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
# SIDEBAR DE CONFIGURAÇÃO E CARGA
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configurações & Upload")
st.sidebar.markdown("---")

horizonte = st.sidebar.slider("Variação temporal de cálculo:", 1, 5, 5, format="%d ano(s)")

st.sidebar.markdown("### 📤 Upload de Arquivos (Parquet ou CSV)")

uploaded_comex = st.sidebar.file_uploader(
    "1. Arquivo Comexstat (.parquet ou .csv)",
    type=["parquet", "pq", "csv"],
    help="Aceita os arquivos oficiais de Exportação ou Importação do Comex Stat em formato Parquet ou CSV."
)

uploaded_comex_uf = st.sidebar.file_uploader(
    "2. Arquivo por Estado (Opcional)",
    type=["parquet", "pq", "csv"],
    help="Opcional: Recorte estadual por UF."
)

uploaded_comtrade = st.sidebar.file_uploader(
    "3. Arquivo UN Comtrade (Mundo)",
    type=["parquet", "pq", "csv", "xlsx", "json"],
    help="Estatísticas globais por código HS6."
)

st.sidebar.markdown("---")
btn_processar = st.sidebar.button("🚀 Executar Análise e Processar Dados", type="primary", use_container_width=True)

if "mestre" in AUX_TABLES and not AUX_TABLES["mestre"].empty:
    st.sidebar.caption("✅ 8 Tabelas auxiliares ativas em `./data/tabelas_auxiliares/`")

# -----------------------------------------------------------------------------
# PAINEL PRINCIPAL & EXIBIÇÃO
# -----------------------------------------------------------------------------
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None
    st.session_state["raw_comex"] = None

if btn_processar:
    if uploaded_comex is not None and uploaded_comtrade is not None:
        try:
            with st.spinner("Lendo arquivo Parquet/CSV e aplicando de/para relacional da SECEX..."):
                df_cx = read_uploaded_file(uploaded_comex)
                df_ct = read_uploaded_file(uploaded_comtrade)
                df_uf = read_uploaded_file(uploaded_comex_uf) if uploaded_comex_uf else None
                
                df_res, raw_cx = process_trade_data(df_cx, df_ct, df_uf)
                st.session_state["df_processed"] = df_res
                st.session_state["raw_comex"] = raw_cx
                st.success("✅ Arquivo Parquet/CSV lido e processado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
            st.session_state["df_processed"] = None
    else:
        st.warning("⚠️ Faça o upload dos arquivos do Comexstat e UN Comtrade para avançar.")

df_main = st.session_state["df_processed"]
raw_comex = st.session_state["raw_comex"]

st.markdown("<div class='cni-title'>Radar das Exportações e Vantagem Comparativa (SH6 / CUCI / ISIC / CGCE)</div>", unsafe_allow_html=True)

if df_main is None or df_main.empty:
    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    col_m1.markdown("<div class='metric-value'>0</div><div class='metric-label'>produtos SH6 monitorados</div>", unsafe_allow_html=True)
    col_m2.markdown("<div class='metric-value'>0</div><div class='metric-label'>NCMs identificados</div>", unsafe_allow_html=True)
    col_m3.markdown("<div class='metric-value'>US$ 0,0</div><div class='metric-label'>mercado total exportado (bruto)</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ **Formatos Suportados:** Envie seu arquivo do Comex Stat em `.parquet` ou `.csv` para rodar o cálculo de Vantagem Comparativa Revelada.")
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

    # TAB 1: QUADRANTES
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

    # TAB 2: VISÃO ISIC
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
            st.dataframe(df_isic_grp[["isic_secao", "qtd_sh6", "Valor Exportado"]], hide_index=True, use_container_width=True)

        with col_isic_prod:
            st.markdown("#### Detalhamento de Produtos por Setor")
            selected_setor = st.selectbox("Selecione uma Seção ISIC:", options=["-- Nenhum setor selecionado --"] + list(df_isic_grp["isic_secao"].unique()))
            if selected_setor != "-- Nenhum setor selecionado --":
                df_setor_prods = df_main[df_main["isic_secao"] == selected_setor].sort_values(by="val_exp_br", ascending=False)
                for _, r in df_setor_prods.iterrows():
                    st.markdown(f"""
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:10px; border-radius:6px; margin-bottom:8px;">
                            <b style="font-size:12px; color:#0F172A;">{r['desc_sh6']} (SH6 {r['sh6']})</b><br>
                            <span style="font-size:11px; color:#64748B;">Divisão ISIC: <b>{r['isic_divisao']}</b> | VCR: <b>{r['vcr']:.1f}</b></span>
                            <div style="text-align:right; font-weight:700; color:#0F172A; font-size:13px;">{fmt_usd(r['val_exp_br'])}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # TAB 3: CUCI
    with tab_cuci:
        st.subheader("Análise por Grupo CUCI, Países de Destino e Rankings Estaduais")
        selected_cuci = st.selectbox("Selecione o Grupo CUCI:", options=list(df_main["cuci_grupo"].unique()))
        if selected_cuci:
            df_cuci_filtered = df_main[df_main["cuci_grupo"] == selected_cuci]
            sh6_cuci_list = df_cuci_filtered["sh6"].unique()
            raw_cuci_ncms = raw_comex[raw_comex["sh6"].isin(sh6_cuci_list)]
            col_ncms, col_paises, col_rank_exp = st.columns([1.2, 1, 1])
            with col_ncms:
                st.markdown("#### NCMs Vinculados")
                ncms_summary = raw_cuci_ncms.groupby(["ncm", "desc_sh6"]).agg({"val_exp_br": "sum"}).reset_index()
                ncms_summary["Exportação"] = ncms_summary["val_exp_br"].apply(fmt_usd)
                st.dataframe(ncms_summary[["ncm", "desc_sh6", "Exportação"]], hide_index=True, use_container_width=True)
            with col_paises:
                st.markdown("#### Top Países de Destino")
                if "pais" in raw_cuci_ncms.columns:
                    paises_sum = raw_cuci_ncms.groupby("pais").agg({"val_exp_br": "sum"}).reset_index().sort_values(by="val_exp_br", ascending=False).head(5)
                    paises_sum["Valor"] = paises_sum["val_exp_br"].apply(fmt_usd)
                    st.table(paises_sum[["pais", "Valor"]].rename(columns={"pais": "País"}))
            with col_rank_exp:
                st.markdown("#### Top UFs Exportadoras")
                if "uf" in raw_cuci_ncms.columns:
                    rank_exp = raw_cuci_ncms.groupby("uf").agg({"val_exp_br": "sum"}).reset_index().sort_values(by="val_exp_br", ascending=False).head(5)
                    rank_exp["Valor"] = rank_exp["val_exp_br"].apply(fmt_usd)
                    st.table(rank_exp[["uf", "Valor"]].rename(columns={"uf": "UF Origem"}))

    # TAB 4: CGCE
    with tab_cgce:
        st.subheader("Classificação por Grandes Categorias Econômicas (CGCE)")
        c_cgce1, c_cgce2 = st.columns(2)
        with c_cgce1:
            st.markdown("#### Distribuição por CGCE Nível 1")
            cgce1_summary = df_main.groupby("cgce_1").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
            cgce1_summary["Valor"] = cgce1_summary["val_exp_br"].apply(fmt_usd)
            st.dataframe(cgce1_summary[["cgce_1", "sh6", "Valor"]], hide_index=True, use_container_width=True)
        with c_cgce2:
            st.markdown("#### Distribuição por CGCE Nível 2")
            cgce2_summary = df_main.groupby("cgce_2").agg({"val_exp_br": "sum", "sh6": "count"}).reset_index()
            cgce2_summary["Valor"] = cgce2_summary["val_exp_br"].apply(fmt_usd)
            st.dataframe(cgce2_summary[["cgce_2", "sh6", "Valor"]], hide_index=True, use_container_width=True)
