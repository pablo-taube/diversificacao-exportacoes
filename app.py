import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER PARA LEITURA E NORMALIZAÇÃO DE COLUNAS
# ==============================================================================
def normalize_dataframe(df):
    """Normaliza nomes de colunas para padrão esperado pelo sistema."""
    if df is None or df.empty:
        return df

    # Limpa nomes das colunas (remove espaços extras)
    df.columns = df.columns.astype(str).str.strip()

    # Mapeamento flexível de aliases comuns do Comex Stat / Comtrade
    column_mapping = {
        'ano': 'Ano', 'CO_ANO': 'Ano', 'Year': 'Ano',
        'Código SH6': 'Código SH6', 'CO_SH6': 'Código SH6', 'SH6': 'Código SH6', 'cmdCode': 'Código SH6',
        'Descrição SH6': 'Descrição SH6', 'NO_SH6_POR': 'Descrição SH6', 'NO_SH6_ESP': 'Descrição SH6',
        'Código CUCI Grupo': 'Código CUCI Grupo', 'CO_CUCI_GRUPO': 'Código CUCI Grupo',
        'Descrição CUCI Grupo': 'Descrição CUCI Grupo', 'NO_CUCI_GRUPO': 'Descrição CUCI Grupo',
        'Código ISIC Divisão': 'Código ISIC Divisão', 'CO_ISIC_DIVISAO': 'Código ISIC Divisão',
        'Descrição ISIC Divisão': 'Descrição ISIC Divisão', 'NO_ISIC_DIVISAO': 'Descrição ISIC Divisão',
        'Código ISIC Seção': 'Código ISIC Seção', 'CO_ISIC_SECAO': 'Código ISIC Seção',
        'Descrição ISIC Seção': 'Descrição ISIC Seção', 'NO_ISIC_SECAO': 'Descrição ISIC Seção',
        'Valor US$ FOB': 'Valor BR FOB', 'VL_FOB': 'Valor BR FOB', 'Valor BR FOB': 'Valor BR FOB', 'primaryValue': 'Valor BR FOB'
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Garante a existência das colunas essenciais preenchendo valores genéricos se faltar metadados
    defaults = {
        'Descrição SH6': df['Código SH6'].astype(str) if 'Código SH6' in df.columns else 'N/A',
        'Código CUCI Grupo': '999', 'Descrição CUCI Grupo': 'Outros Grupos',
        'Código ISIC Divisão': '99', 'Descrição ISIC Divisão': 'Outras Divisões',
        'Código ISIC Seção': 'Z', 'Descrição ISIC Seção': 'Outras Seções',
        'Valor World FOB': df['Valor BR FOB'] * 5 if 'Valor BR FOB' in df.columns else 0
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    return df


@st.cache_data(show_spinner="Carregando e processando arquivo...")
def load_uploaded_file(file):
    """Lê arquivos enviados garantindo suporte a múltiplos separadores de CSV."""
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith('.parquet'):
            df = pd.read_parquet(file)
        elif name.endswith('.csv'):
            # Tenta ler com autodetecção de separador e engine python para evitar erro de tokenização
            try:
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
            except Exception:
                file.seek(0)
                df = pd.read_csv(file, sep=';', on_bad_lines='skip')
        elif name.endswith('.json'):
            df = pd.read_json(file)
        elif name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error(f"Formato não suportado: {file.name}")
            return None

        return normalize_dataframe(df)

    except Exception as e:
        st.error(f"Erro ao ler o arquivo **{file.name}**: {e}")
        return None


def calc_cagr(start_val, end_val, periods):
    """Calcula a Taxa de Crescimento Anual Composta (CAGR)."""
    if pd.isna(start_val) or pd.isna(end_val) or start_val <= 0 or periods <= 0:
        return np.nan
    return ((end_val / start_val) ** (1 / periods)) - 1


def generate_mock_comtrade():
    """Gera dados fictícios do UN Comtrade para demonstração."""
    anos = [2023, 2024, 2025]
    sh6_list = [
        ("090111", "Café não torrado, não descafeinado", "071", "Café e sucedâneos", "01", "Agricultura e Pecuária", "A", "Agricultura, Pecuária e Florestal"),
        ("120190", "Soja, mesmo triturada", "222", "Sementes oleaginosas", "01", "Agricultura e Pecuária", "A", "Agricultura, Pecuária e Florestal"),
        ("260111", "Minérios de ferro não aglomerados", "281", "Minérios de ferro", "07", "Extração de Minerais Metálicos", "B", "Indústrias Extrativas"),
        ("270900", "Óleos brutos de petróleo", "333", "Petróleo bruto", "06", "Extração de Petróleo e Gás", "B", "Indústrias Extrativas"),
        ("470321", "Pasta química de madeira (celulose)", "251", "Pasta de madeira", "17", "Fabricação de Celulose e Papel", "C", "Indústria de Transformação"),
        ("020230", "Carne bovina desossada congelada", "012", "Carne bovina", "10", "Fabricação de Produtos Alimentícios", "C", "Indústria de Transformação")
    ]
    
    rows = []
    np.random.seed(42)
    for ano in anos:
        for item in sh6_list:
            base_world = np.random.uniform(1000, 5000) * (1 + (ano - 2023) * 0.05)
            base_br = base_world * np.random.uniform(0.1, 0.3)
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
                "Valor World FOB": round(base_world * 1000, 2),
                "Valor BR FOB": round(base_br * 1000, 2)
            })
    return pd.DataFrame(rows)


def generate_mock_comexstat():
    """Gera dados fictícios do Comex Stat por Estado (UF) para demonstração."""
    anos = [2023, 2024, 2025]
    ufs = ["SP", "MG", "PR", "RS", "MT", "TO"]
    sh6_codes = ["090111", "120190", "260111", "270900", "470321", "020230"]
    
    rows = []
    np.random.seed(100)
    for ano in anos:
        for sh in sh6_codes:
            for uf in np.random.choice(ufs, size=2, replace=False):
                rows.append({
                    "Ano": ano,
                    "UF": uf,
                    "Código SH6": sh,
                    "Valor US$ FOB": round(np.random.uniform(50, 500) * 1000, 2)
                })
    return pd.DataFrame(rows)


@st.cache_data
def process_analytics(df):
    """Calcula métricas agregadas de competitividade (RCA, CAGR, Quadrantes)."""
    if df is None or "Ano" not in df.columns:
        return None, 0, 0, 0, 0

    anos = sorted(df["Ano"].dropna().unique())
    if len(anos) < 2:
        st.warning("⚠️ A base de dados precisa ter pelo menos 2 anos distintos para análise temporal.")
        return None, 0, 0, 0, 0

    t_ini, t_fim = anos[0], anos[-1]
    n_periodos = int(t_fim - t_ini)

    group_cols = [
        "Código SH6", "Descrição SH6", "Código CUCI Grupo", "Descrição CUCI Grupo",
        "Código ISIC Divisão", "Descrição ISIC Divisão", "Código ISIC Seção", "Descrição ISIC Seção"
    ]
    
    # Valida colunas existentes no agrupamento
    existing_group_cols = [c for c in group_cols if c in df.columns]

    df_ini = df[df["Ano"] == t_ini].groupby(existing_group_cols)[["Valor World FOB", "Valor BR FOB"]].sum().reset_index()
    df_fim = df[df["Ano"] == t_fim].groupby(existing_group_cols)[["Valor World FOB", "Valor BR FOB"]].sum().reset_index()

    merged = pd.merge(df_ini, df_fim, on=existing_group_cols, suffixes=("_ini", "_fim"))

    total_world_ini = merged["Valor World FOB_ini"].sum()
    total_world_fim = merged["Valor World FOB_fim"].sum()
    total_br_ini = merged["Valor BR FOB_ini"].sum()
    total_br_fim = merged["Valor BR FOB_fim"].sum()

    # RCA
    merged["RCA_ini"] = np.where(total_br_ini > 0, (merged["Valor BR FOB_ini"] / total_br_ini) / (merged["Valor World FOB_ini"] / total_world_ini), 0)
    merged["RCA_fim"] = np.where(total_br_fim > 0, (merged["Valor BR FOB_fim"] / total_br_fim) / (merged["Valor World FOB_fim"] / total_world_fim), 0)

    # Market Share
    merged["Share_BR_ini"] = np.where(merged["Valor World FOB_ini"] > 0, merged["Valor BR FOB_ini"] / merged["Valor World FOB_ini"], 0)
    merged["Share_BR_fim"] = np.where(merged["Valor World FOB_fim"] > 0, merged["Valor BR FOB_fim"] / merged["Valor World FOB_fim"], 0)
    merged["Var_Share_BR"] = merged["Share_BR_fim"] - merged["Share_BR_ini"]

    # CAGRs
    merged["CAGR_World"] = merged.apply(lambda r: calc_cagr(r["Valor World FOB_ini"], r["Valor World FOB_fim"], n_periodos), axis=1)
    merged["CAGR_BR"] = merged.apply(lambda r: calc_cagr(r["Valor BR FOB_ini"], r["Valor BR FOB_fim"], n_periodos), axis=1)

    cagr_world_avg = calc_cagr(total_world_ini, total_world_fim, n_periodos)
    cagr_br_avg = calc_cagr(total_br_ini, total_br_fim, n_periodos)

    # Quadrantes
    def get_quadrant(row):
        mkt_growth = row["CAGR_World"] > cagr_world_avg if not pd.isna(row["CAGR_World"]) else False
        space_gain = row["Var_Share_BR"] > 0
        if mkt_growth and space_gain:
            return "Ganho de Espaço & Mercado Cresce (Oportunidade)"
        elif mkt_growth and not space_gain:
            return "Perda de Espaço & Mercado Cresce (Ameaça)"
        elif not mkt_growth and space_gain:
            return "Ganho de Espaço & Mercado Cai (Vulnerabilidade)"
        else:
            return "Retirada / Declínio"

    merged["Quadrante"] = merged.apply(get_quadrant, axis=1)

    return merged, cagr_world_avg, cagr_br_avg, t_ini, t_fim


# ==============================================================================
# 3. BARRA LATERAL
# ==============================================================================
st.sidebar.title("🛠️ Configurações & Dados")

st.sidebar.subheader("1. Base UN Comtrade / Brasil")
comtrade_file = st.sidebar.file_uploader(
    "Upload Comtrade/Brasil (CSV, Parquet, JSON, Excel)",
    type=["csv", "parquet", "json", "xlsx", "xls"],
    key="comtrade"
)

st.sidebar.subheader("2. Base Comex Stat (Opcional)")
comexstat_file = st.sidebar.file_uploader(
    "Upload Comex Stat por UF (CSV, Parquet, Excel)",
    type=["csv", "parquet", "xlsx", "xls"],
    key="comexstat"
)

# Carregamento e Fallback
if comtrade_file is not None:
    df_raw = load_uploaded_file(comtrade_file)
else:
    st.sidebar.info("💡 Usando dados demonstrativos.")
    df_raw = generate_mock_comtrade()

if comexstat_file is not None:
    df_uf_raw = load_uploaded_file(comexstat_file)
else:
    df_uf_raw = generate_mock_comexstat()

# Processamento
if df_raw is not None and "Ano" in df_raw.columns:
    analytics_df, cagr_w_avg, cagr_b_avg, t_start, t_end = process_analytics(df_raw)
else:
    analytics_df = None


# ==============================================================================
# 4. PÁGINAS DO SISTEMA
# ==============================================================================
def page_dashboard():
    st.title("📊 Dashboard Executivo de Diversificação")

    if analytics_df is None:
        st.error("Não foi possível processar os dados. Verifique a estrutura do arquivo enviado.")
        return

    st.caption(f"Análise comparativa do período de **{t_start} a {t_end}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR Médio Global", f"{cagr_w_avg:.2%}")
    col2.metric("CAGR Total Brasil", f"{cagr_b_avg:.2%}")
    
    diff_cagr = cagr_b_avg - cagr_w_avg
    col3.metric("Performance Relativa", f"{diff_cagr:+.2%}", delta="Superou o Mundo" if diff_cagr > 0 else "Abaixo do Mundo")

    n_oportunidades = len(analytics_df[analytics_df["Quadrante"] == "Ganho de Espaço & Mercado Cresce (Oportunidade)"])
    col4.metric("Oportunidades (SH6)", f"{n_oportunidades} de {len(analytics_df)}")

    st.markdown("---")
    st.subheader("📍 Matriz Estratégica de Posicionamento (Quadrantes)")

    fig = px.scatter(
        analytics_df,
        x="Var_Share_BR",
        y="CAGR_World",
        size="Valor BR FOB_fim",
        color="Quadrante",
        hover_name="Descrição SH6",
        hover_data=["Código SH6", "RCA_fim", "CAGR_BR"],
        labels={"Var_Share_BR": "Variação de Share do Brasil", "CAGR_World": "CAGR do Mercado Global"},
        color_discrete_map={
            "Ganho de Espaço & Mercado Cresce (Oportunidade)": "#2ea44f",
            "Perda de Espaço & Mercado Cresce (Ameaça)": "#cb2431",
            "Ganho de Espaço & Mercado Cai (Vulnerabilidade)": "#dbab09",
            "Retirada / Declínio": "#6a737d"
        },
        height=550
    )

    fig.add_hline(y=cagr_w_avg, line_dash="dash", line_color="gray", annotation_text="CAGR Médio Global")
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Share Neutro")

    st.plotly_chart(fig, use_container_width=True)


def page_sh6():
    st.title("📋 Tabela Analítica Completa por SH6")

    if analytics_df is None:
        st.error("Sem dados para exibir.")
        return

    col_f1, col_f2, col_f3 = st.columns(3)
    list_cuci = ["Todos"] + sorted(analytics_df["Descrição CUCI Grupo"].astype(str).unique().tolist())
    list_isic_sec = ["Todos"] + sorted(analytics_df["Descrição ISIC Seção"].astype(str).unique().tolist())
    list_isic_div = ["Todos"] + sorted(analytics_df["Descrição ISIC Divisão"].astype(str).unique().tolist())

    sel_cuci = col_f1.selectbox("Filtrar por CUCI Grupo", list_cuci)
    sel_isic_sec = col_f2.selectbox("Filtrar por ISIC Seção", list_isic_sec)
    sel_isic_div = col_f3.selectbox("Filtrar por ISIC Divisão", list_isic_div)

    filtered = analytics_df.copy()
    if sel_cuci != "Todos":
        filtered = filtered[filtered["Descrição CUCI Grupo"] == sel_cuci]
    if sel_isic_sec != "Todos":
        filtered = filtered[filtered["Descrição ISIC Seção"] == sel_isic_sec]
    if sel_isic_div != "Todos":
        filtered = filtered[filtered["Descrição ISIC Divisão"] == sel_isic_div]

    cols_display = [
        "Código SH6", "Descrição SH6", "Descrição CUCI Grupo", "Valor BR FOB_fim",
        "Share_BR_ini", "Share_BR_fim", "Var_Share_BR", "CAGR_World", "CAGR_BR", "RCA_fim", "Quadrante"
    ]

    st.dataframe(
        filtered[cols_display].style.format({
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


def page_cuci():
    st.title("📦 Agrupamento por CUCI Grupo")

    if analytics_df is None:
        st.error("Sem dados para exibir.")
        return

    cuci_summary = analytics_df.groupby(["Código CUCI Grupo", "Descrição CUCI Grupo"]).agg(
        Total_BR_Fob=("Valor BR FOB_fim", "sum"),
        Total_World_Fob=("Valor World FOB_fim", "sum"),
        Qtd_SH6=("Código SH6", "count"),
        CAGR_World_Medio=("CAGR_World", "mean"),
        Var_Share_Medio=("Var_Share_BR", "mean")
    ).reset_index()

    st.dataframe(
        cuci_summary.style.format({
            "Total_BR_Fob": "US$ {:,.2f}",
            "Total_World_Fob": "US$ {:,.2f}",
            "CAGR_World_Medio": "{:.2%}",
            "Var_Share_Medio": "{:+.2%}"
        }),
        use_container_width=True, hide_index=True
    )


def page_uf():
    st.title("🗺️ Análise por Estado Exportador (Comex Stat)")

    if df_uf_raw is None or df_uf_raw.empty:
        st.warning("⚠️ Nenhum dado de exportação estadual carregado.")
        return

    st.dataframe(df_uf_raw.head(50), use_container_width=True)


# ==============================================================================
# 5. ROTEAMENTO DAS PÁGINAS
# ==============================================================================
pg = st.navigation([
    st.Page(page_dashboard, title="Dashboard Geral", icon="📊"),
    st.Page(page_sh6, title="Detalhamento SH6", icon="📋"),
    st.Page(page_cuci, title="Visão CUCI Grupo", icon="📦"),
    st.Page(page_uf, title="Exportação por UF", icon="🗺️"),
])

pg.run()
