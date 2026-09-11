"""
Radar de Comércio Exterior & VCR - CNI / CIN
Módulo simplificado focado em processamento direto de dados UN Comtrade.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# =============================================================================
# CONSTANTES E LISTA UE-27
# =============================================================================
PAGE_TITLE = "Análise UN Comtrade & RCA - CNI / CIN"
PAGE_ICON = "📡"

# ISO3 ou Nomes Oficiais comumente encontrados nos relatórios do UN Comtrade
EU_27_REPORTERS = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
    "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT",
    "ROU", "SVK", "SVN", "ESP", "SWE"
]

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO E TRATAMENTO DE DADOS
# =============================================================================
def fmt_usd(valor: float) -> str:
    if pd.isna(valor) or valor == 0:
        return "US$ 0,0"
    abs_val = abs(valor)
    if abs_val >= 1e9:
        return f"US$ {valor / 1e9:.2f} Bi"
    if abs_val >= 1e6:
        return f"US$ {valor / 1e6:.2f} Mi"
    if abs_val >= 1e3:
        return f"US$ {valor / 1e3:.2f} Mil"
    return f"US$ {valor:.2f}"


def fmt_pct(valor: float) -> str:
    if pd.isna(valor):
        return "0,0%"
    return f"{valor * 100:.2f}%"


def read_comtrade_file(file) -> pd.DataFrame | None:
    """Lê arquivos exportados do UN Comtrade (CSV, Parquet, Excel ou JSON)."""
    file_name = file.name.lower()
    df = None

    if file_name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(file)
    elif file_name.endswith(".csv"):
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
        # Padroniza colunas do UN Comtrade (layout V1 / API recente)
        cols_map = {
            "period": "ano", "periodCode": "ano", "Year": "ano",
            "cmdCode": "sh6", "cmdCodeHS": "sh6", "Commodity Code": "sh6",
            "reporterDesc": "reporter", "reporterISO": "reporter", "Reporter": "reporter",
            "partnerDesc": "partner", "partnerISO": "partner", "Partner": "partner",
            "primaryValue": "valor", "tradeValue": "valor", "Value": "valor"
        }
        
        renames = {c: cols_map[c] for c in df.columns if c in cols_map}
        df.rename(columns=renames, inplace=True)
        
        # Garante tratamento do código SH6 e valores
        if "sh6" in df.columns:
            df["sh6"] = df["sh6"].astype(str).str.zfill(6).str[:6]
        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
        if "valor" in df.columns:
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)

    return df

# =============================================================================
# NÚCLEO DE CÁLCULO E PROCESSAMENTO ANALÍTICO
# =============================================================================
def process_comtrade_analytics(df_raw: pd.DataFrame, agregar_ue: bool) -> pd.DataFrame:
    df = df_raw.copy()

    # 1. Trata Agregação da União Europeia
    if agregar_ue and "reporter" in df.columns:
        df["reporter"] = df["reporter"].apply(
            lambda r: "União Europeia" if str(r).strip() in EU_27_REPORTERS else r
        )

    # 2. Identifica período de análise (Primeiro e Último ano)
    anos = sorted(df["ano"].unique())
    if len(anos) < 2:
        st.error("O arquivo precisa conter ao menos 2 anos de histórico para calcular CAGR e Variação.")
        return pd.DataFrame()

    ano_inicio, ano_fim = anos[0], anos[-1]
    n_anos = ano_fim - ano_inicio

    # 3. Filtragem por parceiros (Brazil x World)
    df["partner_clean"] = df["partner"].astype(str).str.lower()
    
    # Isola fluxos do Mundo e do Brasil
    is_world = df["partner_clean"].isin(["world", "all", "0", "total"])
    is_brazil = df["partner_clean"].isin(["brazil", "bra", "76"])

    df_w = df[is_world]
    df_br = df[is_brazil]

    # Agrupamentos base por SH6 e Ano
    world_by_sh6_year = df_w.groupby(["sh6", "ano"])["valor"].sum().unstack(fill_value=0)
    brazil_by_sh6_year = df_br.groupby(["sh6", "ano"])["valor"].sum().unstack(fill_value=0)

    # Garante que todos os SH6 existam em ambas as estruturas
    all_sh6 = sorted(list(set(world_by_sh6_year.index).union(set(brazil_by_sh6_year.index))))
    world_by_sh6_year = world_by_sh6_year.reindex(all_sh6, fill_value=0.0)
    brazil_by_sh6_year = brazil_by_sh6_year.reindex(all_sh6, fill_value=0.0)

    # Totais globais para o cálculo do RCA
    tot_w_inicio = world_by_sh6_year[ano_inicio].sum()
    tot_w_fim = world_by_sh6_year[ano_fim].sum()
    tot_br_inicio = brazil_by_sh6_year[ano_inicio].sum()
    tot_br_fim = brazil_by_sh6_year[ano_fim].sum()

    resultados = []

    for sh6 in all_sh6:
        w_ini = world_by_sh6_year.loc[sh6, ano_inicio]
        w_fim = world_by_sh6_year.loc[sh6, ano_fim]
        br_ini = brazil_by_sh6_year.loc[sh6, ano_inicio]
        br_fim = brazil_by_sh6_year.loc[sh6, ano_fim]

        # Shares
        share_ini = (br_ini / w_ini) if w_ini > 0 else 0.0
        share_fim = (br_fim / w_fim) if w_fim > 0 else 0.0
        variacao_share = share_fim - share_ini

        # CAGRs
        cagr_w = ((w_fim / w_ini) ** (1 / n_anos) - 1) if (w_ini > 0 and w_fim > 0) else 0.0
        cagr_br = ((br_fim / br_ini) ** (1 / n_anos) - 1) if (br_ini > 0 and br_fim > 0) else 0.0

        # Verificação do maior crescimento
        maior_crescimento_br = cagr_br > cagr_w

        # Cálculo RCA / VCR no último ano
        rca = (br_fim / tot_br_fim) / (w_fim / tot_w_fim) if (tot_br_fim > 0 and w_fim > 0 and tot_w_fim > 0) else 0.0

        resultados.append({
            "Código SH6": sh6,
            f"Imp. Mundo ({ano_fim})": w_fim,
            f"Imp. Brasil ({ano_fim})": br_fim,
            f"Share Brasil ({ano_fim})": share_fim,
            "Variação Share (pp)": variacao_share,
            f"CAGR Global ({ano_inicio}-{ano_fim})": cagr_w,
            f"CAGR Brasil ({ano_inicio}-{ano_fim})": cagr_br,
            "Maior Crescimento BR": "Sim" if maior_crescimento_br else "Não",
            "RCA / VCR": rca,
        })

    return pd.DataFrame(resultados)

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================
def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title("📡 Módulo de Análise de Importações UN Comtrade & RCA")

    st.sidebar.header("⚙️ Parâmetros de Entrada")
    
    file_comtrade = st.sidebar.file_uploader(
        "Upload da Base UN Comtrade",
        type=["csv", "parquet", "pq", "xlsx", "json"],
        help="Envie a extração contendo importações por SH6, Reporters e Partners (World e Brazil)."
    )

    agregar_ue = st.sidebar.checkbox(
        "Agrupar União Europeia (UE-27)",
        value=True,
        help="Soma os dados de importação dos 27 membros da União Europeia em um único registro."
    )

    if file_comtrade is not None:
        with st.spinner("Processando base UN Comtrade..."):
            df_raw = read_comtrade_file(file_comtrade)
            
            if df_raw is not None and not df_raw.empty:
                df_res = process_comtrade_analytics(df_raw, agregar_ue)

                if not df_res.empty:
                    st.success("✅ Cálculos concluídos com sucesso!")

                    # Métrica Resumo
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Produtos SH6 Analisados", len(df_res))
                    m2.metric("Produtos c/ RCA > 1", len(df_res[df_res["RCA / VCR"] > 1]))
                    m3.metric("Produtos c/ Maior Crescimento BR", len(df_res[df_res["Maior Crescimento BR"] == "Sim"]))

                    st.markdown("---")
                    st.subheader("📊 Tabela Consolidada de Indicadores")

                    # Formatação visual para a tabela do Streamlit
                    df_display = df_res.copy()
                    cols_usd = [c for c in df_display.columns if "Imp." in c]
                    cols_pct = [c for c in df_display.columns if "Share" in c or "CAGR" in c or "Variação" in c]

                    for c in cols_usd:
                        df_display[c] = df_display[c].apply(fmt_usd)
                    for c in cols_pct:
                        df_display[c] = df_display[c].apply(fmt_pct)
                    
                    df_display["RCA / VCR"] = df_display["RCA / VCR"].map("{:.2f}".format)

                    st.dataframe(df_display, width="stretch", hide_index=True)

                    # Botão de Download
                    csv_bytes = df_res.to_csv(index=False, sep=";").encode("utf-8-sig")
                    st.download_button(
                        "📥 Baixar Resultado Consolidado (CSV)",
                        data=csv_bytes,
                        file_name="un_comtrade_rca_analytics.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Não foi possível mapear as colunas do arquivo enviado. Verifique se o formato é do UN Comtrade.")
    else:
        st.info("💡 Envie o arquivo contendo os dados do UN Comtrade na barra lateral para iniciar a análise.")

if __name__ == "__main__":
    main()
