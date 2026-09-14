import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Análise de RCA e CAGR - UN Comtrade",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Calculadora de RCA e CAGR (UN Comtrade)")
st.markdown("""
Esta aplicação processa planilhas exportadas do **UN Comtrade** contendo fluxos comerciais por **SH6 (HS 6 dígitos)** e calcula:
* **RCA (Revealed Comparative Advantage / Índice de Balassa)** para cada país (*Reporter*) e produto (*SH6*).
* **RSCA (Revealed Symmetric Comparative Advantage / Índice de Laursen)**.
* **CAGR (Compound Annual Growth Rate / Taxa Composta de Crescimento Anual)** das exportações ao longo do período presente na base.
""")

st.sidebar.header("⚙️ Configurações & Upload")
uploaded_file = st.sidebar.file_uploader(
    "Envie a planilha do UN Comtrade (CSV ou XLSX)", type=["csv", "xlsx"]
)

if uploaded_file is not None:
  try:
    with st.spinner("Carregando e processando a base de dados..."):
      if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, low_memory=False)
      else:
        df = pd.read_excel(uploaded_file)

    st.sidebar.success("Arquivo carregado com sucesso!")

    cols = list(df.columns)

    # Identificação inteligente de colunas comuns da API/Download do UN Comtrade
    def find_col(candidates, default):
      for c in candidates:
        if c in cols:
          return c
      return default

    reporter_col_def = find_col(
        [
            "reporterISO",
            "reporterDesc",
            "Reporter",
            "Reporter Code",
            "reporterCode",
        ],
        cols[0],
    )
    partner_col_def = find_col(
        [
            "partnerISO",
            "partnerDesc",
            "Partner",
            "Partner Code",
            "partnerCode",
        ],
        cols[1] if len(cols) > 1 else cols[0],
    )
    cmd_col_def = find_col(
        ["cmdCode", "cmdDesc", "Commodity Code", "HSCode", "cmdCodeDescription"],
        cols[2] if len(cols) > 2 else cols[0],
    )
    year_col_def = find_col(
        ["period", "refYear", "Year", "year", "periodCode"],
        cols[3] if len(cols) > 3 else cols[0],
    )
    val_col_def = find_col(
        [
            "primaryValue",
            "tradeValue",
            "TradeValue",
            "Value",
            "Trade Value (US$)",
        ],
        cols[4] if len(cols) > 4 else cols[0],
    )

    with st.sidebar.expander("📌 Ajuste de Colunas Mapeadas", expanded=False):
      col_reporter = st.selectbox(
          "País Exportador (Reporter):",
          cols,
          index=cols.index(reporter_col_def),
      )
      col_partner = st.selectbox(
          "Parceiro Comercial (Partner):",
          cols,
          index=cols.index(partner_col_def),
      )
      col_cmd = st.selectbox(
          "Código SH6 (Commodity):", cols, index=cols.index(cmd_col_def)
      )
      col_year = st.selectbox(
          "Ano (Period/Year):", cols, index=cols.index(year_col_def)
      )
      col_val = st.selectbox(
          "Valor da Exportação (US$):", cols, index=cols.index(val_col_def)
      )

    # Tratamento de Tipos de Dados
    df[col_cmd] = df[col_cmd].astype(str).str.zfill(6)
    df[col_year] = pd.to_numeric(df[col_year], errors="coerce").astype(int)
    df[col_val] = pd.to_numeric(df[col_val], errors="coerce").fillna(0)

    # Seleção do Parceiro Global ('World') para o cálculo do RCA
    partners_available = df[col_partner].unique().tolist()
    world_default = next(
        (
            p
            for p in partners_available
            if str(p).lower() in ["world", "wld", "0", "todos", "mundo"]
        ),
        partners_available[0],
    )

    st.sidebar.subheader("🎯 Filtros para Cálculo do RCA")
    selected_partner = st.sidebar.selectbox(
        "Selecione o Parceiro (Selecione 'World' para RCA Global):",
        partners_available,
        index=partners_available.index(world_default),
    )

    years = sorted(df[col_year].unique())
    selected_year_rca = st.sidebar.selectbox(
        "Ano de Referência para o RCA:", years, index=len(years) - 1
    )

    # -------------------------------------------------------------
    # 1. CÁLCULO DO RCA & RSCA
    # -------------------------------------------------------------
    df_rca_year = df[
        (df[col_partner] == selected_partner)
        & (df[col_year] == selected_year_rca)
    ].copy()

    if df_rca_year.empty:
      st.error("Não há dados encontrados para o Parceiro e Ano selecionados.")
    else:
      # Tabela pivô: Produtos (linhas) x Países (colunas)
      pv = df_rca_year.pivot_table(
          index=col_cmd, columns=col_reporter, values=col_val, aggfunc="sum"
      ).fillna(0)

      X_ij = pv
      X_i = pv.sum(axis=0)  # Total exportado pelo País i
      X_wj = pv.sum(axis=1)  # Total exportado do Produto j no mundo
      X_w = pv.values.sum()  # Comércio global total

      # Matriz RCA = (X_ij / X_i) / (X_wj / X_w)
      share_pais = X_ij.div(X_i, axis=1)
      share_mundo = X_wj / X_w

      rca_df = share_pais.div(share_mundo, axis=0)
      rca_df = rca_df.replace([np.inf, -np.inf], np.nan).fillna(0)

      # Transformação para Formato Longo
      rca_long = rca_df.reset_index().melt(
          id_vars=col_cmd, var_name="Reporter", value_name="RCA"
      )
      rca_long["RSCA"] = (rca_long["RCA"] - 1) / (rca_long["RCA"] + 1)
      rca_long["RSCA"] = rca_long["RSCA"].fillna(-1)

      # -------------------------------------------------------------
      # 2. CÁLCULO DO CAGR
      # -------------------------------------------------------------
      start_year = min(years)
      end_year = max(years)
      n_years = end_year - start_year

      st.sidebar.subheader("📈 Janela Temporal do CAGR")
      st.sidebar.info(
          f"Período: **{start_year} a {end_year}** ({n_years} anos)"
      )

      if n_years > 0:
        df_cagr_base = (
            df[df[col_partner] == selected_partner]
            .groupby([col_reporter, col_cmd, col_year])[col_val]
            .sum()
            .reset_index()
        )
        piv_cagr = df_cagr_base.pivot_table(
            index=[col_reporter, col_cmd],
            columns=col_year,
            values=col_val,
            fill_value=0,
        )

        val_start = piv_cagr[start_year]
        val_end = piv_cagr[end_year]

        with np.errstate(divide="ignore", invalid="ignore"):
          cagr = np.where(
              (val_start > 0) & (val_end > 0),
              ((val_end / val_start) ** (1 / n_years)) - 1,
              np.nan,
          )

        piv_cagr["CAGR"] = cagr
        cagr_df = piv_cagr[["CAGR"]].reset_index()
        cagr_df.rename(
            columns={col_reporter: "Reporter", col_cmd: col_cmd}, inplace=True
        )
      else:
        cagr_df = pd.DataFrame(columns=["Reporter", col_cmd, "CAGR"])

      # Consolidação da Tabela Final
      final_df = pd.merge(
          rca_long, cagr_df, on=["Reporter", col_cmd], how="left"
      )

      # Adicionar valor absoluto exportado no ano do RCA
      val_actual = (
          df_rca_year.groupby([col_reporter, col_cmd])[col_val]
          .sum()
          .reset_index()
      )
      val_actual.rename(
          columns={col_reporter: "Reporter", col_val: "Valor_Exportado_USD"},
          inplace=True,
      )
      final_df = pd.merge(
          final_df, val_actual, on=["Reporter", col_cmd], how="left"
      ).fillna({"Valor_Exportado_USD": 0})

      final_df = final_df[
          ["Reporter", col_cmd, "Valor_Exportado_USD", "RCA", "RSCA", "CAGR"]
      ]

      # -------------------------------------------------------------
      # 3. INTERFACE DE EXIBIÇÃO E FILTROS
      # -------------------------------------------------------------
      st.header("📌 Painel de Resultados")

      col_f1, col_f2, col_f3 = st.columns(3)
      with col_f1:
        sel_reporters = st.multiselect(
            "Filtrar Países (Reporter):", sorted(final_df["Reporter"].unique())
        )
      with col_f2:
        sel_sh6 = st.multiselect(
            "Filtrar Produtos (SH6):", sorted(final_df[col_cmd].unique())
        )
      with col_f3:
        only_rca_gt_1 = st.checkbox(
            "Apenas com Vantagem Comparativa (RCA > 1)"
        )

      filtered_df = final_df.copy()
      if sel_reporters:
        filtered_df = filtered_df[filtered_df["Reporter"].isin(sel_reporters)]
      if sel_sh6:
        filtered_df = filtered_df[filtered_df[col_cmd].isin(sel_sh6)]
      if only_rca_gt_1:
        filtered_df = filtered_df[filtered_df["RCA"] > 1]

      m1, m2, m3 = st.columns(3)
      m1.metric("Linhas Exibidas", f"{len(filtered_df):,}")
      m2.metric(
          "Produtos com RCA > 1", f"{(filtered_df['RCA'] > 1).sum():,}"
      )
      cagr_mean = filtered_df["CAGR"].dropna().mean()
      m3.metric(
          "CAGR Médio do Filtro",
          f"{cagr_mean*100:.2f}%" if pd.notnull(cagr_mean) else "N/A",
      )

      st.dataframe(
          filtered_df.style.format({
              "Valor_Exportado_USD": "${:,.2f}",
              "RCA": "{:.4f}",
              "RSCA": "{:.4f}",
              "CAGR": "{:.2%}",
          }),
          use_container_width=True,
          height=480,
      )

      # Exportação de Dados
      st.subheader("📥 Exportação")
      col_d1, col_d2 = st.columns(2)

      csv_file = filtered_df.to_csv(index=False).encode("utf-8")
      col_d1.download_button(
          label="📄 Baixar Tabela em CSV",
          data=csv_file,
          file_name=f"RCA_CAGR_{selected_year_rca}.csv",
          mime="text/csv",
      )

      excel_buffer = io.BytesIO()
      with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="RCA_CAGR")
      col_d2.download_button(
          label="📊 Baixar Tabela em Excel (.xlsx)",
          data=excel_buffer.getvalue(),
          file_name=f"RCA_CAGR_{selected_year_rca}.xlsx",
          mime=(
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          ),
      )

  except Exception as e:
    st.error(f"Erro no processamento do arquivo: {str(e)}")
else:
  st.info(
      "👈 Envie sua planilha do UN Comtrade através da barra lateral para iniciar."
  )
