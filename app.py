import streamlit as st
from logic import run_analysis

st.set_page_config(page_title="Weight Discrepancy Checker")

st.title("📦 Weight Discrepancy Checker")
st.write(
    "Sube los PDFs del shipment (1 GR + 1 o más Invoices). "
    "El sistema hará el chequeo de discrepancias automáticamente."
)

uploaded_files = st.file_uploader(
    "Sube los archivos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🔍 Ejecutar análisis"):
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Debes subir mínimo 2 PDFs (1 GR + 1 Invoice).")
    else:
        with st.spinner("Procesando shipment..."):
            try:
                uploaded = {
                    f.name: f.read() for f in uploaded_files
                }

                summary, df_full, df_adjusted, validation_df = run_analysis(uploaded)

                st.success("✅ Análisis completado")

                st.subheader("📊 Resumen del shipment")
                st.dataframe(summary)

                st.subheader("📦 Tabla CAT completa")
                st.dataframe(df_full)

                st.subheader("📦 Solo piezas ajustadas")
                st.dataframe(df_adjusted)

                st.subheader("📊 Validación Invoice vs GR")
                st.dataframe(validation_df)

            except Exception as e:
                st.error("Ocurrió un error durante el análisis")
                st.exception(e)
st.subheader("📊 Totales del shipment")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="📦 Total Invoice (kg)",
        value=round(summary["Invoice total (kg)"].iloc[0], 2)
    )

with col2:
    st.metric(
        label="🏭 Total GR (kg)",
        value=round(summary["GR total (kg)"].iloc[0], 2)
    )

