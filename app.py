import streamlit as st
from logic import run_analysis

st.set_page_config(page_title="Weight Discrepancy Checker")

st.title("📦 Weight Discrepancy Checker")
st.write("Sube los PDFs del shipment (1 GR + 1 o más Invoices).")

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
                uploaded = {f.name: f.read() for f in uploaded_files}

                # 🔴 AQUÍ se crean summary, df_full, etc.
                summary, df_full, df_adjusted, validation_df = run_analysis(uploaded)

                st.success("✅ Análisis completado")

                # ===============================
                # 📊 TOTALES
                # ===============================
                st.subheader("📊 Totales del shipment")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "📦 Total Invoice (kg)",
                        round(summary["Invoice total (kg)"].iloc[0], 2)
                    )

                with col2:
                    st.metric(
                        "🏭 Total GR (kg)",
                        round(summary["GR total (kg)"].iloc[0], 2)
                    )

                # ===============================
                # ⚖️ TOLERANCIA ±10%
                # ===============================
                st.subheader("⚖️ Validación de tolerancia (±10%)")

                in_before = summary["In tolerance BEFORE"].iloc[0]
                in_after = summary["In tolerance AFTER"].iloc[0]

                col1, col2 = st.columns(2)

                with col1:
                    if in_before:
                        st.success("🟢 BEFORE: IN TOLERANCE")
                    else:
                        st.error("🔴 BEFORE: OUT OF TOLERANCE")

                with col2:
                    if in_after:
                        st.success("🟢 AFTER: IN TOLERANCE")
                    else:
                        st.error("🔴 AFTER: OUT OF TOLERANCE")

                st.caption(
                    f"📏 Rango permitido según GR (±10%): "
                    f"{summary['Allowed low (kg)'].iloc[0]} kg → "
                    f"{summary['Allowed high (kg)'].iloc[0]} kg"
                )

                # ===============================
                # 📦 TABLAS
                # ===============================
                st.subheader("📦 Tabla CAT completa")
                st.dataframe(df_full, use_container_width=True)

                st.subheader("📦 Solo piezas ajustadas")
                st.dataframe(df_adjusted, use_container_width=True)

                st.subheader("📊 Validación Invoice vs GR")
                st.dataframe(validation_df, use_container_width=True)

            except Exception as e:
                st.error("❌ Error durante el análisis")
                st.exception(e)

