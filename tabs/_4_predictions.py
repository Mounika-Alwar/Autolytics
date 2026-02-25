import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

from agents import get_or_create_agents


def show():
    st.subheader("Predictions Page")
    st.markdown("Upload a new dataset and apply your trained model to get predictions.")

    # Check if model is trained
    if "trained_model" not in st.session_state:
        st.warning("Please train a model first on the ML Training page.")
        return

    # Sync analyst agent with trained model and metadata
    _, analyst_agent, _ = get_or_create_agents(st.session_state)
    analyst_agent.trained_models["latest"] = st.session_state["trained_model"]
    if "ml_results" in st.session_state:
        analyst_agent.analysis_results["ml"] = st.session_state["ml_results"]

    ml_results = analyst_agent.analysis_results.get("ml", {})
    model_type = ml_results.get("type", "Regression")

    # ----------------- File uploader -----------------
    uploaded_file = st.file_uploader(
        "Upload new dataset (CSV or Excel)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Read uploaded dataset
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

            st.success("✅ Dataset uploaded successfully!")
            st.dataframe(df_new.head())

            # ----------------- Optional target for classification -----------------
            if model_type == "Classification":
                target_col = st.selectbox(
                    "Select target column (if exists in new dataset, else skip):",
                    [None] + list(df_new.columns),
                )
                if target_col:
                    df_new = df_new.drop(columns=[target_col])

            # ----------------- Predict button -----------------
            if st.button("Predict"):
                result = analyst_agent.generatePrediction(df_new)
                y_pred_new = result["predictions"]
                df_result = result["result_frame"]

                if model_type == "Clustering":
                    st.write("### Predicted Clusters")
                    st.dataframe(df_result)

                    # Scatter plot for first 2 numeric columns
                    numeric_cols = df_result.select_dtypes(include=np.number).columns.tolist()
                    if "Cluster" in df_result.columns and len(numeric_cols) >= 2:
                        fig = px.scatter(
                            df_result,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color="Cluster",
                            title="Cluster Visualization",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("### Predictions")
                    st.dataframe(df_result)

                    # Regression plot: predicted vs first numeric feature
                    if model_type == "Regression":
                        numeric_cols = df_result.select_dtypes(include=np.number).columns.tolist()
                        if numeric_cols:
                            fig = px.scatter(
                                x=df_result[numeric_cols[0]],
                                y=y_pred_new,
                                labels={"x": numeric_cols[0], "y": "Predicted"},
                                title="Predicted vs Feature",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # ----------------- Store predictions -----------------
                st.session_state["predictions_new"] = y_pred_new
                analyst_agent.analysis_results["predictions_new"] = {
                    "shape": list(df_result.shape),
                    "columns": list(df_result.columns),
                }

                # ----------------- Download as CSV -----------------
                csv_buffer = io.StringIO()
                df_result.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode()
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("Upload a new dataset to generate predictions.")

