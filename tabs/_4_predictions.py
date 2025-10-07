import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px

def show():
    st.subheader("Predictions Page")
    st.markdown("Upload a new dataset and apply your trained model to get predictions.")

    # Check if model is trained
    if "trained_model" not in st.session_state:
        st.warning("Please train a model first on the ML Training page.")
        return

    model = st.session_state["trained_model"]
    ml_results = st.session_state.get("ml_results", {})
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

            # ----------------- Prepare features -----------------
            if model_type != "Clustering":
                target_col = None
                if model_type == "Classification":
                    target_col = st.selectbox(
                        "Select target column (if exists in new dataset, else skip):",
                        [None] + list(df_new.columns)
                    )

                if target_col:
                    df_features = df_new.drop(columns=[target_col])
                else:
                    df_features = df_new.copy()

                # Encode categorical columns
                X_new = pd.get_dummies(df_features, drop_first=True)

                # Align with training columns
                if "feature_columns" in st.session_state:
                    X_new = X_new.reindex(
                        columns=st.session_state["feature_columns"], fill_value=0
                    )

            else:
                # Clustering: use all features
                X_new = pd.get_dummies(df_new, drop_first=True)

            # ----------------- Predict button -----------------
            if st.button("Predict"):
                if model_type == "Clustering":
                    y_pred_new = model.predict(X_new)
                    df_new["Cluster"] = y_pred_new
                    st.write("### Predicted Clusters")
                    st.dataframe(df_new)

                    # Scatter plot for first 2 numeric columns
                    numeric_cols = df_new.select_dtypes(include=np.number).columns.tolist()
                    if len(numeric_cols) >= 2:
                        fig = px.scatter(
                            df_new,
                            x=numeric_cols[0],
                            y=numeric_cols[1],
                            color="Cluster",
                            title="Cluster Visualization"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    y_pred_new = model.predict(X_new)
                    df_new["Predictions"] = y_pred_new
                    st.write("### Predictions")
                    st.dataframe(df_new)

                    # Regression plot: predicted vs first numeric feature
                    if model_type == "Regression":
                        numeric_cols = df_new.select_dtypes(include=np.number).columns.tolist()
                        if numeric_cols:
                            fig = px.scatter(
                                x=df_new[numeric_cols[0]],
                                y=y_pred_new,
                                labels={"x": numeric_cols[0], "y": "Predicted"},
                                title="Predicted vs Feature"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # ----------------- Store predictions -----------------
                st.session_state["predictions_new"] = y_pred_new

                # ----------------- Download as CSV -----------------
                csv_buffer = io.StringIO()
                df_new.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode()
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("Upload a new dataset to generate predictions.")

