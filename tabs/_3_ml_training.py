import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from agents import get_or_create_agents


def show():
    st.subheader("ML Model Training")
    st.markdown("Train ML models dynamically on your uploaded dataset.")

    if "dataset" not in st.session_state:
        st.warning("Please upload a dataset on the Home page first.")
        return

    # Get shared agents and dataset
    _, analyst_agent, _ = get_or_create_agents(st.session_state)
    df = st.session_state["dataset"]
    analyst_agent.dataset = df

    st.markdown("#### Choose Problem Type")
    model_type = st.selectbox(
        "Select task type:",
        ["Regression", "Classification", "Clustering"],
    )

    target_col = None
    if model_type != "Clustering":
        target_col = st.selectbox("Select target variable:", df.columns)

    st.markdown("#### Choose Model")
    if model_type == "Regression":
        model_choice = st.selectbox(
            "Select regression model:",
            ["Linear Regression", "Random Forest Regressor"],
        )
    elif model_type == "Classification":
        model_choice = st.selectbox(
            "Select classification model:",
            ["Logistic Regression", "Random Forest Classifier"],
        )
    else:
        model_choice = st.selectbox("Select clustering model:", ["KMeans"])

    st.markdown("#### Hyperparameters")
    params: dict = {}
    if "Random Forest" in model_choice:
        params["n_estimators"] = st.slider("Number of trees:", 10, 300, 100)
        params["max_depth"] = st.slider("Max depth:", 1, 20, 5)
    elif model_choice == "KMeans":
        params["n_clusters"] = st.slider("Number of clusters:", 2, 10, 3)

    if st.button("Train Model"):
        try:
            progress = st.progress(0)
            st.write("Preparing data...")
            progress.progress(30)

            st.write("Training model with AIAnalystAgent...")
            progress.progress(60)

            # Delegate training logic to AIAnalystAgent
            result = analyst_agent.trainModel(
                df=df,
                model_type=model_type,
                model_choice=model_choice,
                target_col=target_col,
                params=params,
            )

            progress.progress(90)
            st.success("✅ Training Complete!")

            # ----------------- Model Performance -----------------
            st.markdown("#### Model Performance")
            metrics = result["metrics"]
            y_test = result["y_test"]
            y_pred = result["y_pred"]

            if model_type == "Regression":
                mse = metrics.get("mse")
                rmse = metrics.get("rmse")
                r2 = metrics.get("r2")
                st.write(f"**MSE:** {mse:.4f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")

                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Predicted vs Actual",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif model_type == "Classification":
                acc = metrics.get("accuracy")
                f1 = metrics.get("f1_score")
                cm = np.array(metrics.get("confusion_matrix"))
                st.write(f"**Accuracy:** {acc:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")
                st.write("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(cm))

            else:  # Clustering
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                y_pred = result["y_pred"]
                if len(numeric_cols) >= 2:
                    silhouette = metrics.get("silhouette_score")
                    if silhouette is not None:
                        st.write(f"**Silhouette Score:** {silhouette:.4f}")
                    df_clustered = df.copy()
                    df_clustered["Cluster"] = y_pred
                    cluster_fig = px.scatter(
                        df_clustered,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        color=df_clustered["Cluster"].astype(str),
                        title="Cluster Visualization",
                    )
                    st.plotly_chart(cluster_fig, use_container_width=True)
                else:
                    st.info("Not enough numeric columns for cluster visualization.")

            # ----------------- Store results in session for backward compatibility -----------------
            st.session_state["trained_model"] = result["model"]
            st.session_state["predictions"] = result["y_pred"]
            st.session_state["ml_results"] = result["ml_results"]
            if result.get("feature_columns") is not None:
                st.session_state["feature_columns"] = result["feature_columns"]

            progress.progress(100)

        except Exception as e:
            st.error(f"Error during training: {e}")
