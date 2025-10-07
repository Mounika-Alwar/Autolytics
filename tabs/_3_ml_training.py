import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    f1_score, confusion_matrix, silhouette_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

def show():
    st.subheader("ML Model Training")
    st.markdown("Train ML models dynamically on your uploaded dataset.")

    if 'dataset' not in st.session_state:
        st.warning("Please upload a dataset on the Home page first.")
        return

    df = st.session_state['dataset']

    st.markdown('#### Choose Problem Type')
    model_type = st.selectbox(
        "Select task type:",
        ['Regression', 'Classification', 'Clustering']
    )

    target_col = None
    if model_type != 'Clustering':
        target_col = st.selectbox('Select target variable:', df.columns)

    st.markdown("#### Choose Model")
    if model_type == 'Regression':
        model_choice = st.selectbox("Select regression model:", ['Linear Regression', 'Random Forest Regressor'])
    elif model_type == 'Classification':
        model_choice = st.selectbox("Select classification model:", ['Logistic Regression', 'Random Forest Classifier'])
    else:
        model_choice = st.selectbox("Select clustering model:", ['KMeans'])

    st.markdown("#### Hyperparameters")
    params = {}
    if "Random Forest" in model_choice:
        params["n_estimators"] = st.slider("Number of trees:", 10, 300, 100)
        params["max_depth"] = st.slider("Max depth:", 1, 20, 5)
    elif model_choice == "KMeans":
        params["n_clusters"] = st.slider("Number of clusters:", 2, 10, 3)

    if st.button("Train Model"):
        try:
            progress = st.progress(0)
            st.write("Preparing data...")

            # ----------------- Encode categorical columns automatically -----------------
            
            if model_type != 'Clustering':
                y = df[target_col]
                X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
                

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

            progress.progress(30)

            # ----------------- Initialize model -----------------
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor(**params, random_state=42)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Random Forest Classifier":
                model = RandomForestClassifier(**params, random_state=42)
            elif model_choice == "KMeans":
                model = KMeans(**params, random_state=42)

            st.write("Training model...")
            progress.progress(60)

            # ----------------- Train model -----------------
            if model_type == "Clustering":
                y_pred = model.fit_predict(df)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            progress.progress(90)
            st.success("✅ Training Complete!")

            # ----------------- Model Performance -----------------
            st.markdown("#### Model Performance")

            if model_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**MSE:** {mse:.4f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")

                fig = px.scatter(
                    x=y_test, y=y_pred,
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title="Predicted vs Actual"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif model_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                cm = confusion_matrix(y_test, y_pred)
                st.write(f"**Accuracy:** {acc:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")
                st.write("**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(cm))

            else:  # Clustering
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) >= 2:
                    silhouette = silhouette_score(df, y_pred)
                    st.write(f"**Silhouette Score:** {silhouette:.4f}")
                    df["Cluster"] = y_pred
                    cluster_fig = px.scatter(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        color=df["Cluster"].astype(str),
                        title="Cluster Visualization"
                    )
                    st.plotly_chart(cluster_fig, use_container_width=True)
                else:
                    st.info("Not enough numeric columns for cluster visualization.")

            # ----------------- Store results -----------------
            st.session_state["trained_model"] = model
            st.session_state["predictions"] = y_pred
            st.session_state["ml_results"] = {
                "type": model_type,
                "model": model_choice,
                "params": params
            }

            progress.progress(100)

        except Exception as e:
            st.error(f"Error during training: {e}")
