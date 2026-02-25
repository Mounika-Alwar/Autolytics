import streamlit as st
import pandas as pd

from agents import get_or_create_agents


def show():
    st.subheader("Home")
    st.markdown(
        """
    Upload the dataset here and get an overview of the **Quick stats** of the dataset.
    """
    )

    # Initialize core agents (BusinessUser, AIAnalystAgent, KnowledgeAgent)
    business_user, analyst_agent, _ = get_or_create_agents(st.session_state)

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # BusinessUser uploads dataset and AIAnalystAgent keeps reference
            df = business_user.uploadDataset(df)
            analyst_agent.dataset = df

            # Keep backward-compatible session key for other tabs
            st.session_state["dataset"] = df

            st.success("✅ Dataset uploaded successfully!")

            st.markdown("### Quick Stats")

            st.markdown("#### Dataset Shape")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

            num_cols = st.slider(
                "Select column range to view: ",
                min_value=1,
                max_value=df.shape[1],
                value=(1, min(5, df.shape[1])),
            )
            num_rows = st.slider(
                "Select row range to view: ",
                min_value=1,
                max_value=df.shape[0],
                value=(1, min(5, df.shape[0])),
            )

            st.markdown("#### Data Preview")
            st.dataframe(df.iloc[num_rows[0] - 1 : num_rows[1], num_cols[0] - 1 : num_cols[1]])

            st.markdown("#### Missing Values")
            st.dataframe(df.isna().sum().to_frame("Missing Values"))

            st.markdown("#### Data Types")
            st.dataframe(df.dtypes.astype(str).to_frame("Data Type"))

            st.markdown("#### Descriptive Statistics")
            st.dataframe(df.describe().T)

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Upload a dataset to view quick statistics and start your analysis journey.")
