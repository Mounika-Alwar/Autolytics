import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from agents import get_or_create_agents


def show():
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown(
        """
    Explore your dataset visually. Select columns, choose plot types, and uncover relationships between features.
    """
    )

    if "dataset" not in st.session_state:
        st.warning("⚠️ Please upload a dataset in the **Home** page first.")
        return

    # Get shared agents and current dataset
    business_user, analyst_agent, _ = get_or_create_agents(st.session_state)
    df = st.session_state["dataset"]
    analyst_agent.dataset = df
    business_user.uploaded_dataset = df

    st.markdown("#### Column Selection")
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    all_cols = df.columns.tolist()

    plot_type = st.selectbox(
        "Choose plot type:",
        ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Value Counts (Bar)"],
    )

    st.divider()

    if plot_type == "Histogram":
        col = st.selectbox("Select column: ", numeric_cols)
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Histogram of {col}",
            color_discrete_sequence=["#0083B8"],
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box Plot":
        col = st.selectbox("Select column:", numeric_cols)
        fig = px.box(df, y=col, title=f"Boxplot of {col}", color_discrete_sequence=["#EF553B"])
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter Plot":
        col_x = st.selectbox("Select X-axis:", numeric_cols, key="scatter_x")
        col_y = st.selectbox("Select Y-axis:", numeric_cols, key="scatter_y")
        fig = px.scatter(
            df,
            x=col_x,
            y=col_y,
            title=f"{col_x} vs {col_y}",
            color_discrete_sequence=["#00CC96"],
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        st.markdown("##### Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            colorscale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Value Counts (Bar)":
        col = st.selectbox("Select column:", all_cols)
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "Count"]
        fig = px.bar(
            vc,
            x=col,
            y="Count",
            title=f"Value Counts for {col}",
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Delegate to AIAnalystAgent to store an EDA summary
    eda_summary = analyst_agent.performEDA(df, plot_type=plot_type)

    # Keep the original session key so other components remain compatible
    st.session_state["eda_results"] = eda_summary

    st.success("✅ EDA completed successfully!")
