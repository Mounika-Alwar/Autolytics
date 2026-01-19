import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
from dotenv import load_dotenv
import os

# LangChain + Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# ----------------- Initialize Gemini LLM -----------------
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",google_api_key=os.getenv("GEMINI_API_KEY"))  # Use Gemini model

# ----------------- Function to generate report using LLM -----------------
def generate_report(eda_summary, model_metrics, predictions, chart_paths):
    charts_text = "\n".join([f"![Chart]({p})" for p in chart_paths]) if chart_paths else "No charts."
    
    prompt = PromptTemplate(
        input_variables=["eda_summary", "model_metrics", "predictions", "charts"],
        template="""
        # Business Report

        ## Exploratory Data Analysis Summary
        {eda_summary}

        ## Model Metrics
        {model_metrics}

        ## Predictions
        {predictions}

        ## Charts
        {charts}

        ## Recommendations
        Based on the analysis, provide actionable insights and recommendations for business decisions.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        eda_summary=eda_summary,
        model_metrics=model_metrics,
        predictions=predictions,
        charts=charts_text
    )

# ----------------- PDF Export Function -----------------
def export_to_pdf(report_text, chart_paths=[]):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add text
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 8, line)
    
    # Add charts
    y_position = pdf.get_y() + 10
    for chart in chart_paths:
        try:
            pdf.image(chart, x=10, y=y_position, w=180)
            y_position += 100
        except:
            pass
    
    pdf_output = "business_report.pdf"
    pdf.output(pdf_output)
    with open(pdf_output, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# ----------------- Streamlit Show Function -----------------
def show():
    st.subheader("📄 Business Report Generation")
    st.markdown("Generate a comprehensive business report using the agent (Gemini + LangChain).")

    # Check if necessary data exists
    if "dataset" not in st.session_state:
        st.warning("Upload a dataset on the Home page first.")
        return
    if "ml_results" not in st.session_state:
        st.warning("Train a model first on the ML Training page.")
        return

    df = st.session_state["dataset"]
    ml_results = st.session_state.get("ml_results", {})
    predictions = st.session_state.get("predictions", None)

    # Optional: User can write EDA summary manually or use placeholder
    eda_summary = st.text_area(
        "EDA Summary (optional, auto-generated if left blank):",
        value="Dataset contains basic statistics, missing values, and feature distributions."
    )

    model_metrics = st.text_area(
        "Model Metrics (optional, auto-generated if left blank):",
        value=str(ml_results)
    )

    # Generate simple charts
    chart_paths = []
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        for col in numeric_cols[:2]:  # Take first 2 numeric columns as example
            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            chart_file = f"chart_{col}.png"
            fig.write_image(chart_file)
            chart_paths.append(chart_file)

    if st.button("Generate Report"):
        with st.spinner("Generating report using Gemini model..."):
            report_text = generate_report(
                eda_summary=eda_summary,
                model_metrics=model_metrics,
                predictions=str(predictions),
                chart_paths=chart_paths
            )
            st.success("✅ Report Generated!")
            st.text_area("Generated Report:", value=report_text, height=400)

            # ----------------- Download Buttons -----------------
            # PDF
            pdf_bytes = export_to_pdf(report_text, chart_paths)
            st.download_button(
                label="Download Report as PDF",
                data=pdf_bytes,
                file_name="business_report.pdf",
                mime="application/pdf"
            )

            # HTML
            st.download_button(
                label="Download Report as HTML",
                data=report_text.encode(),
                file_name="business_report.html",
                mime="text/html"
            )
