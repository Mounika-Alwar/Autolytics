import streamlit as st 
import pathlib as Path 

st.set_page_config(page_title = 'Autolytica',layout="wide")

st.markdown("""
<h1 style='text-align: center;'>🧠 Agentic AI-Powered Autonomous Business Analyst</h1>
<h3 style='text-align: center; color: gray;'>A Next Generation Framework for Intelligent Business Automation</h3>
""", 
unsafe_allow_html=True)
st.write("""
This is your autonomous business analyst agent.
Upload any dataset and explore insights, perform ML analysis, generate reports, and interact with chat agent to get actionable recommendations.
""")

st.markdown("""
<style>
/* Make tab container full width */
.css-1v0mbdj.e1fqkh3o3 { 
    font-size: 50px !important;
    font-weight: 1000 !important;
    color: #0F213E !important;
}

/* Make all tabs flex evenly */
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: space-between !important;
}
</style>
""", unsafe_allow_html=True)


tabs = st.tabs(["Home", "EDA", "ML Training", "Predictions", "Reports", "Chat Agent"])

with tabs[0]:
    from tabs import _1_home
    _1_home.show()

with tabs[1]:
    from tabs import _2_eda
    _2_eda.show()

with tabs[2]:
    from tabs import _3_ml_training
    _3_ml_training.show()

with tabs[3]:
    from tabs import _4_predictions
    _4_predictions.show()

with tabs[4]:
    from tabs import _5_reports
    _5_reports.show()

with tabs[5]:
    from tabs import _6_chatagent
    _6_chatagent.show()