# Autolytica – AI-Powered Business Analytics Platform

**Autolytica** is an end-to-end, interactive **Streamlit-based machine learning application** that automates the entire analytics pipeline — from data upload and EDA to model training, prediction, and AI-powered report generation.


## Features

### 1️⃣ Data Upload & Exploration
- Upload CSV or Excel files.
- Automatic display of dataset preview and structure.
- Interactive Exploratory Data Analysis (EDA) — summary stats, missing values, and visual charts.

### 2️⃣ ML Training
- Train ML models dynamically:
  - Regression
  - Classification
  - Clustering
- Auto-encodes categorical variables, splits data, trains models, and shows metrics & visualizations.
- Stores trained models and results in session state for further use.

### 3️⃣ Predictions
- Upload a new dataset and apply any trained model.
- Supports Regression, Classification, and Clustering predictions.
- Download predictions as CSV.
- Auto plots visualizations for model outputs.

### 4️⃣ Automated Business Reports
- Uses **Google Gemini API** via **LangChain** to generate human-readable reports.
- Combines EDA, model performance, and predictions into structured insights.
- Exports report as a formatted PDF with embedded charts.

### 5️⃣ Chat Agent
- Intelligent conversational agent built using **LangChain + Gemini**.
- Answers natural-language business questions like:
  - “Which feature is most important for predicting churn?”
  - “Show me the top 3 customer segments.”
  - “Which model is used here?”
- Uses stored data, metrics, and charts to give contextual responses.

## Tech Stack

- **Frontend/UI:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Plotly, Matplotlib, Seaborn  
- **AI/LLM Integration:** LangChain + Google Gemini API  
- **Reporting:** FPDF  



