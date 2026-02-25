import streamlit as st
import pandas as pd
import plotly.express as px

from agents import get_or_create_agents


def try_generate_chart(df: pd.DataFrame, question: str) -> str:
    """
    Best-effort helper to render quick visualizations based on the user's question.
    This stays in the UI layer while analytical context comes from the agents.
    """
    try:
        if "distribution" in question.lower():
            col = df.select_dtypes(include=["number"]).columns[0]
            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
            return f"Displayed distribution chart for {col}."
        if "correlation" in question.lower():
            corr = df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            return "Displayed correlation heatmap."
    except Exception as e:
        return f"Chart generation skipped: {e}"
    return ""


def show():
    st.subheader(" Chat Agent")
    st.write("Ask the AI Analyst anything about your dataset, EDA, or model.")

    if "dataset" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first from the Upload page.")
        return

    # Initialize agents and sync with current session data
    business_user, analyst_agent, knowledge_agent = get_or_create_agents(st.session_state)

    df = st.session_state["dataset"]
    analyst_agent.dataset = df
    business_user.uploaded_dataset = df

    # Let the analyst agent know about existing analysis results if present
    if "eda_results" in st.session_state:
        analyst_agent.analysis_results["eda"] = st.session_state["eda_results"]
    if "ml_results" in st.session_state:
        analyst_agent.analysis_results["ml"] = st.session_state["ml_results"]
    if "predictions_new" in st.session_state and isinstance(
        st.session_state["predictions_new"], (pd.DataFrame, pd.Series)
    ):
        preds_new = st.session_state["predictions_new"]
        analyst_agent.analysis_results["predictions_new"] = {
            "shape": list(preds_new.shape),
            "preview": str(getattr(preds_new, "head", lambda: preds_new)()),
        }

    user_question = st.text_area(
        "Your question:", placeholder="E.g., Which feature is most correlated with churn?"
    )

    if st.button("Ask Agent"):
        if not user_question.strip():
            st.info("Please enter a question first.")
            return

        with st.spinner("Analyzing your question..."):
            try:
                # BusinessUser integrates with AIAnalystAgent and KnowledgeAgent
                answer = business_user.askQuestion(
                    question=user_question,
                    analyst=analyst_agent,
                    knowledge_agent=knowledge_agent,
                )
            except Exception as e:
                st.error(f"Agent error: {e}")
                return

        # Display response
        st.markdown("### 💬 Agent’s Response")
        st.write(answer)

        # Try charts if relevant
        msg = try_generate_chart(df, user_question)
        if msg:
            st.info(msg)

    # Sidebar chat history from KnowledgeAgent's memory
    with st.sidebar.expander("🕒 Chat History", expanded=False):
        messages = getattr(knowledge_agent.chat_history, "chat_memory", None)
        if messages and messages.messages:
            for msg in messages.messages:
                if msg.type == "human":
                    st.markdown(f"🧍‍♀️ **You:** {msg.content}")
                else:
                    st.markdown(f"🤖 **Agent:** {msg.content}")
        else:
            st.write("No chat history yet.")

    # Clear chat option
    if st.sidebar.button("🧹 Clear Chat"):
        knowledge_agent.chat_history.clear()
        st.sidebar.success("Chat cleared!")


if __name__ == "__main__":
    show()

import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import io

load_dotenv()

llm = ChatGoogleGenerativeAI(
      model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ✅ Initialize memory (expects input key 'input' to avoid errors)
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="history", input_key="input", return_messages=True
    )

# ✅ Prompt Template with single variable (we’ll merge context + question)
prompt_template = PromptTemplate(
    input_variables=["input"],
    template="""
You are a professional data analysis assistant working in a platform called Autolytica.

Below is all the available context and the user’s question. Use it to provide a rich, 
business-analyst-style answer. If relevant, include bullet points, tables, or short insights.

{input}

Your answer should be:
- Clear and concise (avoid markdown like ***)
- Written like a real analyst speaking to a manager
- Include key numbers, metrics, and conclusions
"""
)

# ✅ Build LangChain chain
chain = LLMChain(llm=llm, prompt=prompt_template, memory=st.session_state.chat_memory)


# ✅ Helper: Combine session info
def collect_context():
    context_parts = []

    # Dataset
    if "dataset" in st.session_state and isinstance(st.session_state["dataset"], pd.DataFrame):
        df = st.session_state["dataset"]
        context_parts.append(
            f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {list(df.columns)}"
        )

    # EDA results
    if "eda_results" in st.session_state:
        context_parts.append(f"EDA Summary: {st.session_state['eda_results']}")

    # ML results (model info, type, parameters)
    if "ml_results" in st.session_state:
        ml = st.session_state["ml_results"]
        model_type = ml.get("type", "Unknown")
        model_name = ml.get("model", "Unknown")
        params = ml.get("params", {})
        context_parts.append(
            f"Machine Learning Model: {model_name} ({model_type}) with parameters {params}"
        )

    # Model metrics
    if "model_metrics" in st.session_state:
        context_parts.append(f"Model Metrics: {st.session_state['model_metrics']}")

    # Predictions
    if "predictions" in st.session_state:
        preds = st.session_state["predictions"]
        if isinstance(preds, pd.DataFrame):
            context_parts.append(f"Predictions Preview: {preds.head(5).to_dict()}")
        else:
            context_parts.append(f"Predictions Summary: {str(preds)[:300]}")

    # New dataset predictions (optional)
    if "predictions_new" in st.session_state:
        preds_new = st.session_state["predictions_new"]
        if isinstance(preds_new, pd.DataFrame):
            context_parts.append(f"New Predictions Preview: {preds_new.head(5).to_dict()}")
        else:
            context_parts.append(f"New Predictions Summary: {str(preds_new)[:300]}")

    return "\n\n".join(context_parts)


# ✅ Try making a chart if user requests one
def try_generate_chart(df, question):
    try:
        if "distribution" in question.lower():
            col = df.select_dtypes(include=["number"]).columns[0]
            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)
            return f"Displayed distribution chart for {col}."
        elif "correlation" in question.lower():
            corr = df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            return "Displayed correlation heatmap."
    except Exception as e:
        return f"Chart generation skipped: {e}"
    return ""


# ✅ Main entry function
def show():
    st.subheader(" Chat Agent")
    st.write("Ask the AI Analyst anything about your dataset, EDA, or model.")

    if "dataset" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first from the Upload page.")
        return

    user_question = st.text_area("Your question:", placeholder="E.g., Which feature is most correlated with churn?")

    if st.button("Ask Agent"):
        if not user_question.strip():
            st.info("Please enter a question first.")
            return

        # Combine question + context into one 'input'
        context = collect_context()
        full_input = f"User Question: {user_question}\n\nContext:\n{context}"

        # Run chain safely
        with st.spinner("Analyzing your question..."):
            try:
                answer = chain.run(input=full_input)
            except Exception as e:
                st.error(f"Agent error: {e}")
                return

        # Display response
        st.markdown("### 💬 Agent’s Response")
        st.write(answer)

        # Try charts if relevant
        df = st.session_state.get("dataset")
        msg = try_generate_chart(df, user_question)
        if msg:
            st.info(msg)

    # Sidebar chat history
    with st.sidebar.expander("🕒 Chat History", expanded=False):
        if st.session_state.chat_memory.chat_memory.messages:
            for msg in st.session_state.chat_memory.chat_memory.messages:
                if msg.type == "human":
                    st.markdown(f"🧍‍♀️ **You:** {msg.content}")
                else:
                    st.markdown(f"🤖 **Agent:** {msg.content}")
        else:
            st.write("No chat history yet.")

    # Clear chat option
    if st.sidebar.button("🧹 Clear Chat"):
        st.session_state.chat_memory.clear()
        st.sidebar.success("Chat cleared!")


if __name__ == "__main__":
    show()