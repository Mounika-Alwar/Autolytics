from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Ensure .env is loaded so GEMINI_API_KEY is available when agents are created
load_dotenv()


@dataclass
class AIAnalystAgent:
    """
    Core analytical agent responsible for EDA, model training and predictions.
    """

    dataset: Optional[pd.DataFrame] = None
    trained_models: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)

    def performEDA(self, df: pd.DataFrame, plot_type: str) -> Dict[str, Any]:
        """
        Store dataset reference and a lightweight EDA summary.
        The actual plotting is handled by the Streamlit UI.
        """
        self.dataset = df

        summary = {
            "plot_type": plot_type,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": list(df.columns),
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.analysis_results["eda"] = summary
        return summary

    def trainModel(
        self,
        df: pd.DataFrame,
        model_type: str,
        model_choice: str,
        target_col: Optional[str],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train a model on the provided dataframe and return metrics
        plus artifacts needed by the UI.
        """
        if params is None:
            params = {}

        self.dataset = df

        if model_type not in {"Regression", "Classification", "Clustering"}:
            raise ValueError(f"Unsupported model_type: {model_type}")

        X_train = X_test = y_train = y_test = None
        feature_columns: Optional[List[str]] = None

        if model_type != "Clustering":
            if target_col is None:
                raise ValueError("target_col must be provided for regression or classification.")
            y = df[target_col]
            X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
            feature_columns = list(X.columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        else:
            # For clustering we use all numeric/categorical features encoded.
            X = pd.get_dummies(df, drop_first=True)
            feature_columns = list(X.columns)

        # Initialize model
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
        else:
            raise ValueError(f"Unsupported model_choice: {model_choice}")

        # Fit and predict
        if model_type == "Clustering":
            y_pred = model.fit_predict(X)
            metrics: Dict[str, Any] = {}
            if X.shape[1] >= 2:
                try:
                    metrics["silhouette_score"] = float(silhouette_score(X, y_pred))
                except Exception:
                    metrics["silhouette_score"] = None
            y_test = None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if model_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                r2 = r2_score(y_test, y_pred)
                metrics = {
                    "mse": float(mse),
                    "rmse": rmse,
                    "r2": float(r2),
                }
            else:  # Classification
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                cm = confusion_matrix(y_test, y_pred)
                metrics = {
                    "accuracy": float(acc),
                    "f1_score": float(f1),
                    "confusion_matrix": cm.tolist(),
                }

        self.trained_models["latest"] = model

        ml_results = {
            "type": model_type,
            "model": model_choice,
            "params": params,
            "metrics": metrics,
            "target_col": target_col,
            "feature_columns": feature_columns,
        }

        self.analysis_results["ml"] = ml_results
        self.analysis_results["predictions_train"] = {
            "y_pred_sample": list(map(lambda v: float(v), np.array(y_pred).ravel()[:20]))
        }

        return {
            "model": model,
            "model_type": model_type,
            "model_choice": model_choice,
            "params": params,
            "y_test": y_test,
            "y_pred": y_pred,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "ml_results": ml_results,
        }

    def generatePrediction(
        self,
        df_new: pd.DataFrame,
        model_key: str = "latest",
    ) -> Dict[str, Any]:
        """
        Use the latest trained model to generate predictions for a new dataset.
        """
        if model_key not in self.trained_models:
            raise ValueError("No trained model available. Please train a model first.")

        model = self.trained_models[model_key]
        ml_info = self.analysis_results.get("ml", {})
        model_type = ml_info.get("type", "Regression")
        feature_columns: Optional[List[str]] = ml_info.get("feature_columns")

        if model_type == "Clustering":
            X_new = pd.get_dummies(df_new, drop_first=True)
            y_pred_new = model.predict(X_new)
            df_with_predictions = df_new.copy()
            df_with_predictions["Cluster"] = y_pred_new
        else:
            df_features = df_new.copy()
            X_new = pd.get_dummies(df_features, drop_first=True)

            if feature_columns is not None:
                X_new = X_new.reindex(columns=feature_columns, fill_value=0)

            y_pred_new = model.predict(X_new)
            df_with_predictions = df_new.copy()
            df_with_predictions["Predictions"] = y_pred_new

        self.analysis_results["predictions_new"] = {
            "shape": list(df_with_predictions.shape),
            "columns": list(df_with_predictions.columns),
        }

        return {
            "model_type": model_type,
            "predictions": y_pred_new,
            "result_frame": df_with_predictions,
        }

    def generateInsights(self, knowledge_agent: "KnowledgeAgent") -> str:
        """
        Ask the knowledge agent to generate a high-level insight summary
        based on the current analysis results.
        """
        return knowledge_agent.explainInsight(self)


@dataclass
class KnowledgeAgent:
    """
    LLM-powered knowledge layer used for contextual QA and insight generation.
    """

    vector_store: Dict[str, Any] = field(default_factory=dict)
    chat_history: ConversationBufferMemory = field(default=None)
    llm_model: Any = field(default=None)

    def __post_init__(self) -> None:
        # Always ensure we have an in-memory chat history
        if self.chat_history is None:
            self.chat_history = ConversationBufferMemory(
                memory_key="history", input_key="input", return_messages=True
            )

        # Build the shared prompt used by this agent
        self._prompt = PromptTemplate(
            input_variables=["input"],
            template=(
                "You are a professional data analysis assistant working in a "
                "platform called Autolytica.\n\n"
                "Below is all the available context and the user's question. "
                "Use it to provide a rich, business-analyst-style answer. "
                "If relevant, include bullet points, tables, or short insights.\n\n"
                "{input}\n\n"
                "Your answer should be:\n"
                "- Clear and concise (avoid markdown like ***)\n"
                "- Written like a real analyst speaking to a manager\n"
                "- Include key numbers, metrics, and conclusions\n"
            ),
        )

        # Lazily build the LLM chain only when a valid API key is available.
        # This avoids triggering Google ADC (and the DefaultCredentialsError)
        # when GEMINI_API_KEY is not configured.
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            if self.llm_model is None:
                self.llm_model = ChatGoogleGenerativeAI(
                    model="models/gemini-2.5-flash",
                    google_api_key=api_key,
                )
            self._chain = LLMChain(llm=self.llm_model, prompt=self._prompt, memory=self.chat_history)
        else:
            # No API key configured – leave the chain unset so we can
            # surface a clear runtime error when the user tries to chat.
            self.llm_model = None
            self._chain = None

    def retrieveContext(self, analyst: AIAnalystAgent) -> str:
        """
        Build a textual context summary from the analyst's state.
        """
        context_parts: List[str] = []

        if analyst.dataset is not None:
            df = analyst.dataset
            context_parts.append(
                f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns. "
                f"Columns: {list(df.columns)}"
            )

        eda = analyst.analysis_results.get("eda")
        if eda:
            context_parts.append(f"EDA Summary: {eda}")

        ml = analyst.analysis_results.get("ml")
        if ml:
            context_parts.append(f"ML Results: {ml}")

        preds_train = analyst.analysis_results.get("predictions_train")
        if preds_train:
            context_parts.append(f"Training Predictions Sample: {preds_train}")

        preds_new = analyst.analysis_results.get("predictions_new")
        if preds_new:
            context_parts.append(f"New Predictions Summary: {preds_new}")

        if self.vector_store:
            context_parts.append(f"Additional knowledge base keys: {list(self.vector_store.keys())}")

        return "\n\n".join(context_parts)

    def generateResponse(self, question: str, analyst: AIAnalystAgent) -> str:
        """
        Generate a response to the user's question using the LLM chain.
        """
        if self._chain is None:
            raise RuntimeError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY "
                "in your .env file or environment variables."
            )
        context = self.retrieveContext(analyst)
        full_input = f"User Question: {question}\n\nContext:\n{context}"
        answer = self._chain.run(input=full_input)
        self.updateMemory(question, answer)
        return answer

    def updateMemory(self, user_message: str, assistant_message: str) -> None:
        """
        Track Q&A pairs in a simple in-memory vector store and conversation history.
        """
        key = f"q_{len(self.vector_store) + 1}"
        self.vector_store[key] = {
            "question": user_message,
            "answer": assistant_message,
        }

    def explainInsight(self, analyst: AIAnalystAgent) -> str:
        """
        Generate an executive-style insight summary of the current analysis.
        """
        question = (
            "Provide a concise executive-style summary of the key business insights "
            "you can infer from the current dataset, EDA results, model metrics "
            "and predictions. Focus on actionable recommendations."
        )
        return self.generateResponse(question, analyst)


@dataclass
class BusinessUser:
    """
    Represents the business user interacting with the system.
    """

    user_id: str
    name: str
    uploaded_dataset: Optional[pd.DataFrame] = None

    def uploadDataset(self, df: pd.DataFrame) -> pd.DataFrame:
        self.uploaded_dataset = df
        return df

    def viewInsights(self, analyst: AIAnalystAgent) -> Dict[str, Any]:
        """
        View high-level analysis results from the analyst agent.
        """
        return analyst.analysis_results

    def askQuestion(
        self,
        question: str,
        analyst: AIAnalystAgent,
        knowledge_agent: KnowledgeAgent,
    ) -> str:
        """
        Ask a natural language question that will be answered
        by the knowledge agent using the analyst's context.
        """
        return knowledge_agent.generateResponse(question, analyst)

    def downloadReport(
        self,
        analyst: AIAnalystAgent,
        knowledge_agent: KnowledgeAgent,
    ) -> str:
        """
        Generate a narrative report text which can be exported
        by the UI layer (e.g., to PDF or HTML).
        """
        return analyst.generateInsights(knowledge_agent)


def get_or_create_agents(
    state: MutableMapping[str, Any],
) -> Tuple[BusinessUser, AIAnalystAgent, KnowledgeAgent]:
    """
    Helper to keep a single set of agents in Streamlit's session_state
    (or any similar mutable mapping).
    """
    if "business_user" not in state:
        state["business_user"] = BusinessUser(user_id="anonymous", name="Business User")
    if "analyst_agent" not in state:
        state["analyst_agent"] = AIAnalystAgent()
    if "knowledge_agent" not in state:
        state["knowledge_agent"] = KnowledgeAgent()

    return (
        state["business_user"],
        state["analyst_agent"],
        state["knowledge_agent"],
    )

