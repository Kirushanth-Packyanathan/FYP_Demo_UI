# Multi-Signal Cascade Framework (MSCF) - Streamlit Dashboard

A professional, real-time interactive Streamlit web dashboard for the **Multi-Signal Cascade Framework (MSCF) Hallucination Detection System**.

This frontend communicates seamlessly with a FastAPI backend to perform multi-sample LLM inference, calculate uncertainty metrics (semantic entropy, embedding dispersion, logit confidence, fused uncertainty), group responses into semantic clusters, execute NLI deep verification, and display real-time hallucination warnings.

---

## 🌟 Key Features

- ⚙️ **Dynamic API Endpoint Configuration**: Paste any public tunnel or local FastAPI URL (e.g. `https://mighty-emus-fall.loca.lt`) directly in the sidebar with live health checks and status indicators.
- 🎯 **Real-Time Uncertainty Quantification**: Interactive Plotly gauge visualization for **Fused Uncertainty** with color-coded risk bands and threshold markers ($\tau_{\text{low}} = 0.30$, $\tau_{\text{high}} = 0.70$).
- 🛡️ **Instant Hallucination Banners**: Prominent **Response Accepted** (Green) or **Potential Hallucination Detected** (Red) status banners.
- 📈 **Streamlit Metric Grid**: 6 dedicated metric cards displaying Semantic Entropy, Dispersion, Logit Confidence, Fused Uncertainty, Execution Time, and Hallucination Confidence.
- 💬 **Generated Responses & Clusters**: Expandable views listing sampled candidate responses and a Plotly bar chart mapping Cluster ID vs Cluster Size.
- 🔬 **NLI Deep Verification Inspector**: Displays Natural Language Inference confidence scores or indicates early cascade exit.
- 📄 **Raw JSON Inspector**: Full inspectability via `st.json()`.
- 🛡️ **Comprehensive Error Handling**: Friendly, actionable error messages for connection timeouts, DNS failures, HTTP 500 errors, invalid JSON payloads, and malformed URLs.

---

## 📁 Repository Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python package dependencies
└── README.md           # Project setup and documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9 or higher installed.

### 2. Environment Setup & Dependency Installation

Create a virtual environment (optional but recommended):

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Launch the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🔌 Connecting to the FastAPI Backend

1. In the Streamlit sidebar under **⚙️ API Configuration**, paste your FastAPI base URL into the **FastAPI Base URL** text input:
   ```
   https://mighty-emus-fall.loca.lt
   ```
2. Click the **Connect** button to test the endpoint connectivity.
3. Once connected (🟢 **API Status: Connected**), enter your query into the text area on the main page.
4. Click **🚀 Run Cascade**. The application sends a POST request to `<API_URL>/predict` with:

```json
{
    "query": "What is the capital of France?"
}
```

---

## 📡 Expected API Response Schema

The backend is expected to return JSON structured like:

```json
{
    "query": "What is the capital of France?",
    "responses": [
        "The capital of France is Paris.",
        "Paris is France's capital city.",
        "The capital of France is Paris."
    ],
    "cluster_map": {
        "0": [0, 1, 2]
    },
    "semantic_entropy": 0.12,
    "dispersion": 0.18,
    "logit_confidence": 0.91,
    "fused_uncertainty": 0.15,
    "gate_decision": "accept",
    "hallucination": false,
    "hallucination_confidence": 0.04,
    "nli_result": {
        "confidence": 0.08
    },
    "final_answer": "The capital of France is Paris.",
    "execution_time": 2.41
}
```

---

## 🛡️ Error Handling Coverage

The application handles edge cases gracefully:
- **Connection Refused**: Displayed if FastAPI backend is offline or tunnel URL is unreachable.
- **Invalid URL**: Warns user if missing protocol scheme (`http://` or `https://`).
- **Timeout**: Triggered if model inference takes longer than 45 seconds.
- **HTTP 500 / Server Errors**: Highlights backend failure details without crashing the UI.
- **Malformed JSON**: Catches invalid JSON payload format from backend.
