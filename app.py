"""
Multi-Signal Cascade Framework (MSCF) Streamlit Frontend
Real-Time Hallucination Detection & Uncertainty Quantification Dashboard

Author: Antigravity AI Team
Description: A production-quality Streamlit application to interface with the MSCF FastAPI backend.
"""

import json
import math
import time
import urllib.parse
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLES
# ==============================================================================

st.set_page_config(
    page_title="MSCF - Hallucination Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics (cards, custom banners, metrics)
CUSTOM_CSS = """
<style>
    /* Main layout tuning */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Modern Card Container */
    .mscf-card {
        background-color: var(--background-secondary, #f8f9fa);
        border: 1px solid var(--border-color, #e0e0e0);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Dark mode adjustments for card */
    @media (prefers-color-scheme: dark) {
        .mscf-card {
            background-color: #1e222a;
            border-color: #2e3440;
        }
    }

    /* Final Answer Card */
    .answer-card {
        border-left: 5px solid #3b82f6;
        background-color: rgba(59, 130, 246, 0.05);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .answer-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }

    .answer-body {
        font-size: 1.1rem;
        line-height: 1.6;
        font-weight: 500;
    }

    /* Dynamic Result Banners */
    .banner-accepted {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
        margin-bottom: 1rem;
    }

    .banner-hallucination {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
        margin-bottom: 1rem;
    }

    /* Response Pill/Card */
    .response-item {
        background-color: rgba(128, 128, 128, 0.08);
        color: var(--text-color, inherit);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    /* Status Badges */
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-error {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
import urllib3

# Suppress SSL certificate warnings when using SSL bypass for local tunnels
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Default HTTP Headers to bypass localtunnel/ngrok reminder screens & handle proxies
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Connection": "close",
    "Bypass-Tunnel-Reminder": "true",
    "bypass-tunnel-reminder": "true",
    "ngrok-skip-browser-warning": "true",
    "Ngrok-Skip-Browser-Warning": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MSCF-Streamlit-Frontend",
}


def sanitize_url(raw_url: str) -> str:
    """
    Sanitize and validate URL input.
    Ensures http:// or https:// prefix and strips trailing slashes and trailing /predict.
    """
    if not raw_url:
        return ""

    url = raw_url.strip()
    # Add default scheme if missing (http:// for localhost/IPs, https:// for domain tunnels)
    if not url.startswith("http://") and not url.startswith("https://"):
        if "localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url:
            url = "http://" + url
        else:
            url = "https://" + url

    url = url.rstrip("/")
    if url.endswith("/predict"):
        url = url[:-8]

    return url.rstrip("/")


def fetch_thresholds_from_backend(
    base_url: str, timeout: int = 5
) -> Tuple[Optional[float], Optional[float]]:
    """
    Query GET /thresholds endpoint on the backend API to dynamically fetch tau_low and tau_high values.
    Sends: GET <URL>/thresholds with Header Accept: application/json
    Example Response: {"tau_low": 0.3149494001764102, "tau_high": 0.5849060288990476}
    """
    cleaned_url = sanitize_url(base_url)
    if not cleaned_url:
        return None, None

    endpoint = f"{cleaned_url}/thresholds"
    headers = dict(DEFAULT_HEADERS)
    headers["Accept"] = "application/json"

    session = requests.Session()
    session.headers.update(headers)

    try:
        res = session.get(endpoint, timeout=timeout, verify=False)
        if res.status_code == 200:
            low_text = res.text.lower()
            if not (
                "<html" in low_text
                and (
                    "localtunnel" in low_text
                    or "reminder" in low_text
                    or "password" in low_text
                    or "tunnel" in low_text
                    or "ngrok" in low_text
                )
            ):
                data = res.json()
                if isinstance(data, dict):
                    tau_low = None
                    tau_high = None
                    for k in ["tau_low", "TAU_LOW", "tau_low_val", "threshold_low"]:
                        if k in data and data[k] is not None:
                            tau_low = float(data[k])
                            break
                    for k in ["tau_high", "TAU_HIGH", "tau_high_val", "threshold_high"]:
                        if k in data and data[k] is not None:
                            tau_high = float(data[k])
                            break
                    return tau_low, tau_high
    except Exception:
        pass

    return None, None


def validate_api_connection(
    base_url: str, timeout: int = 20
) -> Tuple[bool, str, Optional[float], Optional[float]]:
    """
    Validate backend URL accessibility by querying GET /thresholds endpoint.
    Retrieves dynamic tau_low and tau_high threshold values without triggering LLM model inference.
    """
    cleaned_url = sanitize_url(base_url)
    if not cleaned_url:
        return False, "URL cannot be empty.", None, None

    # Fetch dynamic thresholds strictly from backend GET /thresholds endpoint
    t_low, t_high = fetch_thresholds_from_backend(cleaned_url, timeout=timeout)

    if t_low is not None and t_high is not None:
        msg = f"Connected successfully to API backend | Fetched GET /thresholds: τ_low = {t_low:.4f}, τ_high = {t_high:.4f}"
        return True, msg, t_low, t_high

    # If /thresholds response had no values, perform lightweight GET connection check
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    try:
        response = session.get(cleaned_url, timeout=timeout, verify=False)
        low_text = response.text.lower()
        if "<html" in low_text and (
            "localtunnel" in low_text
            or "reminder" in low_text
            or "password" in low_text
            or "friendly reminder" in low_text
            or "ngrok" in low_text
        ):
            return (
                False,
                f"Tunnel password/reminder screen detected! Please open {cleaned_url} in a new browser tab once to submit your password or click 'Click to Continue'.",
                None,
                None,
            )

        if response.status_code in (200, 404, 405, 422, 400):
            msg = f"Connected to backend server ({response.status_code})"
            return True, msg, t_low, t_high

        return False, f"API returned status code {response.status_code}", None, None

    except Exception as exc:
        if isinstance(exc, requests.exceptions.MissingSchema):
            return (
                False,
                "Invalid URL schema. Please include http:// or https://",
                None,
                None,
            )
        elif isinstance(exc, requests.exceptions.InvalidURL):
            return False, "Invalid URL format.", None, None
        elif isinstance(exc, requests.exceptions.Timeout):
            return (
                False,
                f"Connection timed out after {timeout}s. Verify your backend server & tunnel are active.",
                None,
                None,
            )
        elif isinstance(exc, requests.exceptions.ConnectionError):
            return (
                False,
                "Connection refused/failed. Ensure FastAPI server & tunnel process are running.",
                None,
                None,
            )
        else:
            return (
                False,
                f"Connection error ({type(exc).__name__}): {str(exc)}",
                None,
                None,
            )


def call_mscf_api(
    base_url: str, query: str, timeout: int = 60, max_retries: int = 3
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Execute HTTP POST request to <API_URL>/predict with query payload.
    Includes retry logic and Connection: close header for localtunnel socket stability.
    Always fetches dynamic threshold values from GET /thresholds.
    """
    cleaned_url = sanitize_url(base_url)
    if not cleaned_url:
        return None, "Invalid API URL provided. Please configure the sidebar API URL."

    endpoint = f"{cleaned_url}/predict"
    payload = {"query": query}
    start_time = time.time()
    last_err_msg = ""

    for attempt in range(1, max_retries + 1):
        try:
            # Create fresh session for each attempt to avoid reused socket resets
            session = requests.Session()
            session.headers.update(DEFAULT_HEADERS)

            response = session.post(
                endpoint, json=payload, timeout=timeout, verify=False
            )
            elapsed = time.time() - start_time

            # Check HTTP status code
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "execution_time" not in data or data["execution_time"] is None:
                        data["execution_time"] = round(elapsed, 2)

                    # Unconditionally fetch dynamic thresholds from GET /thresholds
                    t_low, t_high = fetch_thresholds_from_backend(cleaned_url)
                    if t_low is not None:
                        data["tau_low"] = t_low
                    if t_high is not None:
                        data["tau_high"] = t_high

                    return data, None

                    return data, None
                except json.JSONDecodeError:
                    return (
                        None,
                        "Malformed Response: Server returned invalid JSON payload.",
                    )

            elif response.status_code == 500:
                return (
                    None,
                    f"500 Internal Server Error: The backend encountered an unhandled exception while processing query.",
                )
            elif response.status_code == 404:
                return (
                    None,
                    f"404 Not Found: Endpoint '{endpoint}' does not exist. Ensure FastAPI has a POST /predict route.",
                )
            else:
                return None, f"HTTP Error {response.status_code}: {response.text}"

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_err_msg = (
                f"Attempt {attempt}/{max_retries} failed ({type(e).__name__}): {str(e)}"
            )
            if attempt < max_retries:
                time.sleep(1.0)
                continue
        except requests.exceptions.Timeout:
            return (
                None,
                f"Timeout Error: Request timed out after {timeout} seconds. The backend model inference is taking too long.",
            )
        except requests.exceptions.InvalidURL:
            return None, "Invalid URL Error: The configured API URL is malformed."
        except Exception as e:
            return None, f"Unexpected Error ({type(e).__name__}): {str(e)}"

    return (
        None,
        f"Connection Failed after {max_retries} attempts ({last_err_msg}). Ensure localtunnel connection is open.",
    )


def parse_semantic_clusters(
    cluster_map: Any, responses: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Flexibly parse semantic cluster maps returned by backend.
    Supports formats:
      - {"0": [0, 2], "1": [1]} (dict of cluster_id -> response indices)
      - {"0": ["resp1", "resp2"]} (dict of cluster_id -> response strings)
      - {"resp1": 0, "resp2": 1} (dict of response -> cluster_id)
    Returns:
      (clusters_dict, cluster_sizes_dict)
    """
    clusters: Dict[str, List[str]] = {}

    if not cluster_map:
        return clusters, {}

    if isinstance(cluster_map, dict):
        for key, val in cluster_map.items():
            cluster_id = f"Cluster {key}"
            if isinstance(val, list):
                # val is list of response indices or texts
                cluster_items = []
                for item in val:
                    if isinstance(item, int) and 0 <= item < len(responses):
                        cluster_items.append(f"Response {item + 1}: {responses[item]}")
                    elif isinstance(item, str):
                        cluster_items.append(item)
                    else:
                        cluster_items.append(str(item))
                clusters[cluster_id] = cluster_items
            elif isinstance(val, (int, str)):
                # key might be response text/index, val is cluster_id
                cid = f"Cluster {val}"
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(str(key))

    cluster_sizes = {cid: len(items) for cid, items in clusters.items()}
    return clusters, cluster_sizes


# ==============================================================================
# 3. PLOTLY VISUALIZATIONS
# ==============================================================================


def create_fused_uncertainty_gauge(
    fused_uncertainty: float,
    tau_low: Optional[float] = None,
    tau_high: Optional[float] = None,
) -> go.Figure:
    """
    Create a Plotly gauge chart visualizing Fused Uncertainty (0 to 1 scale).
    Threshold markers tau_low and tau_high are dynamically retrieved from backend GET /thresholds.
    Color zones:
      Low uncertainty (0.0 - tau_low): Green
      Medium uncertainty (tau_low - tau_high): Orange/Yellow
      High uncertainty (tau_high - 1.00): Red
    """
    # Use dynamically fetched backend values or fallback defaults (0.30, 0.70)
    t_low_val = 0.30 if tau_low is None else float(tau_low)
    t_high_val = 0.70 if tau_high is None else float(tau_high)

    val = max(0.0, min(1.0, float(fused_uncertainty)))
    t_low = max(0.0, min(1.0, t_low_val))
    t_high = max(t_low, min(1.0, t_high_val))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"<b>Fused Uncertainty Metric</b><br><span style='font-size:12px;color:#555555;'>Dynamic Thresholds (GET /thresholds): Low (≤ {t_low:.4f}) | Med ({t_low:.4f}–{t_high:.4f}) | High (≥ {t_high:.4f})</span>",
                "font": {"size": 17},
            },
            number={"valueformat": ".3f", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#888888"},
                "bar": {"color": "#1f2937", "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#cccccc",
                "steps": [
                    {
                        "range": [0.0, t_low],
                        "color": "rgba(46, 204, 113, 0.45)",
                    },  # Low (Green)
                    {
                        "range": [t_low, t_high],
                        "color": "rgba(243, 156, 18, 0.45)",
                    },  # Medium (Yellow/Orange)
                    {
                        "range": [t_high, 1.00],
                        "color": "rgba(231, 76, 60, 0.45)",
                    },  # High (Red)
                ],
                "threshold": {
                    "line": {"color": "#000000", "width": 4},
                    "thickness": 0.8,
                    "value": val,
                },
            },
        )
    )

    # Position annotations accurately along semi-circle gauge arc using trigonometry
    rad_low = math.pi * (1.0 - t_low)
    x_low = round(0.50 + 0.36 * math.cos(rad_low), 3)
    y_low = round(0.18 + 0.28 * math.sin(rad_low), 3)

    rad_high = math.pi * (1.0 - t_high)
    x_high = round(0.50 + 0.36 * math.cos(rad_high), 3)
    y_high = round(0.18 + 0.28 * math.sin(rad_high), 3)

    fig.add_annotation(
        x=x_low,
        y=y_low,
        text=f"<b>τ_low = {t_low:.4f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#27ae60",
        ax=0,
        ay=25 if t_low < 0.5 else -25,
        font=dict(size=12, color="#27ae60", family="sans-serif"),
    )
    fig.add_annotation(
        x=x_high,
        y=y_high,
        text=f"<b>τ_high = {t_high:.4f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#c0392b",
        ax=0,
        ay=25 if t_high < 0.5 else -25,
        font=dict(size=12, color="#c0392b", family="sans-serif"),
    )

    fig.update_layout(
        height=310,
        margin=dict(l=25, r=25, t=65, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def create_cluster_bar_chart(cluster_sizes: Dict[str, int]) -> go.Figure:
    """
    Generate a Plotly bar chart visualizing Cluster ID vs Cluster Size.
    """
    if not cluster_sizes:
        df = pd.DataFrame({"Cluster ID": ["No Clusters"], "Cluster Size": [0]})
    else:
        df = pd.DataFrame(
            list(cluster_sizes.items()), columns=["Cluster ID", "Cluster Size"]
        )

    fig = px.bar(
        df,
        x="Cluster ID",
        y="Cluster Size",
        text="Cluster Size",
        color="Cluster ID",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Semantic Cluster Sizes",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Cluster ID",
        yaxis_title="Number of Responses",
        showlegend=False,
        yaxis=dict(dtick=1),
    )
    return fig


# ==============================================================================
# 4. STREAMLIT APPLICATION COMPONENT & MAIN LOOP
# ==============================================================================


def main():
    """
    Main Streamlit application entrypoint rendering MSCF Dashboard.
    """

    # Initialize session state variables
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = "https://mighty-emus-fall.loca.lt"
    if "connection_status" not in st.session_state:
        st.session_state["connection_status"] = None  # None, True, or False
    if "connection_msg" not in st.session_state:
        st.session_state["connection_msg"] = ""
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "tau_low" not in st.session_state:
        st.session_state["tau_low"] = 0.30
    if "tau_high" not in st.session_state:
        st.session_state["tau_high"] = 0.70

    # Auto-check API connection on initial load to fetch dynamic thresholds
    if st.session_state["connection_status"] is None and st.session_state.get(
        "api_url"
    ):
        is_valid, msg, t_low, t_high = validate_api_connection(
            st.session_state["api_url"]
        )
        st.session_state["connection_status"] = is_valid
        st.session_state["connection_msg"] = msg
        if is_valid and t_low is not None and t_high is not None:
            st.session_state["tau_low"] = t_low
            st.session_state["tau_high"] = t_high

    # --------------------------------------------------------------------------
    # SIDEBAR COMPONENT
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ API Configuration")
        st.caption("Connect to your MSCF FastAPI backend endpoint.")

        # API URL Input field
        url_input = st.text_input(
            "FastAPI Base URL",
            value=st.session_state["api_url"],
            placeholder="https://mighty-emus-fall.loca.lt",
            help="Paste your public tunnel or local FastAPI base URL. The application automatically appends /predict.",
        )

        cleaned_input = url_input.strip()
        if cleaned_input and cleaned_input != st.session_state["api_url"]:
            st.session_state["api_url"] = cleaned_input
            st.session_state["connection_status"] = None

        # Auto-validate API connection if not checked yet for current URL
        if (
            st.session_state["connection_status"] is None
            and st.session_state["api_url"]
        ):
            is_valid, msg, t_low, t_high = validate_api_connection(
                st.session_state["api_url"]
            )
            st.session_state["connection_status"] = is_valid
            st.session_state["connection_msg"] = msg
            if is_valid and t_low is not None and t_high is not None:
                st.session_state["tau_low"] = t_low
                st.session_state["tau_high"] = t_high

        st.caption(
            "💡 **Localtunnel Tip**: If using `loca.lt`, open the URL in a browser tab once to bypass the localtunnel IP password screen if prompted."
        )

        col_conn, col_space = st.columns([1, 1])
        with col_conn:
            connect_btn = st.button("Connect", use_container_width=True, type="primary")

        if connect_btn:
            with st.spinner("Testing API connection & fetching thresholds..."):
                is_valid, msg, t_low, t_high = validate_api_connection(
                    st.session_state["api_url"]
                )
                st.session_state["connection_status"] = is_valid
                st.session_state["connection_msg"] = msg
                if is_valid and t_low is not None and t_high is not None:
                    st.session_state["tau_low"] = t_low
                    st.session_state["tau_high"] = t_high

        # Display connection status badge
        if st.session_state["connection_status"] is True:
            st.success("🟢 API Status: Connected")
            st.caption(st.session_state["connection_msg"])
            t_l = st.session_state.get("tau_low", 0.30)
            t_h = st.session_state.get("tau_high", 0.70)
            st.info(
                f"📐 **Dynamic Thresholds**: τ_low = **{t_l:.4f}**, τ_high = **{t_h:.4f}**"
            )
        elif st.session_state["connection_status"] is False:
            st.error("🔴 API Status: Disconnected")
            st.caption(st.session_state["connection_msg"])
        else:
            st.info("ℹ️ Status: Not Checked")

        st.divider()

        # Sample Queries for Quick Testing
        st.subheader("💡 Example Queries")
        sample_queries = [
            "How many seconds are there in one hour?",
            "What is the capital of Sri Lanka?",
            "What are the specifications of Apple's iPhone 18 Ultra Fold Mini?",
            "Explain quantum entanglement in simple terms.",
            "Who won the 2024 FIFA World Cup?",
            "What are the primary causes of climate change?",
            "Who won the Nobel Prize in Artificial General Intelligence in 2024?",
        ]

        selected_sample = st.radio("Click to prefill query:", sample_queries, index=0)

        st.divider()
        st.markdown(
            "**About MSCF System**\n"
            "The Multi-Signal Cascade Framework integrates semantic entropy, "
            "embedding dispersion, logit confidence, and NLI verification to detect "
            "LLM hallucinations in real-time."
        )

    # --------------------------------------------------------------------------
    # MAIN PAGE HEADER
    # --------------------------------------------------------------------------
    st.title("🛡️ Multi-Signal Cascade Framework")
    st.subheader("Real-Time Hallucination Detection & Uncertainty Quantification")
    st.markdown(
        "Submit a prompt to run multi-sample inference, compute uncertainty metrics, "
        "and determine response validity through the cascade pipeline."
    )
    st.divider()

    # --------------------------------------------------------------------------
    # QUERY INPUT SECTION
    # --------------------------------------------------------------------------
    # Use selected sample as default prompt text if user hasn't overridden
    query_text = st.text_area(
        "Enter User Query:",
        value=(
            selected_sample if selected_sample else "What is the capital of Sri Lanka?"
        ),
        height=120,
        placeholder="Type your question or query here...",
        help="The query will be dispatched to <API_URL>/predict for multi-signal hallucination analysis.",
    )

    col_run, col_clear = st.columns([2, 10])
    with col_run:
        run_button = st.button(
            "🚀 Run Cascade", type="primary", use_container_width=True
        )

    # Handle Cascade Execution
    if run_button:
        if not query_text.strip():
            st.warning("⚠️ Please enter a query before running the cascade.")
        elif not st.session_state["api_url"].strip():
            st.error("❌ Please provide a valid FastAPI endpoint URL in the sidebar.")
        else:
            with st.spinner(
                "⏳ Running MSCF Cascade Pipeline... Sampling responses and calculating uncertainty metrics..."
            ):
                result_data, error_msg = call_mscf_api(
                    base_url=st.session_state["api_url"], query=query_text.strip()
                )

                if error_msg:
                    st.error(f"❌ API Request Failed: {error_msg}")
                    st.session_state["last_result"] = None
                else:
                    st.session_state["last_result"] = result_data
                    if isinstance(result_data, dict):
                        for k_l in ["tau_low", "TAU_LOW", "tau_low_val"]:
                            if k_l in result_data and result_data[k_l] is not None:
                                st.session_state["tau_low"] = float(result_data[k_l])
                                break
                        for k_h in ["tau_high", "TAU_HIGH", "tau_high_val"]:
                            if k_h in result_data and result_data[k_h] is not None:
                                st.session_state["tau_high"] = float(result_data[k_h])
                                break

    # --------------------------------------------------------------------------
    # RESULTS RENDERING SECTION
    # --------------------------------------------------------------------------
    if st.session_state["last_result"]:
        res = st.session_state["last_result"]

        st.subheader("📊 Cascade Execution Results")

        # 1. Dynamic Status Banner (Accepted vs Hallucination Detected)
        is_hallucination = res.get("hallucination", False)
        gate_decision = res.get("gate_decision", "N/A").upper()

        if is_hallucination:
            st.markdown(
                f"""
                <div class="banner-hallucination">
                    <span>⚠️ Potential Hallucination Detected</span>
                    <span style="font-size: 0.9rem; font-weight: normal; margin-left: auto;">Gate Decision: <b>{gate_decision}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="banner-accepted">
                    <span>✅ Response Accepted</span>
                    <span style="font-size: 0.9rem; font-weight: normal; margin-left: auto;">Gate Decision: <b>{gate_decision}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Final Answer Highlighted Card (Only displayed if response is accepted)
        if not is_hallucination:
            final_answer = res.get("final_answer", "No answer returned.")
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-title">Final Answer Output</div>
                    <div class="answer-body">{final_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("🚫 **Final Answer Suppressed**: Output was rejected by the cascade pipeline because potential hallucination was detected.")

        # ----------------------------------------------------------------------
        # METRICS SECTION
        # ----------------------------------------------------------------------
        st.markdown("### 📈 Core Uncertainty Metrics")

        # Safely extract metric values with formatting defaults
        sem_entropy = res.get("semantic_entropy", 0.0)
        dispersion = res.get("dispersion", 0.0)
        logit_conf = res.get("logit_confidence", 0.0)
        fused_unc = res.get("fused_uncertainty", 0.0)
        exec_time = res.get("execution_time", 0.0)
        hallucination_conf = res.get("hallucination_confidence", 0.0)

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            st.metric(
                label="Semantic Entropy",
                value=(
                    f"{sem_entropy:.4f}"
                    if isinstance(sem_entropy, (int, float))
                    else str(sem_entropy)
                ),
                help="Measures semantic variability across sampled model responses.",
            )
        with m2:
            st.metric(
                label="Dispersion",
                value=(
                    f"{dispersion:.4f}"
                    if isinstance(dispersion, (int, float))
                    else str(dispersion)
                ),
                help="Embedding space variance/dispersion across generated responses.",
            )
        with m3:
            st.metric(
                label="Logit Confidence",
                value=(
                    f"{logit_conf:.4f}"
                    if isinstance(logit_conf, (int, float))
                    else str(logit_conf)
                ),
                help="Average output probability confidence from model logits.",
            )
        with m4:
            st.metric(
                label="Fused Uncertainty",
                value=(
                    f"{fused_unc:.4f}"
                    if isinstance(fused_unc, (int, float))
                    else str(fused_unc)
                ),
                help="Combined multi-signal cascade uncertainty score.",
            )
        with m5:
            st.metric(
                label="Execution Time",
                value=(
                    f"{exec_time:.2f} s"
                    if isinstance(exec_time, (int, float))
                    else str(exec_time)
                ),
                help="Total processing time elapsed for cascade analysis.",
            )
        with m6:
            st.metric(
                label="Hallucination Conf.",
                value=(
                    f"{hallucination_conf:.4f}"
                    if isinstance(hallucination_conf, (int, float))
                    else str(hallucination_conf)
                ),
                help="Overall probability score indicating potential hallucination.",
            )

        st.divider()

        # ----------------------------------------------------------------------
        # UNCERTAINTY VISUALIZATION & GENERATED RESPONSES
        # ----------------------------------------------------------------------
        # Extract dynamic thresholds from result object first, fallback to session state
        t_low_val = float(
            res.get(
                "tau_low", res.get("TAU_LOW", st.session_state.get("tau_low", 0.30))
            )
        )
        t_high_val = float(
            res.get(
                "tau_high", res.get("TAU_HIGH", st.session_state.get("tau_high", 0.70))
            )
        )
        st.session_state["tau_low"] = t_low_val
        st.session_state["tau_high"] = t_high_val

        col_gauge, col_info = st.columns([5, 4])

        with col_gauge:
            st.markdown("### 🎯 Uncertainty Gauge")
            gauge_fig = create_fused_uncertainty_gauge(
                fused_uncertainty=fused_unc, tau_low=t_low_val, tau_high=t_high_val
            )
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                key=f"plotly_gauge_{fused_unc:.4f}_{t_low_val:.4f}_{t_high_val:.4f}",
            )

        with col_info:
            unc_level = (
                "HIGH"
                if fused_unc >= t_high_val
                else ("MEDIUM" if fused_unc >= t_low_val else "LOW")
            )
            st.markdown("### 🔍 Cascade Evaluation Summary")
            st.markdown(
                f"""
                - **Query**: *"{res.get('query', query_text)}"*
                - **Gate Decision**: `{gate_decision}`
                - **Hallucination Detected**: `{"TRUE" if is_hallucination else "FALSE"}`
                - **Uncertainty Level**: `{unc_level}` (τ_low = {t_low_val:.4f}, τ_high = {t_high_val:.4f})
                """
            )

        st.divider()

        # ----------------------------------------------------------------------
        # EXPANDABLE SECTIONS
        # ----------------------------------------------------------------------

        # 1. Generated Responses
        responses = res.get("responses", [])
        with st.expander(
            f"💬 Generated Responses ({len(responses)} Samples)", expanded=False
        ):
            if responses:
                for idx, resp_text in enumerate(responses, start=1):
                    st.markdown(
                        f"""
                        <div class="response-item" style="padding: 1rem; border-radius: 8px; margin-bottom: 0.75rem;">
                            <strong style="color: #3b82f6; font-size: 1.05rem;">Response {idx}:</strong><br/>
                            <div style="margin-top: 0.4rem; line-height: 1.5;">{resp_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No sampled responses provided in API output.")

        # 2. Semantic Clusters
        cluster_map = res.get("cluster_map", {})
        clusters, cluster_sizes = parse_semantic_clusters(cluster_map, responses)

        with st.expander("🧩 Semantic Clusters & Size Distribution", expanded=False):
            col_cl_list, col_cl_chart = st.columns([1, 1])

            with col_cl_list:
                st.markdown("#### Cluster Groupings")
                if clusters:
                    for cid, items in clusters.items():
                        st.markdown(f"**{cid}** ({len(items)} responses)")
                        for item in items:
                            st.caption(f"• {item}")
                else:
                    st.info("No cluster mappings available.")

            with col_cl_chart:
                st.markdown("#### Cluster Size Chart")
                cluster_fig = create_cluster_bar_chart(cluster_sizes)
                st.plotly_chart(cluster_fig, use_container_width=True)

        # 3. Deep Verification (NLI)
        nli_result = res.get("nli_result", None)
        with st.expander("🔬 Deep Verification (NLI Analysis)", expanded=False):
            if (
                nli_result
                and isinstance(nli_result, dict)
                and "confidence" in nli_result
            ):
                nli_conf = nli_result.get("confidence", 0.0)
                st.success("✅ NLI Verification Executed")
                st.metric(
                    label="NLI Entailment Confidence",
                    value=(
                        f"{nli_conf:.2%}"
                        if isinstance(nli_conf, (int, float))
                        else str(nli_conf)
                    ),
                )
                if len(nli_result) > 1:
                    st.json(nli_result)
            else:
                st.info("ℹ️ NLI Not Executed (Early Cascade Exit)")

        # 4. Technical Details (Raw JSON Response)
        with st.expander("📄 Technical Details (Raw JSON Payload)", expanded=False):
            st.json(res)


# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
