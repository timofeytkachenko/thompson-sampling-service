import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from requests.exceptions import ConnectionError, RequestException

# --- Configuration ---
st.set_page_config(
    page_title="Thompson Sampling Admin",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL_DEFAULT = "http://localhost:8000"
API_BASE_URL = os.environ.get("API_BASE_URL", API_BASE_URL_DEFAULT)

# Configure basic logging for the admin panel itself
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- API Client Helper Functions ---

SESSION = requests.Session()  # Use a session for potential connection reuse


def _make_request(
    method: str, endpoint: str, **kwargs: Any
) -> Tuple[Optional[Dict[str, Any] | List[Dict[str, Any]]], Optional[str]]:
    """
    Helper function to make requests to the API and handle common errors.

    Parameters
    ----------
    method : str
        HTTP method (GET, POST, PUT, DELETE).
    endpoint : str
        API endpoint path (e.g., '/ads').
    **kwargs : Any
        Additional arguments for requests (json, params, etc.).

    Returns
    -------
    Tuple[Optional[Dict | List], Optional[str]]
        A tuple containing the JSON response (if successful) and an error message (if failed).
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = SESSION.request(method, url, timeout=10, **kwargs)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        try:
            return response.json(), None
        except requests.exceptions.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON response from {url}. Content: {response.text[:100]}"
            )
            return None, f"Invalid JSON response from server: {response.text[:100]}..."

    except ConnectionError:
        logger.error(f"Connection error connecting to API at {url}")
        return (
            None,
            f"Connection Error: Could not connect to API at {url}. Is the service running?",
        )
    except RequestException as e:
        logger.error(f"API request failed: {e}")
        error_detail = str(e)
        if e.response is not None:
            try:
                # Try to get FastAPI's error detail
                error_detail = e.response.json().get("detail", e.response.text)
            except requests.exceptions.JSONDecodeError:
                error_detail = e.response.text
        return None, f"API Request Failed: {error_detail}"


@st.cache_data(ttl=10)  # Cache data for 10 seconds to avoid hammering the API
def get_ads_stats() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch all advertisement statistics from the API."""
    data, error = _make_request("GET", "/ads")
    if error:
        return None, error
    if isinstance(data, list):
        if not data:  # Handle empty list case
            return pd.DataFrame(), None
        df = pd.DataFrame(data)
        # Ensure correct dtypes
        df["impressions"] = (
            pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
        )
        df["clicks"] = (
            pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype(int)
        )
        df["alpha"] = (
            pd.to_numeric(df["alpha"], errors="coerce").fillna(1.0).astype(float)
        )
        df["beta"] = (
            pd.to_numeric(df["beta"], errors="coerce").fillna(1.0).astype(float)
        )
        df["ctr"] = pd.to_numeric(df["ctr"], errors="coerce").fillna(0.0).astype(float)
        df["probability_best"] = (
            pd.to_numeric(df["probability_best"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        df = df.sort_values(by="probability_best", ascending=False)
        return df, None
    return None, "Unexpected data format received for ads stats."


@st.cache_data(ttl=10)
def get_experiment_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current experiment status from the API."""
    return _make_request("GET", "/experiment/status")


@st.cache_data(ttl=10)
def get_warmup_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current warmup status from the API."""
    return _make_request("GET", "/experiment/warmup/status")


@st.cache_data(ttl=60)  # Cache config longer
def get_experiment_config() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current experiment configuration from the API."""
    # Assuming the config is stored and retrievable, else use status endpoint.
    # The current API doesn't have a GET /experiment/config, let's use the status
    # and parse relevant fields or potentially add a GET endpoint later.
    status_data, error = get_experiment_status()
    if error:
        return None, error
    if status_data:
        # Attempt to get config from API; if not present, infer from status or defaults
        # Let's try to get it via the PUT endpoint's structure (though it's not GET)
        # We'll simulate fetching it or use defaults for now.
        # A dedicated GET /config endpoint in main.py would be better.
        ads_stats, _ = get_ads_stats()
        warmup_impressions = 100  # Default, ideally fetched
        min_samples = 1000  # Default
        confidence_threshold = 0.95  # Default
        if ads_stats is not None and not ads_stats.empty:
            # Infer from warmup status if possible
            warmup_data, _ = get_warmup_status()
            if warmup_data and "warmup_impressions_per_ad" in warmup_data:
                warmup_impressions = warmup_data["warmup_impressions_per_ad"]

        # We can't reliably get min_samples and confidence_threshold from status
        # Need to add GET /experiment/config to main.py or allow setting here.
        config = {
            "min_samples": min_samples,  # Placeholder
            "confidence_threshold": confidence_threshold,  # Placeholder
            "warmup_impressions": warmup_impressions,
            # simulation_count is backend internal
        }
        return config, None

    return None, "Could not determine experiment config."


def create_ad(
    name: str, content: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Create a new advertisement."""
    payload = {"name": name, "content": content}
    return _make_request("POST", "/ads", json=payload)


def delete_ad(ad_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Delete an advertisement."""
    return _make_request("DELETE", f"/ads/{ad_id}")


def reset_all_ads() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Delete all advertisements."""
    return _make_request("DELETE", "/ads")


def reset_winner() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Reset the stored experiment winner."""
    return _make_request("DELETE", "/experiment/winner/reset")


def update_config(
    min_samples: int, confidence: float, warmup: int
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Update experiment configuration."""
    payload = {
        "min_samples": min_samples,
        "confidence_threshold": confidence,
        "warmup_impressions": warmup,
        "simulation_count": 10000,  # Keep backend default or make configurable
    }
    return _make_request("PUT", "/experiment/config", json=payload)


# --- Streamlit UI ---

st.title("📊 Thompson Sampling A/B Test Admin Panel")
st.caption(f"Connected to API: {API_BASE_URL}")

# Sidebar for Controls
st.sidebar.header("🛠️ Controls")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()  # Clear cache on manual refresh
    st.rerun()

# Add a placeholder for status messages at the top
status_placeholder = st.empty()

# Fetch data
ads_df, ads_error = get_ads_stats()
exp_status, exp_error = get_experiment_status()
warmup_status, warmup_error = get_warmup_status()
exp_config, config_error = get_experiment_config()

# Display errors prominently if any
if ads_error:
    status_placeholder.error(f"Failed to load ad statistics: {ads_error}")
if exp_error:
    status_placeholder.error(f"Failed to load experiment status: {exp_error}")
if warmup_error:
    status_placeholder.error(f"Failed to load warmup status: {warmup_error}")
# Don't block rendering if only config fails, show defaults maybe
# if config_error:
#     status_placeholder.warning(f"Failed to load experiment config: {config_error}")

# --- Main Dashboard Area ---

if exp_status:
    st.header("🔬 Experiment Status")
    col1, col2, col3, col4 = st.columns(4)

    can_stop = exp_status.get("can_stop", False)
    confidence = exp_status.get("confidence", 0.0)
    winner = exp_status.get("winning_ad")
    recommendation = exp_status.get("recommendation", "N/A")
    in_warmup = exp_status.get("in_warmup_phase", True)
    min_samples_reached = exp_status.get("min_samples_reached", False)
    total_impressions = exp_status.get("total_impressions", 0)

    col1.metric("🏁 Can Stop Experiment?", "Yes" if can_stop else "No")
    col2.metric("🎯 Best Ad Confidence", f"{confidence:.2%}")
    col3.metric("🔥 In Warmup Phase?", "Yes" if in_warmup else "No")
    col4.metric("📈 Total Impressions", f"{total_impressions:,}")

    st.info(f"💡 Recommendation: {recommendation}")

    if winner:
        st.success(
            f"🏆 **Winner Determined:** '{winner.get('name', 'N/A')}' (ID: {winner.get('id', 'N/A')}) "
            f"with {confidence:.1%} confidence. CTR: {winner.get('ctr', 0.0):.3f}"
        )
    elif can_stop:
        st.warning("⚠️ Experiment can be stopped, but no winning ad data returned.")

else:
    st.warning("Could not load experiment status.")

# Ad Performance Section
st.header("📈 Ad Performance")

if ads_df is not None and not ads_df.empty:
    st.subheader("📊 Current Statistics")
    st.dataframe(
        ads_df[
            [
                "id",
                "name",
                "impressions",
                "clicks",
                "ctr",
                "probability_best",
                "alpha",
                "beta",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        # Impressions Distribution
        if not ads_df["impressions"].sum() == 0:
            fig_impressions = px.pie(
                ads_df,
                values="impressions",
                names="name",
                title="Impression Distribution",
                hole=0.3,
            )
            fig_impressions.update_traces(
                textposition="inside", textinfo="percent+label"
            )
            st.plotly_chart(fig_impressions, use_container_width=True)
        else:
            st.info("No impressions recorded yet.")

        # CTR Comparison
        fig_ctr = px.bar(
            ads_df,
            x="name",
            y="ctr",
            title="Measured Click-Through Rate (CTR)",
            labels={"name": "Advertisement", "ctr": "CTR"},
            text="ctr",
        )
        fig_ctr.update_traces(texttemplate="%{text:.3%}", textposition="outside")
        fig_ctr.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig_ctr, use_container_width=True)

    with col2:
        # Probability of Being Best
        fig_prob = px.bar(
            ads_df,
            x="name",
            y="probability_best",
            title="Probability of Being the Best Ad",
            labels={"name": "Advertisement", "probability_best": "Probability"},
            text="probability_best",
        )
        fig_prob.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_prob.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_prob, use_container_width=True)

        # Alpha/Beta Parameters (Beta Distribution Visualization)
        # Note: Plotly doesn't have a direct Beta distribution plot.
        # We could sample or show parameters. Let's show parameters for now.
        fig_params = px.scatter(
            ads_df,
            x="alpha",
            y="beta",
            size="impressions",  # Size bubbles by impressions
            color="name",
            hover_name="name",
            hover_data=["impressions", "clicks", "ctr", "probability_best"],
            title="Beta Distribution Parameters (Alpha vs Beta)",
            labels={"alpha": "Alpha (Successes + 1)", "beta": "Beta (Failures + 1)"},
        )
        st.plotly_chart(fig_params, use_container_width=True)

    # Time Series Data - Placeholder/Note
    st.markdown(
        """
        **Note on Time Series Plots:** Visualizing CTR or Probability over time requires
        historical data logging. The current API provides the *latest* state.
        For time-based analysis, use the `simulation.py` script's output
        or enhance the API service to store and serve historical snapshots.
    """
    )

elif ads_df is not None and ads_df.empty:
    st.info("No advertisements have been created yet.")
else:
    st.warning("Could not load ad statistics.")


# Warmup Status Section
if warmup_status:
    st.header("🔥 Warmup Phase Status")
    required_warmup = warmup_status.get("warmup_impressions_per_ad", "N/A")
    st.metric("Required Warmup Impressions per Ad", required_warmup)

    if warmup_status.get("in_warmup_phase", True):
        st.info(warmup_status.get("message", "Warmup in progress."))
    else:
        st.success(warmup_status.get("message", "Warmup complete."))

    ads_warmup_status = warmup_status.get("ads_status", [])
    if ads_warmup_status:
        warmup_df = pd.DataFrame(ads_warmup_status)
        warmup_df["progress"] = (
            warmup_df["impressions"] / warmup_df["required_warmup"]
        ).clip(
            0, 1
        )  # Ensure progress is between 0 and 1

        st.dataframe(
            warmup_df[
                [
                    "name",
                    "impressions",
                    "required_warmup",
                    "remaining",
                    "warmup_complete",
                ]
            ]
        )

        # Progress bars
        st.subheader("Warmup Progress")
        for _, row in warmup_df.iterrows():
            st.text(f"{row['name']}: {row['impressions']} / {row['required_warmup']}")
            st.progress(row["progress"])
    else:
        st.text("No ad-specific warmup status available.")
else:
    st.warning("Could not load warmup status.")


# --- Sidebar Controls Implementation ---

st.sidebar.subheader("🔧 Manage Ads")
with st.sidebar.expander("Create New Ad", expanded=False):
    with st.form("create_ad_form"):
        new_ad_name = st.text_input("Ad Name")
        new_ad_content = st.text_area("Ad Content")
        create_submitted = st.form_submit_button("Create Ad")
        if create_submitted:
            if not new_ad_name or not new_ad_content:
                st.warning("Please provide both name and content.")
            else:
                with st.spinner("Creating ad..."):
                    response, error = create_ad(new_ad_name, new_ad_content)
                if error:
                    status_placeholder.error(f"Failed to create ad: {error}")
                else:
                    status_placeholder.success(
                        f"Ad '{response.get('name', '')}' created successfully (ID: {response.get('id', '')})"
                    )
                    st.cache_data.clear()  # Clear cache to show new ad
                    st.rerun()  # Rerun to update UI immediately


with st.sidebar.expander("Delete Ad", expanded=False):
    if ads_df is not None and not ads_df.empty:
        ad_to_delete = st.selectbox(
            "Select Ad to Delete",
            options=ads_df["id"],
            format_func=lambda x: f"{ads_df[ads_df['id']==x]['name'].iloc[0]} ({x})",
        )
        if st.button("🗑️ Delete Selected Ad", type="secondary"):
            confirm_delete = st.checkbox(
                f"Confirm deletion of ad '{ads_df[ads_df['id']==ad_to_delete]['name'].iloc[0]}'?"
            )
            if confirm_delete:
                with st.spinner(f"Deleting ad {ad_to_delete}..."):
                    _, error = delete_ad(ad_to_delete)
                if error:
                    status_placeholder.error(f"Failed to delete ad: {error}")
                else:
                    status_placeholder.success(f"Ad {ad_to_delete} deleted.")
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.warning("Deletion not confirmed.")
    else:
        st.info("No ads available to delete.")


st.sidebar.subheader("⚙️ Experiment Settings")
with st.sidebar.expander("Configure Experiment", expanded=False):
    if exp_config:
        current_min_samples = exp_config.get("min_samples", 1000)
        current_confidence = exp_config.get("confidence_threshold", 0.95)
        current_warmup = exp_config.get("warmup_impressions", 100)

        with st.form("config_form"):
            conf_min_samples = st.number_input(
                "Min Samples per Ad", min_value=10, value=current_min_samples
            )
            conf_confidence = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.99,
                value=current_confidence,
                step=0.01,
                format="%.2f",
            )
            conf_warmup = st.number_input(
                "Warmup Impressions per Ad", min_value=0, value=current_warmup
            )
            config_submitted = st.form_submit_button("Update Configuration")

            if config_submitted:
                with st.spinner("Updating configuration..."):
                    response, error = update_config(
                        conf_min_samples, conf_confidence, conf_warmup
                    )
                if error:
                    status_placeholder.error(f"Failed to update config: {error}")
                else:
                    status_placeholder.success("Experiment configuration updated.")
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.warning("Could not load current configuration to display form.")


st.sidebar.subheader("💣 Danger Zone")
with st.sidebar.expander("Reset Options", expanded=False):
    if st.button("⚠️ Reset Stored Winner", type="secondary"):
        confirm_reset_winner = st.checkbox("Confirm resetting the stored winner?")
        if confirm_reset_winner:
            with st.spinner("Resetting winner..."):
                response, error = reset_winner()
            if error:
                status_placeholder.error(f"Failed to reset winner: {error}")
            else:
                status_placeholder.success(
                    response.get("message", "Winner reset successfully.")
                )
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Winner reset not confirmed.")

    if st.button("🚨 Reset ALL Ads & Winner", type="primary"):
        st.warning(
            "This will delete ALL advertisements and reset the experiment winner."
        )
        confirm_reset_all = st.checkbox("Confirm resetting ALL ads and the winner?")
        if confirm_reset_all:
            with st.spinner("Resetting all ads..."):
                _, ad_error = reset_all_ads()
            with st.spinner("Resetting winner..."):
                _, winner_error = reset_winner()

            if ad_error or winner_error:
                status_placeholder.error(
                    f"Reset failed. Ads: {ad_error or 'OK'}. Winner: {winner_error or 'OK'}"
                )
            else:
                status_placeholder.success("All advertisements and winner reset.")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Full reset not confirmed.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Last refresh: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
