import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, RequestException

load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Thompson Sampling Admin",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL_DEFAULT = "http://localhost:8000"
API_BASE_URL = os.environ.get("API_BASE_URL", API_BASE_URL_DEFAULT)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- API Client Helper Functions ---

SESSION = requests.Session()


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
        response.raise_for_status()
        try:
            # Handle potential empty responses for DELETE etc.
            if response.status_code == 204 or not response.content:
                return {"message": "Operation successful (No content)"}, None
            return response.json(), None
        except requests.exceptions.JSONDecodeError:
            logger.error(
                f"Failed to decode JSON response from {url}. Status: {response.status_code}. Content: {response.text[:100]}"
            )
            # Return raw text if it's likely an error message not in JSON
            if 400 <= response.status_code < 600 and response.text:
                return None, f"Server error (non-JSON): {response.text[:200]}..."
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
                error_json = e.response.json()
                error_detail = error_json.get(
                    "detail", str(error_json)
                )  # FastAPI often uses 'detail'
            except requests.exceptions.JSONDecodeError:
                error_detail = e.response.text
        return None, f"API Request Failed: {error_detail}"


@st.cache_data(ttl=10)
def get_ads_stats() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch all advertisement statistics from the API and ensure data types.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        DataFrame with ad stats or None, and an optional error message.
    """
    data, error = _make_request("GET", "/ads")
    if error:
        return None, error
    if isinstance(data, list):
        if not data:
            # Ensure columns exist even for empty dataframe
            return (
                pd.DataFrame(
                    columns=[
                        "id",
                        "name",
                        "content",
                        "impressions",
                        "clicks",
                        "alpha",
                        "beta",
                        "ctr",
                        "probability_best",
                    ]
                ),
                None,
            )
        try:
            df = pd.DataFrame(data)
            # Define expected columns and their types
            expected_dtypes = {
                "id": str,
                "name": str,
                "content": str,  # Added content if present
                "impressions": int,
                "clicks": int,
                "alpha": float,
                "beta": float,
                "ctr": float,
                "probability_best": float,
            }
            # Select and cast columns, handling missing ones gracefully
            processed_cols = {}
            for col, dtype in expected_dtypes.items():
                if col not in df.columns:
                    # Add missing columns with default values
                    if dtype == int:
                        processed_cols[col] = pd.Series(
                            [0] * len(df), index=df.index, dtype=int
                        )
                    elif dtype == float:
                        default_val = 1.0 if col in ["alpha", "beta"] else 0.0
                        processed_cols[col] = pd.Series(
                            [default_val] * len(df), index=df.index, dtype=float
                        )
                    else:  # str
                        processed_cols[col] = pd.Series(
                            [""] * len(df), index=df.index, dtype=str
                        )
                else:
                    # Apply casting with error handling
                    if dtype == int:
                        processed_cols[col] = (
                            pd.to_numeric(df[col], errors="coerce")
                            .fillna(0)
                            .astype(int)
                        )
                    elif dtype == float:
                        default_val = 1.0 if col in ["alpha", "beta"] else 0.0
                        processed_cols[col] = (
                            pd.to_numeric(df[col], errors="coerce")
                            .fillna(default_val)
                            .astype(float)
                        )
                    else:  # str
                        processed_cols[col] = df[col].astype(str)

            df_processed = pd.DataFrame(processed_cols)

            # Ensure CTR is calculated if missing or incorrect (though API should provide it)
            # Recalculate based on current impressions/clicks for consistency
            # Avoid division by zero using numpy.where or .replace
            df_processed["ctr"] = (
                (
                    df_processed["clicks"]
                    / df_processed["impressions"].replace(
                        0, pd.NA
                    )  # Replace 0 with NA for division
                )
                .fillna(0.0)
                .astype(float)
            )  # Fill NA results (from 0 impressions) with 0.0 CTR

            df_processed = df_processed.sort_values(
                by="probability_best", ascending=False
            )
            return df_processed, None
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(f"Error processing ad data: {e}", exc_info=True)
            return None, f"Error processing ad data: {e}"
    logger.warning(f"Unexpected data format received for ads stats: {type(data)}")
    return None, "Unexpected data format received for ads stats."


@st.cache_data(ttl=10)
def get_experiment_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current experiment status from the API."""
    return _make_request("GET", "/experiment/status")


@st.cache_data(ttl=10)
def get_warmup_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current warmup status from the API."""
    return _make_request("GET", "/experiment/warmup/status")


# Using PUT /experiment/config to fetch current config now
@st.cache_data(ttl=60)
def get_experiment_config() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current experiment configuration from the API via GET."""
    # Assuming the API provides a GET endpoint for the config
    data, error = _make_request("GET", "/experiment/config")
    if error:
        # If GET fails, attempt to infer from status as fallback (less reliable)
        logger.warning("GET /experiment/config failed, attempting fallback via status.")
        status_data, status_error = get_experiment_status()
        if status_error:
            # If both fail, return the original error
            return (
                None,
                f"Config GET failed ({error}) and Status fallback failed ({status_error})",
            )

        # Defaults
        config = {
            "min_samples": 1000,
            "confidence_threshold": 0.95,
            "warmup_impressions": 100,
            "simulation_count": 10000,  # Backend internal usually
        }
        if status_data:
            # Try to update defaults from status
            config["min_samples"] = status_data.get(
                "min_samples", config["min_samples"]
            )  # Assuming key exists
            config["confidence_threshold"] = status_data.get(
                "confidence_threshold", config["confidence_threshold"]
            )
            warmup_data, _ = get_warmup_status()
            if warmup_data and "warmup_impressions_per_ad" in warmup_data:
                config["warmup_impressions"] = warmup_data["warmup_impressions_per_ad"]
            logger.info("Using inferred config values from status/warmup endpoints.")
            return config, None  # Return inferred config, no error string
        else:
            return (
                None,
                f"Config GET failed ({error}), Status fallback provided no data.",
            )

    # If GET /experiment/config was successful
    if data and isinstance(data, dict):
        return data, None
    else:
        return (
            None,
            f"Invalid data format received from GET /experiment/config: {type(data)}",
        )


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
    """Delete ALL advertisements and reset winner (handled by API endpoint)."""
    # This single API call should handle resetting ads and winner based on main.py logic
    return _make_request("DELETE", "/ads")


# Removed reset_winner() function as it's redundant and the button is removed.


def update_config(
    min_samples: int,
    confidence: float,
    warmup: int,
    simulation_count: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Update experiment configuration."""
    payload = {
        "min_samples": min_samples,
        "confidence_threshold": confidence,
        "warmup_impressions": warmup,
    }
    # Only include simulation_count if provided, assuming API might allow it
    if simulation_count is not None:
        payload["simulation_count"] = simulation_count

    # Fetch existing config to potentially include unchanged values like simulation_count
    existing_config, _ = get_experiment_config()
    if (
        existing_config
        and "simulation_count" in existing_config
        and simulation_count is None
    ):
        payload["simulation_count"] = existing_config["simulation_count"]

    return _make_request("PUT", "/experiment/config", json=payload)


# --- Visualization Helper ---


def create_color_map(df: pd.DataFrame, name_col: str = "name") -> Dict[str, str]:
    """
    Creates a consistent color map for categories in a DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    name_col : str
        The name of the column containing categories (e.g., ad names).

    Returns
    -------
    Dict[str, str]
        A dictionary mapping category names to hex color codes.
    """
    if df is None or df.empty or name_col not in df.columns:
        return {}
    unique_names = df[name_col].unique()
    # Use a vibrant qualitative color scale from Plotly
    colors = px.colors.qualitative.Plotly  # Or try Vivid, Set3, etc.
    if len(unique_names) > len(colors):
        # Extend colors if more unique names than default palette size
        colors = px.colors.qualitative.Alphabet * (
            len(unique_names) // len(px.colors.qualitative.Alphabet) + 1
        )

    color_map = {name: colors[i % len(colors)] for i, name in enumerate(unique_names)}
    return color_map


# --- Streamlit UI ---

st.title("📊 Thompson Sampling Admin Panel")

# Log the API URL being used
logging.info(f"Connecting to API at: {API_BASE_URL}")

# Optional: Add a connection status check
try:
    health_response = SESSION.get(f"{API_BASE_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.caption(f"✅ API Connected")
        st.caption(f"Connected to API: {API_BASE_URL}")
    else:
        st.caption(f"⚠️ API Status: {health_response.status_code}")
except requests.exceptions.RequestException:
    st.caption("❌ API Unreachable")

# Sidebar
st.sidebar.header("🛠️ Controls")
if st.sidebar.button("🔄 Refresh Data"):
    # Clear specific caches relevant to displayed data
    st.cache_data.clear()  # Clear all for simplicity, or target specific functions
    st.rerun()

# Status placeholder
status_placeholder = st.empty()

# Fetch data - Execute all fetches upfront
ads_df, ads_error = get_ads_stats()
exp_status, exp_error = get_experiment_status()
warmup_status, warmup_error = get_warmup_status()
exp_config, config_error = get_experiment_config()  # Uses GET /experiment/config now


# Display errors
errors = {
    "Ad Statistics": ads_error,
    "Experiment Status": exp_error,
    "Warmup Status": warmup_error,
    "Experiment Config": config_error,
}
displayed_errors = {k: v for k, v in errors.items() if v}
if displayed_errors:
    error_messages = [f"Failed to load {k}: {v}" for k, v in displayed_errors.items()]
    # Use markdown for better formatting if needed
    status_placeholder.error("\n\n".join(error_messages))


# --- Main Dashboard Area ---

# Experiment Status Section
st.header("🔬 Experiment Status")
if exp_status:
    col1, col2, col3, col4 = st.columns(4)

    can_stop = exp_status.get("can_stop", False)
    confidence = exp_status.get("confidence", 0.0)
    winner = exp_status.get("winning_ad")  # Can be None or a dict
    recommendation = exp_status.get("recommendation", "N/A")
    in_warmup = exp_status.get(
        "in_warmup_phase", True
    )  # Default to True if key missing
    min_samples_reached = exp_status.get("min_samples_reached", False)
    total_impressions = exp_status.get("total_impressions", 0)

    col1.metric(
        "🏁 Can Stop Experiment?",
        "✅ Yes" if can_stop else "❌ No",
        delta=None,
        help="Indicates if the stopping criteria (min samples, confidence) are met.",
    )
    col2.metric(
        "🎯 Best Ad Confidence",
        f"{confidence:.2%}",
        delta=None,
        help="The probability that the current leading ad is truly the best.",
    )
    col3.metric(
        "🔥 In Warmup Phase?",
        "⏳ Yes" if in_warmup else "✅ No",
        delta=None,
        help="Is the experiment still ensuring minimum impressions for each ad?",
    )
    col4.metric(
        "📈 Total Impressions",
        f"{total_impressions:,}",
        delta=None,
        help="Total impressions served across all ads during the experiment.",
    )

    st.info(f"**Recommendation:** {recommendation}", icon="💡")

    # Only show success message if experiment CAN be stopped AND a winner is identified
    if can_stop and winner and isinstance(winner, dict):
        winner_name = winner.get("name", "N/A")
        winner_id = winner.get("id", "N/A")
        winner_ctr = winner.get("ctr", 0.0)
        # Use confidence from the main status, not potentially stale winner dict confidence if stored separately
        st.success(
            f"**Winner Determined:** **'{winner_name}'** (ID: `{winner_id}`) "
            f"with **{confidence:.1%}** confidence. CTR: **{winner_ctr:.3f}**",
            icon="🏆",
        )
    elif can_stop and not winner:
        # Handles the edge case where can_stop is true but API didn't return a winner object
        st.warning(
            "⚠️ Experiment stopping criteria met, but no specific winning ad identified by the API (check confidence levels).",
            icon="❓",
        )
    elif not can_stop and not in_warmup:
        # Explicitly state experiment is running post-warmup but hasn't concluded
        st.info(
            "Experiment running: Monitoring ads to reach confidence threshold or other stopping criteria.",
            icon="🏃",
        )
    elif in_warmup:
        # Message handled by the "In Warmup Phase?" metric and recommendation
        pass

else:
    # Only show warning if there wasn't a more specific error message already displayed
    if not exp_error:
        st.warning(
            "Could not load experiment status. Check API connection and logs.", icon="⚠️"
        )


# Ad Performance Section
st.header("📈 Ad Performance")

if ads_df is not None and not ads_df.empty:
    st.subheader("📊 Current Statistics")
    # Define columns to display
    display_cols = [
        "name",
        "impressions",
        "clicks",
        "ctr",
        "probability_best",
        "alpha",
        "beta",
        "id",  # Keep ID for reference
    ]
    # Filter df to only include display cols that actually exist in the processed df
    cols_to_show = [col for col in display_cols if col in ads_df.columns]
    st.dataframe(
        ads_df[cols_to_show].style.format(
            {  # Apply formatting
                "ctr": "{:.3%}",
                "probability_best": "{:.2%}",
                "alpha": "{:.2f}",
                "beta": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,  # Cleaner look
        # Add tooltips to column headers
        column_config={
            "name": st.column_config.TextColumn(
                label="Ad Name", help="Unique name of the advertisement.", max_chars=50
            ),
            "impressions": st.column_config.NumberColumn(
                label="Impressions", help="Total times the ad was shown."
            ),
            "clicks": st.column_config.NumberColumn(
                label="Clicks", help="Total times the ad was clicked."
            ),
            "ctr": st.column_config.NumberColumn(
                label="CTR",
                format="%.3f%%",
                help="Click-Through Rate (Clicks / Impressions).",
            ),  # Direct format
            "probability_best": st.column_config.NumberColumn(
                label="P(Best)",
                format="%.2f%%",
                help="Calculated probability this ad is the best based on current data.",
            ),  # Direct format
            "alpha": st.column_config.NumberColumn(
                label="Alpha",
                format="%.2f",
                help="Beta distribution parameter (Successes + 1). Higher means more evidence of success.",
            ),  # Direct format
            "beta": st.column_config.NumberColumn(
                label="Beta",
                format="%.2f",
                help="Beta distribution parameter (Failures + 1). Higher means more evidence of failure.",
            ),  # Direct format
            "id": st.column_config.TextColumn(
                label="Ad ID", help="Unique identifier for the ad."
            ),
        },
    )

    st.subheader("Visualizations")
    # Create consistent color map
    ad_color_map = create_color_map(ads_df, "name")

    col1, col2 = st.columns(2)

    # --- Plotting Area ---
    with col1:
        # Impressions Distribution
        if not ads_df["impressions"].sum() == 0:
            fig_impressions = px.pie(
                ads_df,
                values="impressions",
                names="name",
                title="Impression Distribution",
                hole=0.4,  # Make it a donut chart
                color="name",  # Use name for color mapping
                color_discrete_map=ad_color_map,  # Apply consistent colors
            )
            fig_impressions.update_traces(
                textposition="outside",
                textinfo="percent+label",
                # Custom hover template
                hovertemplate="<b>Ad:</b> %{label}<br>"
                + "<b>Impressions:</b> %{value:,}<br>"
                + "<b>Percentage:</b> %{percent:.1%}<extra></extra>",  # <extra></extra> removes trace info
            )
            fig_impressions.update_layout(
                showlegend=False,  # Legend is redundant with labels
                margin=dict(t=50, b=0, l=0, r=0),  # Adjust margin for title
            )
            st.plotly_chart(fig_impressions, use_container_width=True)
        else:
            st.info("No impressions recorded yet to plot distribution.")

        # CTR Comparison
        if not ads_df["ctr"].isnull().all() and not (ads_df["ctr"] == 0).all():
            fig_ctr = px.bar(
                ads_df.sort_values("ctr", ascending=False),  # Sort for clarity
                x="name",
                y="ctr",
                title="Measured Click-Through Rate (CTR)",
                labels={"name": "Advertisement", "ctr": "Click-Through Rate (CTR)"},
                color="name",
                color_discrete_map=ad_color_map,
                text="ctr",  # Display CTR value on bar
                custom_data=["impressions", "clicks"],  # Add data for hover
            )
            fig_ctr.update_traces(
                texttemplate="%{text:.2%}",
                textposition="outside",
                # Custom hover template
                hovertemplate="<b>Ad:</b> %{x}<br>"
                + "<b>CTR:</b> %{y:.3%}<br>"
                + "<b>Clicks:</b> %{customdata[1]:,}<br>"
                + "<b>Impressions:</b> %{customdata[0]:,}<extra></extra>",
            )
            fig_ctr.update_layout(
                yaxis_tickformat=".2%",
                xaxis_title=None,  # Remove redundant x-axis title
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                showlegend=False,  # Color applied directly
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_ctr, use_container_width=True)
        else:
            st.info("No CTR data available to plot (or all CTRs are 0).")

    with col2:
        # Probability of Being Best
        if (
            not ads_df["probability_best"].isnull().all()
            and not (ads_df["probability_best"] == 0).all()
        ):
            fig_prob = px.bar(
                ads_df.sort_values(
                    "probability_best", ascending=False
                ),  # Sort for clarity
                x="name",
                y="probability_best",
                title="Probability of Being the Best Ad",
                labels={"name": "Advertisement", "probability_best": "P(Best)"},
                color="name",
                color_discrete_map=ad_color_map,
                text="probability_best",  # Source for texttemplate
                custom_data=["alpha", "beta"],  # Correctly passed here
            )

            # --- Apply updates using for_each_trace ---
            fig_prob.for_each_trace(
                lambda t: t.update(
                    texttemplate="%{text:.1%}",  # Format the text provided by text='probability_best'
                    textposition="outside",
                    hovertemplate="<b>Ad:</b> %{x}<br>"
                    + "<b>P(Best):</b> %{y:.2%}<br>"
                    + "<b>Alpha:</b> %{customdata[0]:.2f}<br>"
                    + "<b>Beta:</b> %{customdata[1]:.2f}<extra></extra>",
                    # Note: 't' represents each trace object (go.Bar in this case)
                )
            )
            # --- End of for_each_trace update ---

            fig_prob.update_layout(
                yaxis_tickformat=".1%",
                xaxis_title=None,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                showlegend=False,  # Color applied directly
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("P(Best) data not available or all probabilities are 0.")

        # Alpha/Beta Parameters (Beta Distribution Visualization - Scatter)
        if not ads_df[["alpha", "beta"]].isnull().all().all():
            fig_params = px.scatter(
                ads_df,
                x="alpha",
                y="beta",
                size="impressions",
                color="name",
                color_discrete_map=ad_color_map,
                title="Beta Distribution Parameters (Alpha vs Beta)",
                labels={
                    "alpha": "Alpha (Successes + 1)",
                    "beta": "Beta (Failures + 1)",
                },
                size_max=40,  # Control max bubble size
                # Custom hover template for richer info
                hover_name="name",  # Use name field for main label in hover
                custom_data=[
                    "impressions",
                    "clicks",
                    "ctr",
                    "probability_best",
                ],  # Data for template
            )
            fig_params.update_traces(
                hovertemplate="<b>Ad:</b> %{hovertext}<br><br>"
                + "<b>Alpha (α):</b> %{x:.2f}<br>"
                + "<b>Beta (β):</b> %{y:.2f}<br>"
                + "<b>Impressions:</b> %{customdata[0]:,}<br>"
                + "<b>Clicks:</b> %{customdata[1]:,}<br>"
                + "<b>CTR:</b> %{customdata[2]:.3%}<br>"
                + "<b>P(Best):</b> %{customdata[3]:.2%}<extra></extra>"
            )
            fig_params.update_layout(
                legend_title_text="Advertisements",
                xaxis_title="Alpha (More Successes →)",
                yaxis_title="Beta (More Failures →)",
                margin=dict(t=50, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_params, use_container_width=True)
        else:
            st.info("Alpha/Beta parameter data not available.")

    # Time Series Note
    st.markdown(
        """
        **Note on Time Series Plots:** Visualizing metrics over time requires historical data logging.
        The current API provides the *latest* state. Consider enhancing the API or using external logging
        (like MLflow or database snapshots) for time-based analysis.
    """
    )

elif ads_df is not None and ads_df.empty:
    st.info("No advertisements created yet. Add ads via the sidebar control.", icon="ℹ️")
else:
    # Error already displayed at the top if ads_error is set
    if not ads_error:  # Handle case where df is None without an error string
        st.warning("Could not load or process ad statistics.", icon="⚠️")


# Warmup Status Section
st.header("🔥 Warmup Phase Status")
if warmup_status:
    required_warmup = warmup_status.get("warmup_impressions_per_ad", "N/A")
    if isinstance(required_warmup, (int, float)):
        st.metric(
            "Required Warmup Impressions per Ad",
            f"{required_warmup:,}",
            help="Each ad needs this many impressions before the main experiment phase begins.",
        )
    else:
        st.metric(
            "Required Warmup Impressions per Ad",
            "N/A",
            help="Could not determine required warmup impressions.",
        )

    in_warmup = warmup_status.get(
        "in_warmup_phase", False
    )  # Default to False if missing
    warmup_message = warmup_status.get("message", "Warmup status pending.")
    if in_warmup:
        st.info(f"⏳ {warmup_message}")
    else:
        st.success(f"✅ {warmup_message}")

    ads_warmup_status = warmup_status.get("ads_status", [])
    if ads_warmup_status and isinstance(ads_warmup_status, list):
        try:
            warmup_df = pd.DataFrame(ads_warmup_status)
            # Ensure required columns exist
            required_cols = [
                "name",
                "impressions",
                "required_warmup",
                "remaining",
                "warmup_complete",
            ]
            if all(col in warmup_df.columns for col in required_cols):
                # Calculate progress if not present or needs recalculation
                warmup_df["progress"] = (
                    (
                        warmup_df["impressions"]
                        / warmup_df["required_warmup"].replace(
                            0, 1
                        )  # Avoid division by zero
                    )
                    .clip(0, 1)
                    .fillna(0.0)
                )

                st.dataframe(
                    warmup_df[required_cols],  # Use defined list
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("Ad Name"),
                        "impressions": st.column_config.NumberColumn(
                            "Current Impressions"
                        ),
                        "required_warmup": st.column_config.NumberColumn("Required"),
                        "remaining": st.column_config.NumberColumn("Remaining"),
                        "warmup_complete": st.column_config.CheckboxColumn("Complete?"),
                    },
                )

                # Progress bars
                st.subheader("Warmup Progress")
                # Sort by remaining to show least progressed first
                warmup_df_sorted = warmup_df.sort_values(
                    by="remaining", ascending=False
                )
                for _, row in warmup_df_sorted.iterrows():
                    st.progress(
                        row["progress"],
                        text=f"{row['name']}: {row['impressions']:,} / {row['required_warmup']:,} ({row['progress']:.0%})",
                    )
            else:
                missing_cols = [
                    col for col in required_cols if col not in warmup_df.columns
                ]
                st.warning(
                    f"Warmup status data is missing required columns: {', '.join(missing_cols)}.",
                    icon="⚠️",
                )
        except Exception as e:
            logger.error(
                f"Failed to process warmup status dataframe: {e}", exc_info=True
            )
            st.warning(f"Could not display detailed warmup progress: {e}", icon="⚠️")
    elif ads_df is not None and not ads_df.empty:
        st.text(
            "Ad-specific warmup status not available from API (or no ads in warmup)."
        )
    # No warning if there are no ads at all
else:
    if not warmup_error:
        st.warning(
            "Could not load warmup status. Check API connection and logs.", icon="⚠️"
        )


# --- Sidebar Controls Implementation ---

st.sidebar.subheader("🔧 Manage Ads")
with st.sidebar.expander("Create New Ad", expanded=False):
    with st.form("create_ad_form"):
        new_ad_name = st.text_input(
            "Ad Name",
            help="A short, descriptive name for the ad.",
            key="new_ad_name_input",
        )
        new_ad_content = st.text_area(
            "Ad Content",
            help="The actual content/description of the ad.",
            key="new_ad_content_input",
        )
        create_submitted = st.form_submit_button("✨ Create Ad")
        if create_submitted:
            if not new_ad_name or not new_ad_content:
                st.warning("Please provide both name and content.", icon="⚠️")
            else:
                with st.spinner("Creating ad..."):
                    response, error = create_ad(new_ad_name, new_ad_content)
                if error:
                    status_placeholder.error(f"Failed to create ad: {error}", icon="❌")
                elif response:
                    ad_name = response.get(
                        "name", new_ad_name
                    )  # Use response name if available
                    ad_id = response.get("id", "N/A")
                    msg = response.get(
                        "message",
                        f"Ad '{ad_name}' created successfully (ID: `{ad_id}`)",
                    )
                    status_placeholder.success(msg, icon="✅")
                    st.cache_data.clear()
                    st.rerun()
                else:  # Should not happen if _make_request is correct
                    status_placeholder.error(
                        "Failed to create ad: Unknown error", icon="❌"
                    )


# Delete Ad Section - Improved Safety and Clarity
st.sidebar.divider()  # Visual separator
with st.sidebar.expander("Delete Ad", expanded=False):
    if ads_df is not None and not ads_df.empty:
        # Create options with name and ID for clarity
        ad_options = {
            f"{row['name']} (ID: {row['id']})": row["id"]
            for _, row in ads_df.iterrows()
        }
        selected_ad_display = st.selectbox(
            "Select Ad to Delete",
            options=list(ad_options.keys()),  # Ensure it's a list
            index=None,  # Default to no selection
            placeholder="Choose an ad...",
            key="delete_ad_select",  # Unique key for selectbox
        )

        if selected_ad_display:  # Check if an ad is selected
            ad_to_delete_id = ad_options[selected_ad_display]
            ad_to_delete_name = selected_ad_display.split(" (ID:")[
                0
            ]  # Extract name for messages

            # Use a button + checkbox confirmation pattern within the expander
            # Encapsulate button and checkbox logic
            if f"confirm_del_state_{ad_to_delete_id}" not in st.session_state:
                st.session_state[f"confirm_del_state_{ad_to_delete_id}"] = False

            clicked_delete = st.button(
                f"🗑️ Delete Ad '{ad_to_delete_name}'",
                key=f"delete_btn_{ad_to_delete_id}",
            )

            if clicked_delete:
                st.session_state[f"confirm_del_state_{ad_to_delete_id}"] = (
                    True  # Show confirmation
                )

            if st.session_state[f"confirm_del_state_{ad_to_delete_id}"]:
                confirm_delete = st.checkbox(
                    f"**Confirm deletion** of '{ad_to_delete_name}'?",
                    key=f"confirm_del_cb_{ad_to_delete_id}",
                )
                if confirm_delete:
                    with st.spinner(f"Deleting ad '{ad_to_delete_name}'..."):
                        response, error = delete_ad(ad_to_delete_id)
                    st.session_state[f"confirm_del_state_{ad_to_delete_id}"] = (
                        False  # Reset state after action
                    )
                    if error:
                        status_placeholder.error(
                            f"Failed to delete ad '{ad_to_delete_name}': {error}",
                            icon="❌",
                        )
                    elif response:
                        msg = response.get(
                            "message",
                            f"Ad '{ad_to_delete_name}' (ID: `{ad_to_delete_id}`) deleted.",
                        )
                        status_placeholder.success(msg, icon="🗑️")
                        st.cache_data.clear()
                        st.rerun()  # Rerun to reflect deletion
                    else:
                        status_placeholder.error(
                            f"Failed to delete ad '{ad_to_delete_name}': Unknown error",
                            icon="❌",
                        )

    elif ads_df is not None and ads_df.empty:
        st.info("No ads available to delete.", icon="ℹ️")
    else:
        # Only show if ads_df is None (meaning fetch failed) and no error displayed yet
        if not ads_error:
            st.warning("Ad list unavailable for deletion.", icon="⚠️")


st.sidebar.subheader("⚙️ Experiment Settings")
with st.sidebar.expander("Configure Experiment", expanded=False):
    if exp_config:
        # Use loaded config values, falling back to safe defaults if keys missing
        default_min_samples = 1000
        default_confidence = 0.95
        default_warmup = 100
        default_sim_count = exp_config.get("simulation_count", 10000)  # Usually backend

        current_min_samples = exp_config.get("min_samples", default_min_samples)
        current_confidence = exp_config.get("confidence_threshold", default_confidence)
        current_warmup = exp_config.get("warmup_impressions", default_warmup)

        # Input validation (ensure values are reasonable)
        if not isinstance(current_min_samples, int) or current_min_samples <= 0:
            logger.warning(
                f"Invalid 'min_samples' in config ({current_min_samples}), using default: {default_min_samples}"
            )
            current_min_samples = default_min_samples
        if not isinstance(current_confidence, (float, int)) or not (
            0 < current_confidence <= 1
        ):
            logger.warning(
                f"Invalid 'confidence_threshold' in config ({current_confidence}), using default: {default_confidence}"
            )
            current_confidence = default_confidence
        if not isinstance(current_warmup, int) or current_warmup < 0:
            logger.warning(
                f"Invalid 'warmup_impressions' in config ({current_warmup}), using default: {default_warmup}"
            )
            current_warmup = default_warmup

        with st.form("config_form"):
            st.caption("Update experiment stopping and warmup criteria.")
            conf_min_samples = st.number_input(
                "Min Samples per Ad (Stopping)",
                min_value=1,
                value=current_min_samples,
                help="Minimum impressions *each* ad needs before the experiment *can* stop (if confidence is also met).",
            )
            conf_confidence = st.slider(
                "Confidence Threshold (Stopping)",
                min_value=0.50,
                max_value=0.999,
                value=float(current_confidence),
                step=0.005,
                format="%.3f",
                help="Minimum P(Best) required for the leading ad to declare a winner and stop the experiment.",
            )
            conf_warmup = st.number_input(
                "Warmup Impressions per Ad",
                min_value=0,
                value=current_warmup,
                help="Impressions *each* ad must receive during the initial warmup phase.",
            )
            # Optionally allow simulation_count update if needed, but usually backend
            # conf_sim_count = st.number_input("Simulation Count (Advanced)", min_value=100, value=default_sim_count, help="Number of simulations for P(Best) calculation. Usually set on backend.")

            config_submitted = st.form_submit_button("💾 Update Configuration")

            if config_submitted:
                # Basic client-side validation before sending
                if conf_min_samples < conf_warmup:
                    st.warning(
                        "Min Samples should generally be >= Warmup Impressions.",
                        icon="⚠️",
                    )
                    # Proceed anyway, but warn the user
                # else: # Remove else if warning should not prevent submission
                with st.spinner("Updating configuration..."):
                    # Pass only the values being updated
                    response, error = update_config(
                        min_samples=conf_min_samples,
                        confidence=conf_confidence,
                        warmup=conf_warmup,
                        # simulation_count=conf_sim_count # Uncomment if adding field
                    )
                if error:
                    status_placeholder.error(
                        f"Failed to update config: {error}", icon="❌"
                    )
                elif response:
                    msg = response.get("message", "Experiment configuration updated.")
                    if isinstance(
                        response.get("config"), dict
                    ):  # Show updated config if returned
                        updated_vals = response["config"]
                        msg += f" New values: Min Samples={updated_vals.get('min_samples', 'N/A')}, Confidence={updated_vals.get('confidence_threshold', 'N/A')}, Warmup={updated_vals.get('warmup_impressions', 'N/A')}"
                    status_placeholder.success(msg, icon="✅")
                    st.cache_data.clear()  # Clear cache as config might affect status/recommendations
                    st.rerun()
                else:
                    status_placeholder.error(
                        "Failed to update config: Unknown error", icon="❌"
                    )
    else:
        # Only show if exp_config is None and no error displayed yet
        if not config_error:
            st.warning(
                "Could not load current configuration to display form.", icon="⚠️"
            )


st.sidebar.subheader("💣 Danger Zone")
with st.sidebar.expander("Reset Options", expanded=False):
    # --- Reset All Button - Uses single API call now ---
    st.markdown("**Reset ALL Ads & Winner**")
    st.caption(
        "This action uses the `DELETE /ads` endpoint, which deletes all ad data and resets the stored winner in one operation."
    )

    # State management for confirmation
    if "confirm_reset_all_state" not in st.session_state:
        st.session_state.confirm_reset_all_state = False

    reset_all_btn = st.button(
        "🚨 Reset ALL",
        key="reset_all_btn",
        type="primary",
        help="Deletes ALL ads and resets the winner via DELETE /ads. This is irreversible!",
    )

    if reset_all_btn:
        st.session_state.confirm_reset_all_state = True  # Trigger confirmation display

    if st.session_state.confirm_reset_all_state:
        st.warning(
            "**DANGER:** This will permanently delete ALL advertisements and reset the experiment winner status via a single API call. Are you sure?",
            icon="🚨",
        )
        confirm_reset_all = st.checkbox(
            "I understand the consequences and confirm resetting ALL ads and the winner.",
            key="confirm_reset_all_cb",
        )

        if confirm_reset_all:
            with st.spinner("Resetting all ads and winner..."):
                # Only call reset_all_ads() which handles both actions per main.py
                response, error = reset_all_ads()

            st.session_state.confirm_reset_all_state = False  # Reset confirmation state

            if error:
                status_placeholder.error(f"Reset failed: {error}", icon="❌")
            elif response:
                msg = response.get(
                    "message", "All advertisements and winner reset successfully."
                )
                status_placeholder.success(msg, icon="✅")
                st.cache_data.clear()
                st.rerun()
            else:
                status_placeholder.error("Reset failed: Unknown error", icon="❌")


st.sidebar.divider()
st.sidebar.caption(
    f"Page loaded/refreshed: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}"
)

# Add some space at the bottom
st.markdown("---")
