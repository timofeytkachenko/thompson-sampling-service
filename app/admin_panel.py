import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import go for potential future complex plots
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
                error_detail = e.response.json().get("detail", e.response.text)
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
            return pd.DataFrame(), None # Return empty DataFrame if no ads
        try:
            df = pd.DataFrame(data)
            # Define expected columns and their types
            expected_dtypes = {
                "id": str,
                "name": str,
                "content": str,
                "impressions": int,
                "clicks": int,
                "alpha": float,
                "beta": float,
                "ctr": float,
                "probability_best": float,
            }
            # Select and cast columns, handling missing ones gracefully
            for col, dtype in expected_dtypes.items():
                if col not in df.columns:
                    # Add missing columns with default values
                    if dtype == int:
                        df[col] = 0
                    elif dtype == float:
                        df[col] = 0.0 if col != 'alpha' and col != 'beta' else 1.0
                    else:
                        df[col] = ""
                else:
                    # Apply casting with error handling
                    if dtype == int:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif dtype == float:
                        default_val = 1.0 if col in ['alpha', 'beta'] else 0.0
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val).astype(float)
                    else:
                         df[col] = df[col].astype(str)

            # Ensure CTR is calculated if missing (though API should provide it)
            if 'ctr' not in df.columns or df['ctr'].isnull().all():
                 df['ctr'] = (df['clicks'] / df['impressions'].replace(0, 1)).fillna(0.0) # Avoid division by zero

            df = df.sort_values(by="probability_best", ascending=False)
            return df, None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error processing ad data: {e}")
            return None, f"Error processing ad data: {e}"
    return None, "Unexpected data format received for ads stats."


@st.cache_data(ttl=10)
def get_experiment_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current experiment status from the API."""
    return _make_request("GET", "/experiment/status")


@st.cache_data(ttl=10)
def get_warmup_status() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch the current warmup status from the API."""
    return _make_request("GET", "/experiment/warmup/status")


@st.cache_data(ttl=60)
def get_experiment_config() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch the current experiment configuration from the API.
    Uses defaults if a dedicated config endpoint isn't available.
    """
    # Ideal: Use a dedicated GET /experiment/config endpoint if available
    # Fallback: Infer from status or use defaults
    status_data, error = get_experiment_status()
    if error and not isinstance(error, ConnectionError): # Allow fallback if connection error
         return None, error # Propagate critical API errors

    # Defaults
    config = {
        "min_samples": 1000,
        "confidence_threshold": 0.95,
        "warmup_impressions": 100,
        "simulation_count": 10000 # Backend internal usually
    }

    # Try to update defaults from status
    if status_data:
         if "config" in status_data and isinstance(status_data["config"], dict):
             config.update(status_data["config"]) # Update with potentially nested config dict
         else: # Try root level keys if not nested
            config["min_samples"] = status_data.get("min_samples_per_ad", config["min_samples"])
            config["confidence_threshold"] = status_data.get("confidence_threshold", config["confidence_threshold"])
            # warmup_impressions often comes from warmup_status
            warmup_data, _ = get_warmup_status() # Ignore error here, fallback is fine
            if warmup_data and "warmup_impressions_per_ad" in warmup_data:
                 config["warmup_impressions"] = warmup_data["warmup_impressions_per_ad"]

    return config, None


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
        # "simulation_count": 10000, # Usually backend-controlled, omit unless needed
    }
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
    colors = px.colors.qualitative.Plotly # Or try Vivid, Set3, etc.
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(unique_names)}
    return color_map


# --- Streamlit UI ---

st.title("📊 Thompson Sampling Admin Panel")
st.caption(f"Connected to API: {API_BASE_URL}")

# Sidebar
st.sidebar.header("🛠️ Controls")
if st.sidebar.button("🔄 Refresh Data"):
    # Clear specific caches relevant to displayed data
    st.cache_data.clear() # Clear all for simplicity, or target specific functions
    st.rerun()

# Status placeholder
status_placeholder = st.empty()

# Fetch data
ads_df, ads_error = get_ads_stats()
exp_status, exp_error = get_experiment_status()
warmup_status, warmup_error = get_warmup_status()
exp_config, config_error = get_experiment_config()

# Display errors
# Consolidate error display
errors = {
    "Ad Statistics": ads_error,
    "Experiment Status": exp_error,
    "Warmup Status": warmup_error,
    "Experiment Config": config_error,
}
displayed_errors = {k: v for k, v in errors.items() if v}
if displayed_errors:
     error_messages = [f"Failed to load {k}: {v}" for k, v in displayed_errors.items()]
     status_placeholder.error("\n\n".join(error_messages))

# --- Main Dashboard Area ---

# Experiment Status Section
st.header("🔬 Experiment Status")
if exp_status:
    col1, col2, col3, col4 = st.columns(4)

    can_stop = exp_status.get("can_stop", False)
    confidence = exp_status.get("confidence", 0.0)
    winner = exp_status.get("winning_ad") # Can be None or a dict
    recommendation = exp_status.get("recommendation", "N/A")
    in_warmup = exp_status.get("in_warmup_phase", True)
    min_samples_reached = exp_status.get("min_samples_reached", False)
    total_impressions = exp_status.get("total_impressions", 0)

    col1.metric("🏁 Can Stop Experiment?", "✅ Yes" if can_stop else "❌ No", delta=None, help="Indicates if the stopping criteria (min samples, confidence) are met.")
    col2.metric("🎯 Best Ad Confidence", f"{confidence:.2%}", delta=None, help="The probability that the current leading ad is truly the best.")
    col3.metric("🔥 In Warmup Phase?", "⏳ Yes" if in_warmup else "✅ No", delta=None, help="Is the experiment still ensuring minimum impressions for each ad?")
    col4.metric("📈 Total Impressions", f"{total_impressions:,}", delta=None, help="Total impressions served across all ads during the experiment.")

    st.info(f"**Recommendation:** {recommendation}", icon="💡")

    if winner and isinstance(winner, dict):
        winner_name = winner.get('name', 'N/A')
        winner_id = winner.get('id', 'N/A')
        winner_ctr = winner.get('ctr', 0.0)
        st.success(
            f"**Winner Determined:** **'{winner_name}'** (ID: `{winner_id}`) "
            f"with **{confidence:.1%}** confidence. CTR: **{winner_ctr:.3f}**",
            icon="🏆"
        )
    elif can_stop and not winner:
        st.warning("⚠️ Experiment can be stopped, but no definitive winning ad identified yet (confidence likely borderline or multiple ads very close).")
    elif not can_stop and not in_warmup:
        st.info("🏃 Experiment running: Minimum samples reached, monitoring for winner confidence.", icon="🏃")

else:
    st.warning("Could not load experiment status. Check API connection and logs.", icon="⚠️")


# Ad Performance Section
st.header("📈 Ad Performance")

if ads_df is not None and not ads_df.empty:
    st.subheader("📊 Current Statistics")
    # Define columns to display
    display_cols = [
        "name", "impressions", "clicks", "ctr", "probability_best",
        "alpha", "beta", "id" # Keep ID for reference
    ]
    # Filter df to only include display cols that actually exist
    cols_to_show = [col for col in display_cols if col in ads_df.columns]
    st.dataframe(
        ads_df[cols_to_show].style.format({ # Apply formatting
            "ctr": "{:.3%}",
            "probability_best": "{:.2%}",
            "alpha": "{:.2f}",
            "beta": "{:.2f}",
        }),
        use_container_width=True,
        # Add tooltips to column headers
        column_config={
            "name": st.column_config.TextColumn(label="Ad Name", help="Unique name of the advertisement."),
            "impressions": st.column_config.NumberColumn(label="Impressions", help="Total times the ad was shown."),
            "clicks": st.column_config.NumberColumn(label="Clicks", help="Total times the ad was clicked."),
            "ctr": st.column_config.NumberColumn(label="CTR", help="Click-Through Rate (Clicks / Impressions)."),
            "probability_best": st.column_config.NumberColumn(label="P(Best)", help="Calculated probability this ad is the best based on current data."),
            "alpha": st.column_config.NumberColumn(label="Alpha", help="Beta distribution parameter (Successes + 1). Higher means more evidence of success."),
            "beta": st.column_config.NumberColumn(label="Beta", help="Beta distribution parameter (Failures + 1). Higher means more evidence of failure."),
            "id": st.column_config.TextColumn(label="Ad ID", help="Unique identifier for the ad."),
        }
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
                hole=0.4, # Make it a donut chart
                color="name", # Use name for color mapping
                color_discrete_map=ad_color_map, # Apply consistent colors
            )
            fig_impressions.update_traces(
                textposition="outside",
                textinfo="percent+label",
                # Custom hover template
                hovertemplate="<b>Ad:</b> %{label}<br>" +
                              "<b>Impressions:</b> %{value:,}<br>" +
                              "<b>Percentage:</b> %{percent:.1%}<extra></extra>" # <extra></extra> removes trace info
            )
            fig_impressions.update_layout(showlegend=False) # Legend is redundant with labels
            st.plotly_chart(fig_impressions, use_container_width=True)
        else:
            st.info("No impressions recorded yet to plot distribution.")

        # CTR Comparison
        fig_ctr = px.bar(
            ads_df.sort_values("ctr", ascending=False), # Sort for clarity
            x="name",
            y="ctr",
            title="Measured Click-Through Rate (CTR)",
            labels={"name": "Advertisement", "ctr": "Click-Through Rate (CTR)"},
            color="name",
            color_discrete_map=ad_color_map,
            text="ctr", # Display CTR value on bar
            custom_data=["impressions", "clicks"] # Add data for hover
        )
        fig_ctr.update_traces(
            texttemplate="%{text:.2%}",
            textposition="outside",
             # Custom hover template
            hovertemplate="<b>Ad:</b> %{x}<br>" +
                          "<b>CTR:</b> %{y:.3%}<br>" +
                          "<b>Clicks:</b> %{customdata[1]:,}<br>" +
                          "<b>Impressions:</b> %{customdata[0]:,}<extra></extra>"
        )
        fig_ctr.update_layout(
            yaxis_tickformat=".2%",
            xaxis_title=None, # Remove redundant x-axis title
            uniformtext_minsize=8,
            uniformtext_mode='hide'
        )
        st.plotly_chart(fig_ctr, use_container_width=True)

    with col2:
        # Probability of Being Best
        fig_prob = px.bar(
            ads_df.sort_values("probability_best", ascending=False), # Sort for clarity
            x="name",
            y="probability_best",
            title="Probability of Being the Best Ad",
            labels={"name": "Advertisement", "probability_best": "P(Best)"},
            color="name",
            color_discrete_map=ad_color_map,
            text="probability_best", # Source for texttemplate
            custom_data=['alpha', 'beta'] # Correctly passed here
        )

        # --- Apply updates using for_each_trace ---
        fig_prob.for_each_trace(lambda t: t.update(
            texttemplate="%{text:.1%}", # Format the text provided by text='probability_best'
            textposition="outside",
            hovertemplate="<b>Ad:</b> %{x}<br>" +
                          "<b>P(Best):</b> %{y:.2%}<br>" +
                          "<b>Alpha:</b> %{customdata[0]:.2f}<br>" +
                          "<b>Beta:</b> %{customdata[1]:.2f}<extra></extra>"
            # Note: 't' represents each trace object (go.Bar in this case)
        ))
        # --- End of for_each_trace update ---

        fig_prob.update_layout(
            yaxis_tickformat=".1%",
            xaxis_title=None,
             uniformtext_minsize=8,
             uniformtext_mode='hide'
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Alpha/Beta Parameters (Beta Distribution Visualization - Scatter)
        fig_params = px.scatter(
            ads_df,
            x="alpha",
            y="beta",
            size="impressions",
            color="name",
            color_discrete_map=ad_color_map,
            title="Beta Distribution Parameters (Alpha vs Beta)",
            labels={"alpha": "Alpha (Successes + 1)", "beta": "Beta (Failures + 1)"},
            size_max=40, # Control max bubble size
             # Custom hover template for richer info
            hover_name="name", # Use name field for main label in hover
            custom_data=["impressions", "clicks", "ctr", "probability_best"] # Data for template
        )
        fig_params.update_traces(
             hovertemplate="<b>Ad:</b> %{hovertext}<br><br>" +
                           "<b>Alpha (α):</b> %{x:.2f}<br>" +
                           "<b>Beta (β):</b> %{y:.2f}<br>" +
                           "<b>Impressions:</b> %{customdata[0]:,}<br>" +
                           "<b>Clicks:</b> %{customdata[1]:,}<br>" +
                           "<b>CTR:</b> %{customdata[2]:.3%}<br>" +
                           "<b>P(Best):</b> %{customdata[3]:.2%}<extra></extra>"
        )
        fig_params.update_layout(
            legend_title_text='Advertisements',
            xaxis_title="Alpha (More Successes →)",
            yaxis_title="Beta (More Failures →)"
        )
        st.plotly_chart(fig_params, use_container_width=True)

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
    if not ads_error: # Handle case where df is None without an error string
        st.warning("Could not load or process ad statistics.", icon="⚠️")


# Warmup Status Section
st.header("🔥 Warmup Phase Status")
if warmup_status:
    required_warmup = warmup_status.get("warmup_impressions_per_ad", "N/A")
    st.metric("Required Warmup Impressions per Ad", required_warmup, help="Each ad needs this many impressions before the main experiment phase begins.")

    in_warmup = warmup_status.get("in_warmup_phase", True)
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
            if all(col in warmup_df.columns for col in ["impressions", "required_warmup"]):
                 warmup_df["progress"] = (
                     warmup_df["impressions"] / warmup_df["required_warmup"].replace(0, 1) # Avoid division by zero
                 ).clip(0, 1)
                 warmup_df["remaining"] = (warmup_df["required_warmup"] - warmup_df["impressions"]).clip(0, None) # Ensure non-negative

                 st.dataframe(
                     warmup_df[
                         [
                             "name",
                             "impressions",
                             "required_warmup",
                             "remaining",
                             "warmup_complete",
                         ]
                     ],
                     use_container_width=True,
                     column_config={
                         "name": st.column_config.TextColumn("Ad Name"),
                         "impressions": st.column_config.NumberColumn("Current Impressions"),
                         "required_warmup": st.column_config.NumberColumn("Required Impressions"),
                         "remaining": st.column_config.NumberColumn("Impressions Remaining"),
                         "warmup_complete": st.column_config.CheckboxColumn("Warmup Complete?"),
                     }
                 )

                 # Progress bars
                 st.subheader("Warmup Progress")
                 for _, row in warmup_df.iterrows():
                     st.progress(
                         row["progress"],
                         text=f"{row['name']}: {row['impressions']:,} / {row['required_warmup']:,} ({row['progress']:.0%})"
                     )
            else:
                 st.warning("Warmup status data is missing required columns (impressions, required_warmup).", icon="⚠️")
        except Exception as e:
            logger.error(f"Failed to process warmup status dataframe: {e}")
            st.warning(f"Could not display detailed warmup progress: {e}", icon="⚠️")
    elif ads_df is not None and not ads_df.empty:
         st.text("Ad-specific warmup status not available from API.")
    # No warning if there are no ads at all
else:
    st.warning("Could not load warmup status. Check API connection and logs.", icon="⚠️")


# --- Sidebar Controls Implementation ---

st.sidebar.subheader("🔧 Manage Ads")
with st.sidebar.expander("Create New Ad", expanded=False):
    with st.form("create_ad_form"):
        new_ad_name = st.text_input("Ad Name", help="A short, descriptive name for the ad.")
        new_ad_content = st.text_area("Ad Content", help="The actual content/description of the ad.")
        create_submitted = st.form_submit_button("✨ Create Ad")
        if create_submitted:
            if not new_ad_name or not new_ad_content:
                st.warning("Please provide both name and content.", icon="⚠️")
            else:
                with st.spinner("Creating ad..."):
                    response, error = create_ad(new_ad_name, new_ad_content)
                if error:
                    status_placeholder.error(f"Failed to create ad: {error}", icon="❌")
                else:
                    ad_name = response.get('name', new_ad_name) # Use response name if available
                    ad_id = response.get('id', 'N/A')
                    status_placeholder.success(
                        f"Ad '{ad_name}' created successfully (ID: `{ad_id}`)", icon="✅"
                    )
                    st.cache_data.clear()
                    st.rerun()

# Delete Ad Section - Improved Safety and Clarity
st.sidebar.divider() # Visual separator
with st.sidebar.expander("Delete Ad", expanded=False):
    if ads_df is not None and not ads_df.empty:
        # Create options with name and ID for clarity
        ad_options = {f"{row['name']} (ID: {row['id']})": row['id'] for _, row in ads_df.iterrows()}
        selected_ad_display = st.selectbox(
            "Select Ad to Delete",
            options=ad_options.keys(),
            key="delete_ad_select" # Unique key for selectbox
        )

        if selected_ad_display: # Check if an ad is selected
            ad_to_delete_id = ad_options[selected_ad_display]
            ad_to_delete_name = selected_ad_display.split(" (ID:")[0] # Extract name for messages

            # Use a button + checkbox confirmation pattern within the expander
            delete_button_placeholder = st.empty()
            confirm_placeholder = st.empty()

            if delete_button_placeholder.button(f"🗑️ Delete Ad '{ad_to_delete_name}'", key=f"delete_btn_{ad_to_delete_id}", type="secondary"):
                 # Show confirmation checkbox only after delete button is clicked
                 confirm_delete = confirm_placeholder.checkbox(
                     f"**Confirm deletion** of ad '{ad_to_delete_name}'?",
                     key=f"confirm_del_{ad_to_delete_id}"
                 )
                 if confirm_delete:
                     with st.spinner(f"Deleting ad '{ad_to_delete_name}'..."):
                         _, error = delete_ad(ad_to_delete_id)
                     if error:
                         status_placeholder.error(f"Failed to delete ad '{ad_to_delete_name}': {error}", icon="❌")
                         # Clear the button/checkbox on error to reset state
                         delete_button_placeholder.empty()
                         confirm_placeholder.empty()
                     else:
                         status_placeholder.success(f"Ad '{ad_to_delete_name}' (ID: `{ad_to_delete_id}`) deleted.", icon="🗑️")
                         st.cache_data.clear()
                         st.rerun() # Rerun to reflect deletion
                 # If checkbox is shown but not checked, do nothing until next interaction
            # Else: Button not clicked, show nothing in confirm_placeholder


    elif ads_df is not None and ads_df.empty:
        st.info("No ads available to delete.", icon="ℹ️")
    else:
        st.warning("Ad list unavailable for deletion.", icon="⚠️")


st.sidebar.subheader("⚙️ Experiment Settings")
with st.sidebar.expander("Configure Experiment", expanded=False):
    if exp_config:
        # Use loaded config values, falling back to safe defaults if keys missing
        default_min_samples = 1000
        default_confidence = 0.95
        default_warmup = 100

        current_min_samples = exp_config.get("min_samples", default_min_samples)
        current_confidence = exp_config.get("confidence_threshold", default_confidence)
        current_warmup = exp_config.get("warmup_impressions", default_warmup)

        # Input validation (ensure values are reasonable)
        if not isinstance(current_min_samples, int) or current_min_samples < 0:
             current_min_samples = default_min_samples
             st.warning(f"Invalid 'min_samples' in config, using default: {default_min_samples}", icon="⚠️")
        if not isinstance(current_confidence, (float, int)) or not (0 < current_confidence < 1):
            current_confidence = default_confidence
            st.warning(f"Invalid 'confidence_threshold' in config, using default: {default_confidence}", icon="⚠️")
        if not isinstance(current_warmup, int) or current_warmup < 0:
            current_warmup = default_warmup
            st.warning(f"Invalid 'warmup_impressions' in config, using default: {default_warmup}", icon="⚠️")


        with st.form("config_form"):
            conf_min_samples = st.number_input(
                "Min Samples per Ad (Stopping)", min_value=1, value=current_min_samples,
                help="Minimum impressions *each* ad needs before the experiment *can* stop (if confidence is also met)."
            )
            conf_confidence = st.slider(
                "Confidence Threshold (Stopping)",
                min_value=0.50, max_value=0.999, value=float(current_confidence), step=0.005, format="%.3f",
                help="Minimum P(Best) required for the leading ad to declare a winner and stop the experiment."
            )
            conf_warmup = st.number_input(
                "Warmup Impressions per Ad", min_value=0, value=current_warmup,
                help="Impressions *each* ad must receive during the initial warmup phase."
            )
            config_submitted = st.form_submit_button("💾 Update Configuration")

            if config_submitted:
                 # Basic client-side validation before sending
                 if conf_min_samples < conf_warmup:
                     st.warning("Min Samples should generally be >= Warmup Impressions.", icon="⚠️")
                 else:
                    with st.spinner("Updating configuration..."):
                        response, error = update_config(
                            conf_min_samples, conf_confidence, conf_warmup
                        )
                    if error:
                        status_placeholder.error(f"Failed to update config: {error}", icon="❌")
                    else:
                        status_placeholder.success("✅ Experiment configuration updated.", icon="✅")
                        st.cache_data.clear() # Clear cache as config might affect status/recommendations
                        st.rerun()
    else:
        st.warning("Could not load current configuration to display form.", icon="⚠️")


st.sidebar.subheader("💣 Danger Zone")
with st.sidebar.expander("Reset Options", expanded=False):

    # Reset Winner Button
    reset_winner_btn = st.button("⚠️ Reset Stored Winner", key="reset_winner_btn", type="secondary", help="Clears the winner status, allowing the experiment to re-evaluate.")
    if reset_winner_btn:
        # Confirmation required via checkbox
        confirm_reset_winner = st.checkbox("Confirm resetting the stored winner?", key="confirm_reset_winner")
        if confirm_reset_winner:
            with st.spinner("Resetting winner..."):
                response, error = reset_winner()
            if error:
                status_placeholder.error(f"Failed to reset winner: {error}", icon="❌")
            else:
                status_placeholder.success(
                    response.get("message", "Winner reset successfully.") if response else "Winner reset successfully.", icon="✅"
                )
                st.cache_data.clear()
                st.rerun()
        else:
            # Give feedback if clicked but not confirmed
            st.warning("Winner reset requires confirmation via the checkbox above.", icon="⚠️")

    st.divider() # Separator

    # Reset All Button - Increased emphasis on danger
    reset_all_btn = st.button("🚨 Reset ALL Ads & Winner", key="reset_all_btn", type="primary", help="Deletes ALL ads and resets the winner. This is irreversible!")
    if reset_all_btn:
        st.warning(
            "**DANGER:** This action will permanently delete ALL advertisements and reset the experiment winner status. Are you sure?", icon="🚨"
        )
        # Requires explicit confirmation
        confirm_reset_all = st.checkbox("I understand the consequences and confirm resetting ALL ads and the winner.", key="confirm_reset_all")
        if confirm_reset_all:
            errors_during_reset = []
            with st.spinner("Resetting all ads..."):
                _, ad_error = reset_all_ads()
                if ad_error: errors_during_reset.append(f"Ads reset failed: {ad_error}")

            with st.spinner("Resetting winner..."):
                _, winner_error = reset_winner()
                if winner_error: errors_during_reset.append(f"Winner reset failed: {winner_error}")

            if errors_during_reset:
                status_placeholder.error(
                    f"Partial or full reset failed:\n- " + "\n- ".join(errors_during_reset), icon="❌"
                )
            else:
                status_placeholder.success("✅ All advertisements and winner reset successfully.", icon="✅")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Full reset requires confirmation via the checkbox above.", icon="⚠️")


st.sidebar.divider()
st.sidebar.caption(
    f"Page loaded/refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)