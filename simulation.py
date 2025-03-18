import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter
from tqdm import tqdm

# Create output directories
RESULTS_DIR = "my_results"
LOG_DIR = f"{RESULTS_DIR}/logs"
CHARTS_DIR = f"{RESULTS_DIR}/charts"
DATA_DIR = f"{RESULTS_DIR}/data"

for directory in [RESULTS_DIR, LOG_DIR, CHARTS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOG_DIR}/simulation_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ThompsonSamplingSimulator:
    """
    Enhanced simulator for testing Thompson Sampling microservice.

    This class provides comprehensive functionality to simulate traffic to a Thompson Sampling
    microservice for A/B testing, with improved visualization, statistics tracking,
    and warmup phase configuration.
    """

    def __init__(self, base_url: str, simulation_config: Dict):
        """
        Initialize the simulator with configuration parameters.

        Parameters
        ----------
        base_url : str
            Base URL of the Thompson Sampling API microservice
        simulation_config : Dict
            Configuration dictionary for the simulation
        """
        self.base_url = base_url.rstrip("/")
        self.config = simulation_config
        self.ads: Dict[str, Dict] = {}  # Dictionary of ads: {ad_id: ad_info}
        self.results: List[Dict] = []  # History of results
        self.session = requests.Session()  # HTTP session for requests
        self.status_history: List[Dict] = []  # Track experiment status over time
        self.ads_stats_history: List[Dict] = (
            []
        )  # Track detailed ad statistics over time
        self.winner_determined_at: Optional[int] = (
            None  # Iteration when winner was determined
        )
        self.warmup_completed_at: Optional[int] = (
            None  # Iteration when warmup phase completed
        )

        logger.info("🚀 Initializing Thompson Sampling simulator")
        logger.info(f"📊 Results will be saved to {os.path.abspath(RESULTS_DIR)}")
        logger.info(f"📝 Log file: {os.path.abspath(log_file)}")

    def reset_service(self) -> bool:
        """
        Reset the microservice (delete all advertisements).

        Returns
        -------
        bool
            True if reset successful, False otherwise
        """
        try:
            response = self.session.delete(f"{self.base_url}/ads")
            logger.info(f"🧹 Service reset response: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Failed to reset service: {e}")
            return False

    def configure_experiment(
        self, min_samples: int, confidence_threshold: float, warmup_impressions: int
    ) -> bool:
        """
        Configure experiment parameters.

        Parameters
        ----------
        min_samples : int
            Minimum number of impressions before stopping
        confidence_threshold : float
            Probability threshold for determining a winner
        warmup_impressions : int
            Number of impressions per ad during warmup phase

        Returns
        -------
        bool
            True if configuration successful, False otherwise
        """
        try:
            config = {
                "min_samples": min_samples,
                "confidence_threshold": confidence_threshold,
                "simulation_count": 10000,
                "warmup_impressions": warmup_impressions,
                "respect_winner": True,
            }
            response = self.session.put(
                f"{self.base_url}/experiment/config", json=config
            )
            logger.info(f"⚙️ Experiment config set to: {json.dumps(config, indent=2)}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Failed to configure experiment: {e}")
            return False

    def create_ads(self, ads_config: List[Dict]) -> bool:
        """
        Create advertisements based on configuration.

        Parameters
        ----------
        ads_config : List[Dict]
            List of advertisement configurations

        Returns
        -------
        bool
            True if all ads created successfully, False otherwise
        """
        success = True
        self.ads = {}

        logger.info(f"📝 Creating {len(ads_config)} advertisements")

        for ad_config in ads_config:
            try:
                response = self.session.post(
                    f"{self.base_url}/ads",
                    json={"name": ad_config["name"], "content": ad_config["content"]},
                )

                if response.status_code == 201:
                    ad_data = response.json()
                    ad_id = ad_data["id"]

                    # Save ad data
                    self.ads[ad_id] = {
                        "id": ad_id,
                        "name": ad_config["name"],
                        "content": ad_config["content"],
                        "ctr": ad_config[
                            "ctr"
                        ],  # True click probability (for simulation)
                        "impressions": 0,
                        "clicks": 0,
                        "selection_counts": {},  # Track selection methods
                    }

                    logger.info(
                        f"✅ Created ad: {ad_id} - {ad_config['name']} (CTR: {ad_config['ctr']:.2%})"
                    )
                else:
                    logger.error(
                        f"❌ Failed to create ad {ad_config['name']}: {response.status_code} - {response.text}"
                    )
                    success = False

            except Exception as e:
                logger.error(f"❌ Error creating ad {ad_config['name']}: {e}")
                success = False

        return success

    def simulate_traffic(
        self,
        total_requests: int,
        respect_winner: bool = True,
        check_interval: int = 100,
    ) -> Tuple[Dict, List[Dict]]:
        """
        Simulate traffic with impressions and clicks.

        Parameters
        ----------
        total_requests : int
            Total number of requests to simulate
        respect_winner : bool, optional
            Whether to respect the experiment winner when selecting ads, by default True
        check_interval : int, optional
            Interval for checking experiment status, by default 100

        Returns
        -------
        Tuple[Dict, List[Dict]]
            Final statistics and history of impressions/clicks
        """
        # Metrics for each ad
        metrics = {ad_id: {"impressions": 0, "clicks": 0} for ad_id in self.ads}

        # Track selection methods
        selection_methods = {}

        # History of ad selections
        history = []

        # Start time for performance tracking
        start_time = time.time()
        last_check_time = start_time

        # Status for display
        winner_found = False
        winner_id = None
        winner_name = None
        experiment_status = None
        warmup_complete = False
        warmup_completed_at = None  # Iteration when warmup was completed

        # Progress bar with rich formatting
        with tqdm(
            total=total_requests,
            desc="🔄 Simulating traffic",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i in range(total_requests):
                try:
                    # Get ad recommendation
                    response = self.session.get(
                        f"{self.base_url}/ads/select?respect_winner={str(respect_winner).lower()}"
                    )

                    if response.status_code != 200:
                        logger.error(
                            f"❌ Failed to select ad: {response.status_code} - {response.text}"
                        )
                        continue

                    ad_data = response.json()
                    ad_id = ad_data["id"]
                    selection_method = ad_data.get("selection_method", "unknown")

                    # Track selection method
                    if selection_method not in selection_methods:
                        selection_methods[selection_method] = 0
                    selection_methods[selection_method] += 1

                    # Track per-ad selection method
                    if ad_id in self.ads:
                        if selection_method not in self.ads[ad_id]["selection_counts"]:
                            self.ads[ad_id]["selection_counts"][selection_method] = 0
                        self.ads[ad_id]["selection_counts"][selection_method] += 1

                    if ad_id not in self.ads:
                        logger.warning(f"⚠️ Unknown ad selected: {ad_id}")
                        continue

                    # Increment impression counter
                    metrics[ad_id]["impressions"] += 1

                    # Simulate click based on true CTR
                    is_clicked = random.random() < self.ads[ad_id]["ctr"]

                    if is_clicked:
                        # If clicked, send click event
                        click_response = self.session.post(
                            f"{self.base_url}/ads/click", json={"ad_id": ad_id}
                        )

                        if click_response.status_code == 200:
                            metrics[ad_id]["clicks"] += 1
                        else:
                            logger.error(
                                f"❌ Failed to record click: {click_response.status_code} - {click_response.text}"
                            )
                    else:
                        # If not clicked, send no-click event
                        no_click_response = self.session.post(
                            f"{self.base_url}/ads/no-click/{ad_id}"
                        )

                        if no_click_response.status_code != 200:
                            logger.error(
                                f"❌ Failed to record no-click: {no_click_response.status_code} - {no_click_response.text}"
                            )

                    # Add experiment phase and winner info to history record
                    phase = "warmup"
                    if warmup_complete:
                        phase = "testing"
                    if winner_found:
                        phase = "post_winner"

                    winner_info = None
                    if winner_found:
                        winner_info = {
                            "winner_id": winner_id,
                            "winner_name": winner_name,
                        }

                    # Record history
                    history.append(
                        {
                            "iteration": i + 1,
                            "ad_id": ad_id,
                            "ad_name": self.ads[ad_id]["name"],
                            "is_clicked": is_clicked,
                            "selection_method": selection_method,
                            "timestamp": time.time(),
                            "phase": phase,
                            "warmup_complete": warmup_complete,
                            "winner_determined": winner_found,
                            "winner_info": winner_info,
                        }
                    )

                    # Check experiment status at regular intervals
                    if (i + 1) % check_interval == 0 or i == total_requests - 1:
                        experiment_status = self.check_experiment_status()

                        # Also get detailed statistics for each ad
                        ads_stats = self.get_ads_statistics()

                        # Store statistics with timestamp
                        if ads_stats:
                            stats_record = {
                                "iteration": i + 1,
                                "timestamp": time.time(),
                                "ads_stats": ads_stats,
                            }
                            self.ads_stats_history.append(stats_record)

                        # Keep track of the status in history with timestamp
                        if experiment_status:
                            # Correctly determine warmup status from in_warmup_phase
                            new_warmup_complete = not experiment_status.get(
                                "in_warmup_phase", True
                            )

                            # Check if warmup has just completed
                            if new_warmup_complete and not warmup_complete:
                                warmup_complete = True
                                self.warmup_completed_at = i + 1
                                # Log the warmup completion with statistics
                                logger.info(f"\n{'=' * 80}")
                                logger.info(
                                    f"🚀 WARMUP PHASE COMPLETED AT ITERATION {i + 1}"
                                )
                                logger.info(f"{'=' * 80}")
                                # Log impression distribution at warmup completion
                                logger.info(
                                    "📊 Impression distribution at warmup completion:"
                                )
                                for ad_id, ad_info in self.ads.items():
                                    ad_impressions = metrics[ad_id]["impressions"]
                                    ad_clicks = metrics[ad_id]["clicks"]
                                    ad_ctr = (
                                        ad_clicks / ad_impressions
                                        if ad_impressions > 0
                                        else 0
                                    )
                                    logger.info(
                                        f"  - {ad_info['name']}: {ad_impressions} impressions, {ad_clicks} clicks, CTR: {ad_ctr:.4f}"
                                    )
                                logger.info(f"{'=' * 80}\n")
                            else:
                                warmup_complete = new_warmup_complete

                            # Add derived warmup_complete flag to status
                            experiment_status["warmup_complete"] = warmup_complete

                            status_record = {
                                "iteration": i + 1,
                                "timestamp": time.time(),
                                "status": experiment_status,
                            }
                            self.status_history.append(status_record)

                            # Update status indicators
                            can_stop = experiment_status.get("can_stop", False)

                            if can_stop and not winner_found:
                                winner_found = True
                                winner_id = experiment_status.get("winning_ad", {}).get(
                                    "id", "Unknown"
                                )
                                winner_name = experiment_status.get(
                                    "winning_ad", {}
                                ).get("name", "Unknown")
                                confidence = experiment_status.get("confidence", 0)
                                self.winner_determined_at = i + 1

                                # Log detailed winner statistics
                                logger.info(f"\n{'=' * 80}")
                                logger.info(
                                    f"🏆 WINNER DETERMINED AT ITERATION {i + 1}: {winner_name}"
                                )
                                logger.info(f"{'=' * 80}")
                                logger.info(f"• Confidence: {confidence:.2%}")
                                logger.info(
                                    f"• Respect winner setting: {respect_winner}"
                                )

                                # Get statistics for all ads at winner determination
                                logger.info(
                                    "\n📊 Ad Statistics at Winner Determination:"
                                )
                                for ad_id, ad_info in self.ads.items():
                                    ad_impressions = metrics[ad_id]["impressions"]
                                    ad_clicks = metrics[ad_id]["clicks"]
                                    ad_ctr = (
                                        ad_clicks / ad_impressions
                                        if ad_impressions > 0
                                        else 0
                                    )
                                    true_ctr = ad_info["ctr"]
                                    is_winner = "🏆 " if ad_id == winner_id else "   "

                                    # Calculate CTR error
                                    ctr_error = (
                                        ((ad_ctr - true_ctr) / true_ctr) * 100
                                        if true_ctr > 0
                                        else 0
                                    )
                                    error_sign = "+" if ctr_error >= 0 else ""

                                    logger.info(
                                        f"{is_winner}{ad_info['name']} ({ad_id}):"
                                    )
                                    logger.info(
                                        f"    • Impressions: {ad_impressions} | Clicks: {ad_clicks}"
                                    )
                                    logger.info(
                                        f"    • Measured CTR: {ad_ctr:.4f} | True CTR: {true_ctr:.4f} | Error: {error_sign}{ctr_error:.2f}%"
                                    )

                                if respect_winner:
                                    logger.info(
                                        f"\n⚠️ Since respect_winner=True, only the winning ad ({winner_name}) will be shown from now on."
                                    )
                                else:
                                    logger.info(
                                        f"\n⚠️ Since respect_winner=False, Thompson Sampling will continue for all ads despite finding a winner."
                                    )

                                logger.info(f"{'=' * 80}\n")

                        # Update progress bar description
                        elapsed = time.time() - last_check_time
                        last_check_time = time.time()

                        # Prepare status message
                        status_msg = "🔄 Simulating"
                        if not warmup_complete:
                            status_msg = "🔄 Warmup phase"
                        elif winner_found:
                            status_msg = "🏆 Winner found"

                        pbar.set_description(
                            f"{status_msg} ({i + 1}/{total_requests}, {check_interval / elapsed:.1f} req/s)"
                        )

                        # Print detailed statistics at checkpoints
                        if i > 0 and (
                            (i + 1) % (check_interval * 5) == 0
                            or i == total_requests - 1
                        ):
                            self.print_interim_stats(metrics, i + 1)

                    # Update progress bar
                    pbar.update(1)

                    # Small delay to avoid API overload
                    time.sleep(0.01)

                except Exception as e:
                    logger.error(f"❌ Error in simulation iteration {i}: {e}")

        # Get final statistics
        for ad_id in self.ads:
            total_impressions = metrics[ad_id]["impressions"]
            total_clicks = metrics[ad_id]["clicks"]

            self.ads[ad_id]["impressions"] = total_impressions
            self.ads[ad_id]["clicks"] = total_clicks
            self.ads[ad_id]["actual_ctr"] = (
                total_clicks / total_impressions if total_impressions > 0 else 0
            )

        # Final experiment status check
        final_status = self.check_experiment_status()
        final_ads_stats = self.get_ads_statistics()

        # Save the raw simulation data
        self.save_simulation_data(
            history, metrics, selection_methods, final_status, final_ads_stats
        )

        # Log selection method distribution
        logger.info("📊 Selection method distribution:")
        total_selections = sum(selection_methods.values())
        for method, count in selection_methods.items():
            percentage = count / total_selections * 100 if total_selections > 0 else 0
            logger.info(f"  - {method}: {count} selections ({percentage:.1f}%)")

        return self.ads, history

    def check_experiment_status(self) -> Dict:
        """
        Check the experiment status.

        Returns
        -------
        Dict
            Experiment status information
        """
        try:
            response = self.session.get(f"{self.base_url}/experiment/status")

            if response.status_code == 200:
                status = response.json()
                can_stop = status.get("can_stop", False)

                if "winning_ad" in status and status["winning_ad"]:
                    winning_ad_id = status["winning_ad"]["id"]
                    winning_ad_name = status["winning_ad"]["name"]
                    confidence = status["confidence"]

                    if can_stop:
                        logger.info(
                            f"🏆 Experiment can be stopped. Winner: {winning_ad_name} ({winning_ad_id}) with {confidence:.2%} confidence"
                        )

                return status
            else:
                logger.error(
                    f"❌ Failed to check experiment status: {response.status_code} - {response.text}"
                )
                return {}

        except Exception as e:
            logger.error(f"❌ Error checking experiment status: {e}")
            return {}

    def get_ads_statistics(self) -> List[Dict]:
        """
        Get statistics for all advertisements.

        Returns
        -------
        List[Dict]
            List of ad statistics
        """
        try:
            response = self.session.get(f"{self.base_url}/ads")

            if response.status_code == 200:
                ads_stats = response.json()
                return ads_stats
            else:
                logger.error(
                    f"❌ Failed to get ads statistics: {response.status_code} - {response.text}"
                )
                return []

        except Exception as e:
            logger.error(f"❌ Error getting ads statistics: {e}")
            return []

    def print_interim_stats(self, metrics: Dict, iteration: int) -> None:
        """
        Print interim statistics during simulation.

        Parameters
        ----------
        metrics : Dict
            Current metrics for each ad
        iteration : int
            Current iteration number
        """
        logger.info(
            f"\n{'=' * 80}\n📊 STATISTICS AT ITERATION {iteration}:\n{'=' * 80}"
        )

        # Print ad statistics
        logger.info("📈 Ad Performance:")
        for ad_id, ad_info in self.ads.items():
            impressions = metrics[ad_id]["impressions"]
            clicks = metrics[ad_id]["clicks"]
            actual_ctr = clicks / impressions if impressions > 0 else 0
            true_ctr = ad_info["ctr"]

            # Calculate CTR error
            ctr_error = (
                ((actual_ctr - true_ctr) / true_ctr) * 100 if true_ctr > 0 else 0
            )
            error_sign = "+" if ctr_error >= 0 else ""

            # Get selection method breakdown
            selection_methods = []
            if "selection_counts" in ad_info:
                for method, count in ad_info["selection_counts"].items():
                    percentage = count / sum(ad_info["selection_counts"].values()) * 100
                    selection_methods.append(f"{method}: {percentage:.1f}%")

            logger.info(
                f"  - {ad_info['name']} ({ad_id}):\n"
                f"    • Impressions: {impressions} | Clicks: {clicks}\n"
                f"    • Actual CTR: {actual_ctr:.4f} | True CTR: {true_ctr:.4f} | Error: {error_sign}{ctr_error:.2f}%\n"
                f"    • Selection methods: {', '.join(selection_methods)}"
            )

        # Print experiment status if available
        if self.status_history:
            latest_status = self.status_history[-1]["status"]
            logger.info("\n📋 Latest Experiment Status:")

            can_stop = latest_status.get("can_stop", False)
            # Use the derived warmup_complete flag we added to the status
            warmup_complete = latest_status.get("warmup_complete", False)
            confidence = latest_status.get("confidence", 0)
            winning_ad = latest_status.get("winning_ad", None)

            status_emoji = "🔄"
            if can_stop:
                status_emoji = "✅"
            elif not warmup_complete:
                status_emoji = "⏳"

            logger.info(f"  {status_emoji} Can stop: {can_stop}")
            logger.info(
                f"  {'✅' if warmup_complete else '⏳'} Warmup complete: {warmup_complete}"
            )
            logger.info(f"  📊 Confidence level: {confidence:.2%}")

            if winning_ad:
                logger.info(
                    f"  🏆 Current best ad: {winning_ad['name']} (ID: {winning_ad['id']})"
                )
                logger.info(
                    f"     CTR: {winning_ad['ctr']:.4f} | Prob. Best: {winning_ad['probability_best']:.4f}"
                )

            logger.info(
                f"  💡 Recommendation: {latest_status.get('recommendation', 'None')}"
            )

        logger.info(f"{'=' * 80}\n")

    def save_simulation_data(
        self,
        history: List[Dict],
        metrics: Dict,
        selection_methods: Dict,
        final_status: Dict,
        final_ads_stats: List[Dict],
    ) -> None:
        """
        Save simulation data to files.

        Parameters
        ----------
        history : List[Dict]
            History of impressions and clicks
        metrics : Dict
            Metrics for each ad
        selection_methods : Dict
            Count of each selection method
        final_status : Dict
            Final experiment status
        final_ads_stats : List[Dict]
            Final statistics for all ads
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save history to CSV
        history_df = pd.DataFrame(history)
        history_file = f"{DATA_DIR}/history_{timestamp}.csv"
        history_df.to_csv(history_file, index=False)
        logger.info(f"💾 Saved selection history to {history_file}")

        # Save status history to JSON
        status_file = f"{DATA_DIR}/status_history_{timestamp}.json"
        with open(status_file, "w") as f:
            json.dump(self.status_history, f, indent=2)
        logger.info(f"💾 Saved status history to {status_file}")

        # Save ad statistics history to JSON
        ads_stats_file = f"{DATA_DIR}/ads_stats_history_{timestamp}.json"
        with open(ads_stats_file, "w") as f:
            json.dump(self.ads_stats_history, f, indent=2)
        logger.info(f"💾 Saved ad statistics history to {ads_stats_file}")

        # Save final results as JSON
        results = {
            "timestamp": timestamp,
            "config": self.config,
            "ads": {ad_id: ad_info for ad_id, ad_info in self.ads.items()},
            "metrics": metrics,
            "selection_methods": selection_methods,
            "final_status": final_status,
            "final_ads_stats": final_ads_stats,
            "warmup_completed_at": self.warmup_completed_at,
            "winner_determined_at": self.winner_determined_at,
        }

        results_file = f"{DATA_DIR}/results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Saved final results to {results_file}")

    def visualize_results(
        self, history: List[Dict], output_prefix: Optional[str] = None
    ) -> None:
        """
        Visualize simulation results with advanced plots.

        Parameters
        ----------
        history : List[Dict]
            History of impressions and clicks
        output_prefix : Optional[str], optional
            Prefix for output filenames, by default None
        """
        if not history:
            logger.warning("⚠️ No history data to visualize")
            return

        # Set timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{output_prefix}_{timestamp}" if output_prefix else timestamp

        logger.info("🎨 Generating visualizations...")

        # Convert history to DataFrame
        df = pd.DataFrame(history)

        # Extract winner information
        respect_winner = self.config.get("respect_winner", True)
        winner_determined = False
        winner_iteration = None
        winner_name = None

        # Extract warmup information
        warmup_completed = False
        warmup_iteration = None

        # Check if warmup was completed during the simulation
        if self.warmup_completed_at is not None:
            warmup_completed = True
            warmup_iteration = self.warmup_completed_at

        # Check if a winner was determined during the simulation
        if self.winner_determined_at is not None:
            winner_determined = True
            winner_iteration = self.winner_determined_at
            # Find the winner name from the history
            winner_rows = df[df["iteration"] >= winner_iteration]
            if (
                not winner_rows.empty
                and "winner_info" in winner_rows.iloc[0]
                and winner_rows.iloc[0]["winner_info"]
            ):
                winner_name = winner_rows.iloc[0]["winner_info"].get(
                    "winner_name", "Unknown"
                )

            # If winner name not found in history, try to find it in status history
            if winner_name is None or winner_name == "Unknown":
                for status_entry in self.status_history:
                    if status_entry["status"].get("can_stop", False) and status_entry[
                        "status"
                    ].get("winning_ad"):
                        winner_name = status_entry["status"]["winning_ad"].get(
                            "name", "Unknown"
                        )
                        break

        # Set Seaborn style for better aesthetics
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = [12, 8]
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10

        # Create a color palette for consistent colors across plots
        ad_names = sorted(df["ad_name"].unique())
        num_ads = len(ad_names)
        colors = sns.color_palette("viridis", num_ads)
        color_dict = dict(zip(ad_names, colors))

        # Standardized annotation styles
        warmup_style = {
            "linestyle": "--",
            "linewidth": 2,
            "color": "green",
            "zorder": 5,  # Ensure line is drawn on top
        }

        winner_style = {
            "linestyle": "--",
            "linewidth": 2,
            "color": "red",
            "zorder": 5,  # Ensure line is drawn on top
        }

        warmup_box_style = {
            "boxstyle": "round,pad=0.5",
            "fc": "lightgreen",
            "alpha": 0.7,
            "ec": "green",
        }

        winner_box_style = {
            "boxstyle": "round,pad=0.5",
            "fc": "lightyellow",
            "alpha": 0.7,
            "ec": "red",
        }

        # Standardized annotation text
        def get_warmup_text():
            return f"Warmup phase completed (iteration {warmup_iteration})"

        def get_winner_text():
            base_text = (
                f"Winner determined: {winner_name} (iteration {winner_iteration})"
            )
            if respect_winner:
                return f"{base_text}\nOnly winner shown after this point"
            return base_text

        # 1. MAIN VISUALIZATION: Multiple subplots in one figure
        # ------------------------------------------------------
        # Create a figure with subplots
        fig, axs = plt.subplots(
            4, 1, figsize=(16, 24), gridspec_kw={"height_ratios": [1, 1, 1, 1.5]}
        )
        fig.suptitle(
            "Thompson Sampling Simulation Results",
            fontsize=24,
            fontweight="bold",
            y=0.98,
        )

        # Add subtitle with configuration details and key events
        subtitle = (
            f"Configuration: {self.config.get('total_requests', 'N/A')} requests, "
            f"min_samples={self.config.get('min_samples', 'N/A')}, "
            f"confidence_threshold={self.config.get('confidence_threshold', 'N/A'):.2f}, "
            f"warmup_impressions={self.config.get('warmup_impressions', 'N/A')}, "
            f"respect_winner={self.config.get('respect_winner', True)}"
        )

        # Add key event information to subtitle if applicable
        key_events = []
        if warmup_completed:
            key_events.append(f"Warmup completed at iteration {warmup_iteration}")
        if winner_determined:
            key_events.append(f"Winner determined at iteration {winner_iteration}")

        if key_events:
            subtitle += "\n" + " | ".join(key_events)
        fig.text(0.5, 0.96, subtitle, fontsize=14, ha="center")

        # 1.1 Cumulative Impressions
        for i, ad_name in enumerate(ad_names):
            ad_df = df[df["ad_name"] == ad_name]
            cumulative_impressions = np.arange(1, len(ad_df) + 1)
            axs[0].plot(
                ad_df["iteration"],
                cumulative_impressions,
                color=color_dict[ad_name],
                label=f"{ad_name} ({len(ad_df)} total)",
                linewidth=2.5,
            )

        axs[0].set_title("Cumulative Impressions by Ad", fontsize=18, pad=15)
        axs[0].set_ylabel("Number of Impressions", fontsize=14)
        axs[0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=12)
        axs[0].grid(True, alpha=0.3)

        # Add standardized vertical line and annotation for warmup completion
        if warmup_completed:
            axs[0].axvline(x=warmup_iteration, **warmup_style)
            axs[0].annotate(
                get_warmup_text(),
                xy=(warmup_iteration, axs[0].get_ylim()[1] * 0.85),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=warmup_box_style,
                fontsize=10,
            )

        # Add standardized vertical line and annotation for winner determination
        if winner_determined:
            axs[0].axvline(x=winner_iteration, **winner_style)
            axs[0].annotate(
                get_winner_text(),
                xy=(winner_iteration, axs[0].get_ylim()[1] * 0.95),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=winner_box_style,
                fontsize=10,
            )

        # Add annotation to show total impressions per ad
        total_iterations = df["iteration"].max()
        for i, ad_name in enumerate(ad_names):
            ad_df = df[df["ad_name"] == ad_name]
            total_impressions = len(ad_df)
            axs[0].annotate(
                f"{total_impressions} impressions",
                xy=(total_iterations, total_impressions),
                xytext=(10, 0),
                textcoords="offset points",
                color=color_dict[ad_name],
                fontweight="bold",
                va="center",
            )

        # 1.2 Rolling CTR with confidence intervals
        window_size = min(100, df["iteration"].max() // 10)

        for ad_name in ad_names:
            ad_df = df[df["ad_name"] == ad_name]
            if len(ad_df) > window_size:
                # Calculate rolling CTR
                rolling_ctr = ad_df["is_clicked"].rolling(window=window_size).mean()

                # Get true CTR for reference
                true_ctr = self.ads[ad_df["ad_id"].iloc[0]]["ctr"]

                # Calculate 95% confidence interval
                # Using Wilson score interval approximation
                z = 1.96  # 95% confidence
                n = window_size

                def wilson_interval(p, n, z):
                    """Calculate Wilson score interval for a proportion."""
                    denominator = 1 + z**2 / n
                    center = (p + z**2 / (2 * n)) / denominator
                    spread = (
                        z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
                    )
                    return center - spread, center + spread

                lower_bound = []
                upper_bound = []

                for i in range(len(rolling_ctr)):
                    if np.isnan(rolling_ctr.iloc[i]):
                        lower_bound.append(np.nan)
                        upper_bound.append(np.nan)
                    else:
                        l, u = wilson_interval(rolling_ctr.iloc[i], n, z)
                        lower_bound.append(l)
                        upper_bound.append(u)

                # Plot the rolling CTR with confidence interval
                x = ad_df["iteration"]
                axs[1].plot(
                    x,
                    rolling_ctr,
                    label=f"{ad_name} (True CTR: {true_ctr:.2%})",
                    color=color_dict[ad_name],
                    linewidth=2.5,
                )
                axs[1].fill_between(
                    x, lower_bound, upper_bound, color=color_dict[ad_name], alpha=0.2
                )

        # Add horizontal lines for true CTR values
        for ad_name in ad_names:
            ad_id = df[df["ad_name"] == ad_name]["ad_id"].iloc[0]
            true_ctr = self.ads[ad_id]["ctr"]
            axs[1].axhline(
                y=true_ctr, color=color_dict[ad_name], linestyle="--", alpha=0.7
            )

        # Add standardized vertical line and annotation for warmup completion
        if warmup_completed:
            axs[1].axvline(x=warmup_iteration, **warmup_style)
            axs[1].annotate(
                get_warmup_text(),
                xy=(warmup_iteration, axs[1].get_ylim()[1] * 0.85),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=warmup_box_style,
                fontsize=10,
            )

        # Add standardized vertical line and annotation for winner determination
        if winner_determined:
            axs[1].axvline(x=winner_iteration, **winner_style)
            axs[1].annotate(
                get_winner_text(),
                xy=(winner_iteration, axs[1].get_ylim()[1] * 0.95),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=winner_box_style,
                fontsize=10,
            )

        axs[1].set_title(
            f"Rolling Average CTR with 95% Confidence Intervals (Window: {window_size})",
            fontsize=18,
            pad=15,
        )
        axs[1].set_ylabel("Click-Through Rate", fontsize=14)
        axs[1].legend(loc="best", frameon=True, framealpha=0.9, fontsize=12)
        axs[1].set_ylim(bottom=0)
        axs[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axs[1].grid(True, alpha=0.3)

        # 1.3 CTR Convergence Plot (cumulative CTR over time)
        for ad_name in ad_names:
            ad_df = df[df["ad_name"] == ad_name]
            if len(ad_df) > 0:
                # Calculate cumulative CTR
                ad_df = ad_df.sort_values("iteration")
                ad_df["cumulative_clicks"] = ad_df["is_clicked"].cumsum()
                ad_df["impression_number"] = range(1, len(ad_df) + 1)
                ad_df["cumulative_ctr"] = (
                    ad_df["cumulative_clicks"] / ad_df["impression_number"]
                )

                # Plot cumulative CTR
                axs[2].plot(
                    ad_df["iteration"],
                    ad_df["cumulative_ctr"],
                    label=f"{ad_name}",
                    color=color_dict[ad_name],
                    linewidth=2.5,
                )

                # Add the final measured CTR
                final_ctr = ad_df["cumulative_ctr"].iloc[-1]
                axs[2].text(
                    ad_df["iteration"].max(),
                    final_ctr,
                    f"  {final_ctr:.3%}",
                    color=color_dict[ad_name],
                    fontweight="bold",
                    va="center",
                    fontsize=12,
                )

        # Add horizontal lines for true CTR values
        for ad_name in ad_names:
            ad_id = df[df["ad_name"] == ad_name]["ad_id"].iloc[0]
            true_ctr = self.ads[ad_id]["ctr"]
            axs[2].axhline(
                y=true_ctr, color=color_dict[ad_name], linestyle="--", alpha=0.7
            )

            # Add labels for true CTR values
            axs[2].text(
                0,
                true_ctr,
                f"True: {true_ctr:.1%} ",
                color=color_dict[ad_name],
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

        # Add standardized vertical line and annotation for warmup completion
        if warmup_completed:
            axs[2].axvline(x=warmup_iteration, **warmup_style)
            axs[2].annotate(
                get_warmup_text(),
                xy=(warmup_iteration, axs[2].get_ylim()[1] * 0.85),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=warmup_box_style,
                fontsize=10,
            )

        # Add standardized vertical line and annotation for winner determination
        if winner_determined:
            axs[2].axvline(x=winner_iteration, **winner_style)
            axs[2].annotate(
                get_winner_text(),
                xy=(winner_iteration, axs[2].get_ylim()[1] * 0.95),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                bbox=winner_box_style,
                fontsize=10,
            )

        axs[2].set_title("Cumulative CTR Convergence Over Time", fontsize=18, pad=15)
        axs[2].set_ylabel("Cumulative CTR", fontsize=14)
        axs[2].set_xlim(left=0)
        axs[2].set_ylim(bottom=0)
        axs[2].yaxis.set_major_formatter(PercentFormatter(1.0))
        axs[2].legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=12)
        axs[2].grid(True, alpha=0.3)

        # 1.4 Ad Selection Distribution Over Time (heatmap)
        # Divide the timeline into intervals
        intervals = min(20, df["iteration"].max() // 100)
        if intervals < 2:
            intervals = 2  # Ensure at least 2 intervals

        max_iteration = df["iteration"].max()
        interval_size = max_iteration // intervals

        # Create a matrix for the heatmap
        heatmap_data = np.zeros((num_ads, intervals))
        interval_labels = []

        # Track which intervals are after winner determination
        intervals_after_winner = []

        for i in range(intervals):
            start = i * interval_size + 1
            end = (i + 1) * interval_size if i < intervals - 1 else max_iteration
            interval_labels.append(f"{start}-{end}")

            # Mark intervals that are entirely after winner determination
            if winner_determined and start > winner_iteration:
                intervals_after_winner.append(i)

            interval_df = df[(df["iteration"] >= start) & (df["iteration"] <= end)]
            total_impressions = len(interval_df)

            for j, ad_name in enumerate(ad_names):
                ad_impressions = len(interval_df[interval_df["ad_name"] == ad_name])
                heatmap_data[j, i] = (
                    ad_impressions / total_impressions if total_impressions > 0 else 0
                )

        # Plot heatmap
        im = axs[3].imshow(heatmap_data, cmap="viridis", aspect="auto")

        # Add colorbar
        cbar = fig.colorbar(im, ax=axs[3])
        cbar.set_label(
            "Proportion of Impressions", rotation=270, labelpad=20, fontsize=14
        )

        # Add labels and ticks
        axs[3].set_yticks(np.arange(num_ads))
        axs[3].set_yticklabels(ad_names)
        axs[3].set_xticks(np.arange(intervals))
        axs[3].set_xticklabels(interval_labels, rotation=45, ha="right")

        # Annotate heatmap cells with percentages
        for i in range(num_ads):
            for j in range(intervals):
                text_color = "white" if heatmap_data[i, j] > 0.5 else "black"
                axs[3].text(
                    j,
                    i,
                    f"{heatmap_data[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=11,
                )

        # Find intervals for warmup completion and winner determination
        interval_warmup_idx = None
        interval_winner_idx = None

        if warmup_completed:
            for i in range(intervals):
                start = i * interval_size + 1
                end = (i + 1) * interval_size if i < intervals - 1 else max_iteration
                if start <= warmup_iteration <= end:
                    interval_warmup_idx = i
                    break

        if winner_determined:
            for i in range(intervals):
                start = i * interval_size + 1
                end = (i + 1) * interval_size if i < intervals - 1 else max_iteration
                if start <= winner_iteration <= end:
                    interval_winner_idx = i
                    break

        # Add standardized vertical lines for phase transitions in heatmap
        if interval_warmup_idx is not None:
            # Draw a vertical line before the interval containing warmup completion
            axs[3].axvline(x=interval_warmup_idx - 0.5, **warmup_style)
            # Add annotation
            axs[3].text(
                interval_warmup_idx - 0.5,
                -0.5,
                get_warmup_text(),
                ha="center",
                va="top",
                bbox=warmup_box_style,
                fontsize=10,
                rotation=90,
            )

        if interval_winner_idx is not None:
            # Draw a vertical line after the interval containing winner determination
            axs[3].axvline(x=interval_winner_idx + 0.5, **winner_style)
            # Add annotation
            axs[3].text(
                interval_winner_idx + 0.5,
                -0.5,
                get_winner_text(),
                ha="center",
                va="top",
                bbox=winner_box_style,
                fontsize=10,
                rotation=90,
            )

        axs[3].set_title("Ad Selection Distribution Over Time", fontsize=18, pad=15)
        axs[3].set_xlabel("Iteration Intervals", fontsize=14)
        axs[3].set_ylabel("Advertisement", fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.05, hspace=0.25)

        # Save the main visualization
        main_viz_file = f"{CHARTS_DIR}/main_visualization_{prefix}.png"
        plt.savefig(main_viz_file, dpi=300, bbox_inches="tight")
        logger.info(f"📊 Main visualization saved to {main_viz_file}")

        plt.close(fig)  # Close to free memory

        # 2. SELECTION METHOD VISUALIZATION
        # ---------------------------------
        if "selection_method" in df.columns:
            plt.figure(figsize=(14, 10))

            # Count selection methods over time
            selection_methods = sorted(df["selection_method"].unique())

            # Divide into intervals
            interval_data = []
            for i in range(intervals):
                start = i * interval_size + 1
                end = (i + 1) * interval_size if i < intervals - 1 else max_iteration
                interval_df = df[(df["iteration"] >= start) & (df["iteration"] <= end)]

                interval_counts = {}
                for method in selection_methods:
                    count = len(interval_df[interval_df["selection_method"] == method])
                    interval_counts[method] = (
                        count / len(interval_df) if len(interval_df) > 0 else 0
                    )

                interval_data.append({"interval": f"{start}-{end}", **interval_counts})

            # Convert to DataFrame for easy plotting
            method_df = pd.DataFrame(interval_data)
            method_df = method_df.set_index("interval")

            # Create stacked bar chart
            ax = method_df.plot(
                kind="bar", stacked=True, figsize=(14, 8), colormap="viridis", width=0.8
            )

            plt.title("Selection Method Distribution Over Time", fontsize=18, pad=15)
            plt.xlabel("Iteration Intervals", fontsize=14)
            plt.ylabel("Proportion of Selections", fontsize=14)
            plt.legend(
                title="Selection Method", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

            # Add value labels to each section
            for c in ax.containers:
                labels = [
                    f"{v.get_height():.1%}" if v.get_height() > 0.03 else "" for v in c
                ]
                ax.bar_label(c, labels=labels, label_type="center")

            # Find intervals for standardized markers
            for i, interval_label in enumerate(method_df.index):
                start_str, end_str = interval_label.split("-")
                start, end = int(start_str), int(end_str)

                # Standardized warmup completion line and annotation
                if warmup_completed and start <= warmup_iteration <= end:
                    plt.axvline(x=i, **warmup_style)
                    plt.annotate(
                        get_warmup_text(),
                        xy=(i, 1.03),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        bbox=warmup_box_style,
                        fontsize=10,
                        transform=ax.get_xaxis_transform(),
                    )

                # Standardized winner determination line and annotation
                if winner_determined and start <= winner_iteration <= end:
                    plt.axvline(x=i, **winner_style)
                    plt.annotate(
                        get_winner_text(),
                        xy=(i, 1.08),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        bbox=winner_box_style,
                        fontsize=10,
                        transform=ax.get_xaxis_transform(),
                    )

            # Save the selection method visualization
            selection_file = f"{CHARTS_DIR}/selection_methods_{prefix}.png"
            plt.savefig(selection_file, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Selection method visualization saved to {selection_file}")

            plt.close()  # Close to free memory

        # 3. SELECTION RATE COMPARISON
        # ---------------------------
        # Create bar chart comparing true CTR vs measured CTR
        plt.figure(figsize=(12, 8))

        # Prepare data
        ad_ids = list(self.ads.keys())
        ad_names = [self.ads[ad_id]["name"] for ad_id in ad_ids]
        true_ctrs = [self.ads[ad_id]["ctr"] for ad_id in ad_ids]
        measured_ctrs = [self.ads[ad_id]["actual_ctr"] for ad_id in ad_ids]

        x = np.arange(len(ad_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(
            x - width / 2, true_ctrs, width, label="True CTR", color="steelblue"
        )
        rects2 = ax.bar(
            x + width / 2, measured_ctrs, width, label="Measured CTR", color="coral"
        )

        # Add labels and title
        ax.set_title("True vs Measured CTR Comparison", fontsize=18, pad=15)
        ax.set_xlabel("Advertisement", fontsize=14)
        ax.set_ylabel("Click-Through Rate", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(ad_names)
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        # Add value labels to bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.2%}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

        autolabel(rects1)
        autolabel(rects2)

        # Add winner annotation if applicable with standardized styling
        if winner_determined:
            winner_index = None
            for i, name in enumerate(ad_names):
                if name == winner_name:
                    winner_index = i
                    break

            if winner_index is not None:
                # Highlight the winner bar
                highlight = plt.Rectangle(
                    (x[winner_index] - width, 0),
                    width * 2,
                    max(true_ctrs[winner_index], measured_ctrs[winner_index]) * 1.1,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=2,
                    zorder=5,
                )
                ax.add_patch(highlight)

                # Add winner text with standardized formatting
                ax.text(
                    x[winner_index],
                    max(true_ctrs[winner_index], measured_ctrs[winner_index]) * 1.15,
                    f"WINNER (iteration {winner_iteration})",
                    ha="center",
                    fontsize=14,
                    weight="bold",
                    bbox=winner_box_style,
                )

        plt.tight_layout()

        # Save the CTR comparison visualization
        ctr_file = f"{CHARTS_DIR}/ctr_comparison_{prefix}.png"
        plt.savefig(ctr_file, dpi=300, bbox_inches="tight")
        logger.info(f"📊 CTR comparison visualization saved to {ctr_file}")

        plt.close()  # Close to free memory

        # 4. EXPERIMENT STATUS EVOLUTION
        # -----------------------------
        if self.status_history:
            # Extract status data
            status_iterations = [s["iteration"] for s in self.status_history]

            # Extract confidence levels
            confidence_levels = [
                s["status"].get("confidence", 0) for s in self.status_history
            ]

            # Extract warmup_complete status
            warmup_status = [
                int(s["status"].get("warmup_complete", False))
                for s in self.status_history
            ]

            # Extract can_stop status
            can_stop_status = [
                int(s["status"].get("can_stop", False)) for s in self.status_history
            ]

            # Extract and track winning ad information
            winning_ads = []
            for s in self.status_history:
                if s["status"].get("winning_ad") is not None:
                    winning_ads.append(s["status"]["winning_ad"].get("name", "Unknown"))
                else:
                    winning_ads.append(None)

            # Create figure with multiple y-axes
            fig, ax1 = plt.subplots(figsize=(14, 8))

            # Plot confidence level
            color = "tab:blue"
            ax1.set_xlabel("Iteration", fontsize=14)
            ax1.set_ylabel("Confidence Level", color=color, fontsize=14)
            ax1.plot(
                status_iterations,
                confidence_levels,
                color=color,
                marker="o",
                label="Confidence",
            )
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.yaxis.set_major_formatter(PercentFormatter(1.0))

            # Draw threshold line
            threshold = self.config.get("confidence_threshold", 0.95)
            ax1.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                label=f"Threshold ({threshold:.0%})",
            )

            # Create second y-axis for status flags
            ax2 = ax1.twinx()

            # No need for y-label as we'll use the legend
            ax2.set_ylim(-0.1, 1.1)  # Binary values plus some margin
            ax2.set_yticks([])  # Hide y-ticks

            # Plot warmup status
            ax2.plot(
                status_iterations,
                warmup_status,
                color="tab:green",
                marker="s",
                label="Warmup Complete",
                linestyle="-",
            )

            # Plot can_stop status
            ax2.plot(
                status_iterations,
                can_stop_status,
                color="tab:red",
                marker="^",
                label="Can Stop",
                linestyle="-",
            )

            # Add title and legend
            plt.title("Experiment Status Evolution", fontsize=18, pad=15)

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=12)

            # Add standardized annotations for key events
            # Find when warmup completed
            if 1 in warmup_status:
                warmup_idx = warmup_status.index(1)
                warmup_iter = status_iterations[warmup_idx]

                # Add standardized warmup completion annotation
                plt.axvline(x=warmup_iter, **warmup_style)
                plt.annotate(
                    get_warmup_text(),
                    xy=(warmup_iter, 0.5),
                    xytext=(warmup_iter + 50, 0.3),
                    arrowprops=dict(facecolor="green", shrink=0.05, width=1.5),
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=warmup_box_style,
                )

            # Find when experiment could be stopped and annotate with winner name
            if 1 in can_stop_status:
                stop_idx = can_stop_status.index(1)
                stop_iter = status_iterations[stop_idx]
                conf_at_stop = confidence_levels[stop_idx]
                winner_name_from_status = (
                    winning_ads[stop_idx]
                    if stop_idx < len(winning_ads) and winning_ads[stop_idx]
                    else "Unknown"
                )

                # Add standardized winner determination annotation
                plt.axvline(x=stop_iter, **winner_style)
                winner_annotation = (
                    f"Winner: {winner_name_from_status} ({conf_at_stop:.1%})"
                )
                if respect_winner:
                    winner_annotation += "\nOnly this ad shown from now on"

                plt.annotate(
                    winner_annotation,
                    xy=(stop_iter, conf_at_stop),
                    xytext=(stop_iter + 50, conf_at_stop + 0.15),
                    arrowprops=dict(facecolor="red", shrink=0.05, width=1.5),
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=winner_box_style,
                )

            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Save the status evolution visualization
            status_file = f"{CHARTS_DIR}/status_evolution_{prefix}.png"
            plt.savefig(status_file, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Status evolution visualization saved to {status_file}")

            plt.close()  # Close to free memory

        # 5. PROBABILITY BEST EVOLUTION
        # ----------------------------
        if self.ads_stats_history:
            # Create a figure for the probability evolution
            plt.figure(figsize=(14, 8))

            # Process and extract data
            ad_probability_data = {}

            # Identify all unique ad IDs and names first
            ad_info = {}
            for stats_point in self.ads_stats_history:
                for ad in stats_point["ads_stats"]:
                    ad_id = ad.get("id")
                    ad_name = ad.get("name")
                    if ad_id and ad_name:
                        ad_info[ad_id] = ad_name

            # Initialize data structure for each ad
            for ad_id, ad_name in ad_info.items():
                ad_probability_data[ad_id] = {
                    "name": ad_name,
                    "iterations": [],
                    "probability_best": [],
                }

            # Collect probability data points over time
            for stats_point in self.ads_stats_history:
                iteration = stats_point["iteration"]
                for ad in stats_point["ads_stats"]:
                    ad_id = ad.get("id")
                    if ad_id in ad_probability_data:
                        probability = ad.get("probability_best", 0)
                        ad_probability_data[ad_id]["iterations"].append(iteration)
                        ad_probability_data[ad_id]["probability_best"].append(
                            probability
                        )

            # Plot probability evolution for each ad
            for ad_id, data in ad_probability_data.items():
                if data["iterations"]:  # Only plot if we have data
                    ad_name = data["name"]
                    plt.plot(
                        data["iterations"],
                        data["probability_best"],
                        marker="o",
                        linewidth=2.5,
                        label=f"{ad_name}",
                        color=color_dict.get(ad_name, None),
                    )

            # Add confidence threshold line
            threshold = self.config.get("confidence_threshold", 0.95)
            plt.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                label=f"Confidence Threshold ({threshold:.0%})",
            )

            # Add standardized vertical line for warmup completion
            if warmup_completed:
                plt.axvline(x=warmup_iteration, **warmup_style)
                plt.annotate(
                    get_warmup_text(),
                    xy=(warmup_iteration, 0.3),
                    xytext=(-10, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    bbox=warmup_box_style,
                    fontsize=10,
                )

            # Add standardized vertical line for winner determination
            if winner_determined:
                plt.axvline(x=winner_iteration, **winner_style)
                plt.annotate(
                    get_winner_text(),
                    xy=(winner_iteration, 0.5),
                    xytext=(-10, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    bbox=winner_box_style,
                    fontsize=10,
                )

            # Add labels, title, and formatting
            plt.title("Probability of Being Best Ad Over Time", fontsize=18, pad=15)
            plt.xlabel("Iterations", fontsize=14)
            plt.ylabel("Probability of Being Best", fontsize=14)
            plt.legend(loc="best", fontsize=12)
            plt.grid(True, alpha=0.3)

            # Format y-axis as percentage
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

            # Set y-axis limits
            plt.ylim(0, 1.05)

            plt.tight_layout()

            # Find iteration where probability crosses threshold
            crossing_annotations = []
            for ad_id, data in ad_probability_data.items():
                if data["iterations"] and data["probability_best"]:
                    # Check for threshold crossing
                    for i in range(1, len(data["probability_best"])):
                        if (
                            data["probability_best"][i - 1] < threshold
                            and data["probability_best"][i] >= threshold
                        ):
                            crossing_annotations.append(
                                {
                                    "name": data["name"],
                                    "iteration": data["iterations"][i],
                                    "probability": data["probability_best"][i],
                                }
                            )

            # Add annotations for threshold crossing
            for annotation in crossing_annotations:
                plt.annotate(
                    f'{annotation["name"]}\nreaches threshold',
                    xy=(annotation["iteration"], annotation["probability"]),
                    xytext=(
                        annotation["iteration"] + 50,
                        annotation["probability"] - 0.1,
                    ),
                    arrowprops=dict(facecolor="black", shrink=0.05, width=1),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                )

            # Save the probability evolution visualization
            prob_file = f"{CHARTS_DIR}/probability_evolution_{prefix}.png"
            plt.savefig(prob_file, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Probability evolution visualization saved to {prob_file}")

            plt.close()  # Close to free memory

        # 6. EXPERIMENT PHASES ANALYSIS
        # ------------------------------------
        # Create a comprehensive visualization analyzing all phases of the experiment
        plt.figure(figsize=(16, 12))

        # Determine phases based on warmup and winner determination
        warmup_df = None
        testing_df = None
        post_winner_df = None

        if warmup_completed and winner_determined:
            # All three phases occurred
            warmup_df = df[df["iteration"] < warmup_iteration]
            testing_df = df[
                (df["iteration"] >= warmup_iteration)
                & (df["iteration"] < winner_iteration)
            ]
            post_winner_df = df[df["iteration"] >= winner_iteration]
        elif warmup_completed:
            # Only warmup and testing phases
            warmup_df = df[df["iteration"] < warmup_iteration]
            testing_df = df[df["iteration"] >= warmup_iteration]
        elif winner_determined:
            # No clear warmup phase, split into before/after winner
            testing_df = df[df["iteration"] < winner_iteration]
            post_winner_df = df[df["iteration"] >= winner_iteration]
        else:
            # Only testing phase
            testing_df = df.copy()

        # Set up subplot grid - adjust based on how many phases we have
        n_phases = sum(x is not None for x in [warmup_df, testing_df, post_winner_df])
        subplot_rows = min(n_phases, 3)  # At most 3 rows

        # Set up the subplot grid
        current_subplot = 1

        # Warmup phase analysis
        if warmup_df is not None and not warmup_df.empty:
            plt.subplot(subplot_rows, 2, current_subplot)
            warmup_impressions = warmup_df["ad_name"].value_counts()
            plt.pie(
                warmup_impressions,
                labels=warmup_impressions.index,
                autopct="%1.1f%%",
                colors=[
                    color_dict.get(name, "gray") for name in warmup_impressions.index
                ],
                startangle=90,
            )
            plt.title(f"WARMUP PHASE (iterations 1-{warmup_iteration})", fontsize=14)

            # Add summary statistics
            warmup_summary = ""
            for ad_name in ad_names:
                ad_df = warmup_df[warmup_df["ad_name"] == ad_name]
                impressions = len(ad_df)
                clicks = ad_df["is_clicked"].sum()
                ctr = clicks / impressions if impressions > 0 else 0
                warmup_summary += (
                    f"{ad_name}: {impressions} imp, {clicks} clicks, {ctr:.2%} CTR\n"
                )

            plt.annotate(
                warmup_summary,
                xy=(0.5, -0.1),
                xycoords="axes fraction",
                ha="center",
                va="center",
                bbox=warmup_box_style,
                fontsize=9,
            )

            current_subplot += 1

        # Testing phase analysis (after warmup, before winner)
        if testing_df is not None and not testing_df.empty:
            plt.subplot(subplot_rows, 2, current_subplot)
            testing_impressions = testing_df["ad_name"].value_counts()
            plt.pie(
                testing_impressions,
                labels=testing_impressions.index,
                autopct="%1.1f%%",
                colors=[
                    color_dict.get(name, "gray") for name in testing_impressions.index
                ],
                startangle=90,
            )

            # Adjust title based on phases - use standardized iteration labels
            if warmup_completed and winner_determined:
                plt.title(
                    f"TESTING PHASE (iterations {warmup_iteration}-{winner_iteration})",
                    fontsize=14,
                )
            elif warmup_completed:
                plt.title(
                    f"TESTING PHASE (iterations {warmup_iteration}+)", fontsize=14
                )
            elif winner_determined:
                plt.title(
                    f"BEFORE WINNER (iterations 1-{winner_iteration})", fontsize=14
                )
            else:
                plt.title("TESTING PHASE (all iterations)", fontsize=14)

            # Add summary statistics
            testing_summary = ""
            for ad_name in ad_names:
                ad_df = testing_df[testing_df["ad_name"] == ad_name]
                impressions = len(ad_df)
                clicks = ad_df["is_clicked"].sum()
                ctr = clicks / impressions if impressions > 0 else 0
                testing_summary += (
                    f"{ad_name}: {impressions} imp, {clicks} clicks, {ctr:.2%} CTR\n"
                )

            plt.annotate(
                testing_summary,
                xy=(0.5, -0.1),
                xycoords="axes fraction",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3),
                fontsize=9,
            )

            current_subplot += 1

        # Post-winner phase analysis
        if post_winner_df is not None and not post_winner_df.empty:
            plt.subplot(subplot_rows, 2, current_subplot)
            post_winner_impressions = post_winner_df["ad_name"].value_counts()

            if not post_winner_impressions.empty:
                plt.pie(
                    post_winner_impressions,
                    labels=post_winner_impressions.index,
                    autopct="%1.1f%%",
                    colors=[
                        color_dict.get(name, "gray")
                        for name in post_winner_impressions.index
                    ],
                    startangle=90,
                )
                # Use standardized iteration label
                plt.title(f"AFTER WINNER (iterations {winner_iteration}+)", fontsize=14)

                # Add text if only winner is shown
                if (
                    len(post_winner_impressions) == 1
                    and post_winner_impressions.index[0] == winner_name
                ):
                    plt.annotate(
                        "Only winner ad shown\nas configured by respect_winner=True",
                        xy=(0.5, 0.5),
                        xycoords="axes fraction",
                        ha="center",
                        va="center",
                        bbox=winner_box_style,
                        fontsize=10,
                    )

                # Add summary statistics
                post_winner_summary = ""
                for ad_name in post_winner_impressions.index:
                    ad_df = post_winner_df[post_winner_df["ad_name"] == ad_name]
                    impressions = len(ad_df)
                    clicks = ad_df["is_clicked"].sum()
                    ctr = clicks / impressions if impressions > 0 else 0
                    post_winner_summary += f"{ad_name}: {impressions} imp, {clicks} clicks, {ctr:.2%} CTR\n"

                plt.annotate(
                    post_winner_summary,
                    xy=(0.5, -0.1),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    bbox=winner_box_style,
                    fontsize=9,
                )
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No impressions after winner determined",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=14,
                )

            current_subplot += 1

        # Experiment summary and key events information
        plt.subplot(subplot_rows, 1, subplot_rows)
        plt.axis("off")  # No axes for text display

        # Prepare experiment phases information - use standardized phase naming
        phases_text = "EXPERIMENT PHASES:\n"

        if warmup_completed:
            phases_text += f"• WARMUP PHASE: Iterations 1-{warmup_iteration}\n"
            phases_text += f"  Each ad received approximately {self.config.get('warmup_impressions', 'N/A')} initial impressions\n\n"

        if warmup_completed and winner_determined:
            phases_text += f"• TESTING PHASE: Iterations {warmup_iteration + 1}-{winner_iteration}\n"
            phases_text += f"  Thompson Sampling actively tested all ads to find the best performer\n\n"
        elif warmup_completed:
            phases_text += f"• TESTING PHASE: Iterations {warmup_iteration + 1}+\n"
            phases_text += f"  Thompson Sampling actively testing all ads (no winner determined yet)\n\n"
        elif winner_determined:
            phases_text += f"• TESTING PHASE: Iterations 1-{winner_iteration}\n"
            phases_text += (
                f"  Thompson Sampling tested all ads until a winner was found\n\n"
            )

        if winner_determined:
            phases_text += f"• POST-WINNER PHASE: Iterations {winner_iteration + 1}+\n"
            if respect_winner:
                phases_text += f"  Only the winning ad ({winner_name}) was shown as configured by respect_winner=True\n\n"
            else:
                phases_text += f"  Thompson Sampling continued despite finding a winner, as configured by respect_winner=False\n\n"

        # Add winner information if a winner was determined
        if winner_determined:
            winner_id = None
            winner_stats = None
            for ad_id, ad_info in self.ads.items():
                if ad_info["name"] == winner_name:
                    winner_id = ad_id
                    winner_stats = ad_info
                    break

            if winner_stats:
                true_ctr = winner_stats.get("ctr", 0)
                measured_ctr = winner_stats.get("actual_ctr", 0)
                impressions = winner_stats.get("impressions", 0)
                clicks = winner_stats.get("clicks", 0)

                # Calculate metrics by phase for the winner
                winner_phases = {}

                if warmup_df is not None and not warmup_df.empty:
                    warmup_data = warmup_df[warmup_df["ad_name"] == winner_name]
                    warmup_impressions = len(warmup_data)
                    warmup_clicks = warmup_data["is_clicked"].sum()
                    warmup_ctr = (
                        warmup_clicks / warmup_impressions
                        if warmup_impressions > 0
                        else 0
                    )
                    winner_phases["warmup"] = {
                        "impressions": warmup_impressions,
                        "clicks": warmup_clicks,
                        "ctr": warmup_ctr,
                    }

                if testing_df is not None and not testing_df.empty:
                    testing_data = testing_df[testing_df["ad_name"] == winner_name]
                    testing_impressions = len(testing_data)
                    testing_clicks = testing_data["is_clicked"].sum()
                    testing_ctr = (
                        testing_clicks / testing_impressions
                        if testing_impressions > 0
                        else 0
                    )
                    winner_phases["testing"] = {
                        "impressions": testing_impressions,
                        "clicks": testing_clicks,
                        "ctr": testing_ctr,
                    }

                if post_winner_df is not None and not post_winner_df.empty:
                    post_data = post_winner_df[post_winner_df["ad_name"] == winner_name]
                    post_impressions = len(post_data)
                    post_clicks = post_data["is_clicked"].sum()
                    post_ctr = (
                        post_clicks / post_impressions if post_impressions > 0 else 0
                    )
                    winner_phases["post_winner"] = {
                        "impressions": post_impressions,
                        "clicks": post_clicks,
                        "ctr": post_ctr,
                    }

                # Selection methods for winner
                selection_methods_text = ""
                if "selection_counts" in winner_stats:
                    for method, count in winner_stats["selection_counts"].items():
                        percentage = (
                            count / sum(winner_stats["selection_counts"].values()) * 100
                        )
                        selection_methods_text += (
                            f"• {method}: {count} selections ({percentage:.1f}%)\n"
                        )

                # Calculate probability at winner determination
                probability_at_winner = None
                for status in self.status_history:
                    if status["iteration"] >= winner_iteration:
                        if "status" in status and "winning_ad" in status["status"]:
                            probability_at_winner = status["status"]["winning_ad"].get(
                                "probability_best", None
                            )
                            break

                winner_text = f"WINNER ANALYSIS: {winner_name}\n\n"
                winner_text += f"Winner determined at iteration: {winner_iteration}\n"
                if probability_at_winner:
                    winner_text += f"Probability of being best at determination: {probability_at_winner:.2%}\n\n"
                else:
                    winner_text += "\n"

                winner_text += "Overall Statistics:\n"
                winner_text += f"• True CTR: {true_ctr:.2%}\n"
                winner_text += f"• Measured CTR: {measured_ctr:.2%}\n"
                winner_text += f"• Total Impressions: {impressions}\n"
                winner_text += f"• Total Clicks: {clicks}\n\n"

                if "warmup" in winner_phases:
                    winner_text += "In Warmup Phase:\n"
                    winner_text += (
                        f"• Impressions: {winner_phases['warmup']['impressions']}\n"
                    )
                    winner_text += f"• Clicks: {winner_phases['warmup']['clicks']}\n"
                    winner_text += f"• CTR: {winner_phases['warmup']['ctr']:.2%}\n\n"

                if "testing" in winner_phases:
                    if warmup_completed and winner_determined:
                        phase_name = "Testing Phase (after warmup, before winner):"
                    elif warmup_completed:
                        phase_name = "Testing Phase (after warmup):"
                    else:
                        phase_name = "Before Winner Determination:"

                    winner_text += f"In {phase_name}\n"
                    winner_text += (
                        f"• Impressions: {winner_phases['testing']['impressions']}\n"
                    )
                    winner_text += f"• Clicks: {winner_phases['testing']['clicks']}\n"
                    winner_text += f"• CTR: {winner_phases['testing']['ctr']:.2%}\n\n"

                if "post_winner" in winner_phases:
                    winner_text += "After Winner Determination:\n"
                    winner_text += f"• Impressions: {winner_phases['post_winner']['impressions']}\n"
                    winner_text += (
                        f"• Clicks: {winner_phases['post_winner']['clicks']}\n"
                    )
                    winner_text += (
                        f"• CTR: {winner_phases['post_winner']['ctr']:.2%}\n\n"
                    )

                winner_text += f"Selection Methods Used:\n{selection_methods_text}"

                # Display phases info and winner information side by side
                plt.text(
                    0.25,
                    0.5,
                    phases_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=1", fc="lightblue", alpha=0.3),
                    transform=plt.gca().transAxes,
                )

                plt.text(
                    0.75,
                    0.5,
                    winner_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=winner_box_style,
                    transform=plt.gca().transAxes,
                )
            else:
                # If winner stats not available, just show phases info
                plt.text(
                    0.5,
                    0.5,
                    phases_text,
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=1", fc="lightblue", alpha=0.3),
                    transform=plt.gca().transAxes,
                )
        else:
            # No winner determined yet, just show phases info
            plt.text(
                0.5,
                0.5,
                phases_text,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=1", fc="lightblue", alpha=0.3),
                transform=plt.gca().transAxes,
            )

        # Create standardized title with key experiment information
        title = "Experiment Phases Analysis"
        if warmup_completed:
            title += f" | Warmup completed at iteration {warmup_iteration}"
        if winner_determined:
            title += f" | Winner determined at iteration {winner_iteration}"

        plt.suptitle(title, fontsize=18, fontweight="bold")

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.4)

        # Save the experiment phases analysis visualization
        phases_analysis_file = f"{CHARTS_DIR}/experiment_phases_{prefix}.png"
        plt.savefig(phases_analysis_file, dpi=300, bbox_inches="tight")
        logger.info(f"📊 Experiment phases analysis saved to {phases_analysis_file}")

        plt.close()  # Close to free memory

    logger.info(f"🎨 All visualizations saved to {CHARTS_DIR}")

    def run_simulation(self) -> None:
        """
        Run complete simulation based on configuration.

        This method orchestrates the entire simulation workflow including
        service reset, experiment configuration, ad creation, traffic simulation,
        and results visualization.
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING THOMPSON SAMPLING SIMULATION")
        logger.info("=" * 80)

        # Reset service
        if self.config.get("reset_service", True):
            logger.info("🧹 Resetting service...")
            self.reset_service()

        # Configure experiment
        logger.info("⚙️ Configuring experiment...")
        self.configure_experiment(
            self.config.get("min_samples", 1000),
            self.config.get("confidence_threshold", 0.95),
            self.config.get("warmup_impressions", 500),
        )

        # Create ads
        logger.info("📝 Creating ads...")
        self.create_ads(self.config["ads"])

        # Simulate traffic
        logger.info("🔄 Starting traffic simulation...")
        ads_stats, history = self.simulate_traffic(
            self.config.get("total_requests", 10000),
            self.config.get("respect_winner", True),
            self.config.get("check_interval", 100),
        )

        # Visualize results
        if self.config.get("visualize", True):
            logger.info("🎨 Generating visualizations...")
            self.visualize_results(
                history, self.config.get("output_prefix", "thompson")
            )

        logger.info("✅ Simulation completed successfully!")

        # Print final statistics
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL STATISTICS")
        logger.info("=" * 80)

        # Format as a nicely formatted table
        fmt = "{:<25} {:<15} {:<15} {:<15} {:<15}"
        logger.info(
            fmt.format("AD NAME", "TRUE CTR", "ACTUAL CTR", "IMPRESSIONS", "CLICKS")
        )
        logger.info("-" * 85)

        for ad_id, ad_info in ads_stats.items():
            logger.info(
                fmt.format(
                    ad_info["name"],
                    f"{ad_info['ctr']:.2%}",
                    f"{ad_info['actual_ctr']:.2%}",
                    ad_info["impressions"],
                    ad_info["clicks"],
                )
            )

        logger.info("=" * 80)

        # Check if we found a winner
        final_status = self.check_experiment_status()
        if (
            final_status
            and final_status.get("can_stop", False)
            and final_status.get("winning_ad")
        ):
            winner = final_status["winning_ad"]
            confidence = final_status["confidence"]

            logger.info(
                f"\n🏆 WINNER: {winner['name']} with {confidence:.2%} confidence"
            )
            logger.info(f"CTR: {winner['ctr']:.4f}")
            logger.info(f"Impressions: {winner['impressions']}")
            logger.info(f"Clicks: {winner['clicks']}")

            # Add information about experiment phases
            logger.info("\n" + "=" * 80)
            logger.info("📋 EXPERIMENT PHASES SUMMARY")
            logger.info("=" * 80)

            if self.warmup_completed_at:
                logger.info(f"🚀 WARMUP PHASE: Iterations 1-{self.warmup_completed_at}")
                logger.info(
                    f"   - Each ad received approximately {self.config.get('warmup_impressions', 'N/A')} initial impressions"
                )

                if self.winner_determined_at:
                    logger.info(
                        f"🧪 TESTING PHASE: Iterations {self.warmup_completed_at + 1}-{self.winner_determined_at}"
                    )
                    logger.info(
                        f"   - Thompson Sampling actively tested all ads to find the best performer"
                    )
                else:
                    logger.info(
                        f"🧪 TESTING PHASE: Iterations {self.warmup_completed_at + 1}+"
                    )
                    logger.info(
                        f"   - Thompson Sampling actively testing (no winner determined yet)"
                    )
            else:
                if self.winner_determined_at:
                    logger.info(
                        f"🧪 TESTING PHASE: Iterations 1-{self.winner_determined_at}"
                    )
                    logger.info(
                        f"   - Thompson Sampling tested all ads until a winner was found"
                    )
                else:
                    logger.info(f"🧪 TESTING PHASE: All iterations")
                    logger.info(
                        f"   - Thompson Sampling actively testing (no winner determined yet)"
                    )

            if self.winner_determined_at:
                logger.info(
                    f"🏆 POST-WINNER PHASE: Iterations {self.winner_determined_at + 1}+"
                )

                # Add information about respect_winner setting
                if self.config.get("respect_winner", True):
                    logger.info(
                        f"   - The simulation was configured with respect_winner=True, "
                        f"which means that after the winner was determined (iteration {self.winner_determined_at}), "
                        f"only the winning ad ({winner['name']}) was shown."
                    )
                else:
                    logger.info(
                        f"   - The simulation was configured with respect_winner=False, "
                        f"which means that even after a winner was determined, "
                        f"Thompson Sampling continued to be used for ad selection."
                    )

            logger.info("=" * 80)
        else:
            logger.info("\n⚠️ No winner determined. Experiment cannot be stopped yet.")
            if final_status and "recommendation" in final_status:
                logger.info(f"Recommendation: {final_status['recommendation']}")

        logger.info("\n" + "=" * 80)
        logger.info(f"📝 Complete logs available at: {os.path.abspath(log_file)}")
        logger.info(f"📊 Results saved to: {os.path.abspath(RESULTS_DIR)}")
        logger.info("=" * 80 + "\n")


def main():
    """
    Main function to run the simulation with command-line arguments.

    Parses command-line arguments, sets up configuration, and runs the simulator.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Thompson Sampling Service Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the Thompson Sampling service",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=10000,
        help="Total number of simulated requests",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum number of impressions before considering stopping",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for determining the winning ad",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Number of warmup impressions per ad",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="thompson",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Disable visualization"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=100,
        help="Interval for checking experiment status",
    )
    parser.add_argument(
        "--ads", type=int, default=3, help="Number of ads to create (3, 4, or 5)"
    )
    parser.add_argument(
        "--respect-winner",
        type=bool,
        default=True,
        help="Whether to respect the winning ad in selection",
    )

    args = parser.parse_args()

    # Display the cool header
    print(
        r"""
    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │  ████████╗██╗  ██╗ ██████╗ ███╗   ███╗██████╗ ███████╗   │
    │  ╚══██╔══╝██║  ██║██╔═══██╗████╗ ████║██╔══██╗██╔════╝   │
    │     ██║   ███████║██║   ██║██╔████╔██║██████╔╝███████╗   │
    │     ██║   ██╔══██║██║   ██║██║╚██╔╝██║██╔═══╝ ╚════██║   │
    │     ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║     ███████║   │
    │     ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝   │
    │                                                           │
    │                  SAMPLING SIMULATOR                       │
    │                                                           │
    └───────────────────────────────────────────────────────────┘
    """
    )

    # Load configuration
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading configuration file: {e}")
            config = {}
    else:
        # Create ad configurations based on the number of ads requested
        ad_configs = []

        if args.ads >= 3:
            ad_configs.extend(
                [
                    {
                        "name": "Ad A - High CTR",
                        "content": "This is advertisement A with high CTR",
                        "ctr": 0.15,  # 15% CTR
                    },
                    {
                        "name": "Ad B - Medium CTR",
                        "content": "This is advertisement B with medium CTR",
                        "ctr": 0.10,  # 10% CTR
                    },
                    {
                        "name": "Ad C - Low CTR",
                        "content": "This is advertisement C with low CTR",
                        "ctr": 0.05,  # 5% CTR
                    },
                ]
            )

        if args.ads >= 4:
            ad_configs.append(
                {
                    "name": "Ad D - Very Low CTR",
                    "content": "This is advertisement D with very low CTR",
                    "ctr": 0.03,  # 3% CTR
                }
            )

        if args.ads >= 5:
            ad_configs.append(
                {
                    "name": "Ad E - Ultra Low CTR",
                    "content": "This is advertisement E with ultra low CTR",
                    "ctr": 0.01,  # 1% CTR
                }
            )

        # Use default configuration
        config = {
            "reset_service": True,
            "min_samples": args.min_samples,
            "confidence_threshold": args.confidence_threshold,
            "warmup_impressions": args.warmup,
            "total_requests": args.total_requests,
            "respect_winner": args.respect_winner,
            "visualize": not args.no_visualization,
            "output_prefix": args.output_prefix,
            "check_interval": args.check_interval,
            "ads": ad_configs[: args.ads],  # Use only the number of ads requested
        }

    # Print configuration summary
    print("\n\n" + "=" * 80)
    print(f"💻 SIMULATION CONFIGURATION")
    print("=" * 80)
    print(f"🔗 API URL: {args.base_url}")
    print(f"🔢 Total Requests: {config['total_requests']}")
    print(f"📝 Ads: {len(config['ads'])}")
    print(f"⚙️ Min Samples: {config['min_samples']}")
    print(f"🎯 Confidence Threshold: {config['confidence_threshold']}")
    print(f"🔄 Warmup Impressions: {config['warmup_impressions']}")
    print(f"🏆 Respect Winner: {config['respect_winner']}")
    print(f"📂 Output Directory: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 80 + "\n")

    # Run simulation
    simulator = ThompsonSamplingSimulator(args.base_url, config)
    simulator.run_simulation()


if __name__ == "__main__":
    main()
