#!/usr/bin/env python3
# simulation.py - Script for traffic simulation and testing of Thompson Sampling microservice

import argparse
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ThompsonSamplingSimulator:
    """
    Simulator for testing Thompson Sampling microservice.

    This class provides functionality to simulate traffic to a Thompson Sampling
    microservice for A/B testing, sending impressions and clicks, and analyzing
    the results to determine the most effective advertisement.
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
            logger.info(f"Service reset response: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to reset service: {e}")
            return False

    def configure_experiment(
        self, min_samples: int, confidence_threshold: float
    ) -> bool:
        """
        Configure experiment parameters.

        Parameters
        ----------
        min_samples : int
            Minimum number of impressions before stopping
        confidence_threshold : float
            Probability threshold for determining a winner

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
            }
            response = self.session.put(
                f"{self.base_url}/experiment/config", json=config
            )
            logger.info(f"Experiment config set to: {config}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to configure experiment: {e}")
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
                    }

                    logger.info(
                        f"Created ad: {ad_id} - {ad_config['name']} (CTR: {ad_config['ctr']})"
                    )
                else:
                    logger.error(
                        f"Failed to create ad {ad_config['name']}: {response.status_code} - {response.text}"
                    )
                    success = False

            except Exception as e:
                logger.error(f"Error creating ad {ad_config['name']}: {e}")
                success = False

        return success

    def simulate_traffic(
        self, total_requests: int, respect_winner: bool = False
    ) -> Tuple[Dict, List[Dict]]:
        """
        Simulate traffic with impressions and clicks.

        Parameters
        ----------
        total_requests : int
            Total number of requests to simulate
        respect_winner : bool, optional
            Whether to respect the experiment winner when selecting ads, by default False

        Returns
        -------
        Tuple[Dict, List[Dict]]
            Final statistics and history of impressions/clicks
        """
        # Metrics for each ad
        metrics = {ad_id: {"impressions": 0, "clicks": 0} for ad_id in self.ads}

        # History of ad selections
        history = []

        # Progress bar
        with tqdm(total=total_requests, desc="Simulating traffic") as pbar:
            for i in range(total_requests):
                try:
                    # Get ad recommendation
                    response = self.session.get(
                        f"{self.base_url}/ads/select?respect_winner={str(respect_winner).lower()}"
                    )

                    if response.status_code != 200:
                        logger.error(
                            f"Failed to select ad: {response.status_code} - {response.text}"
                        )
                        continue

                    ad_data = response.json()
                    ad_id = ad_data["id"]
                    selection_method = ad_data.get("selection_method", "unknown")

                    if ad_id not in self.ads:
                        logger.warning(f"Unknown ad selected: {ad_id}")
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
                                f"Failed to record click: {click_response.status_code} - {click_response.text}"
                            )
                    else:
                        # If not clicked, send no-click event
                        no_click_response = self.session.post(
                            f"{self.base_url}/ads/no-click/{ad_id}"
                        )

                        if no_click_response.status_code != 200:
                            logger.error(
                                f"Failed to record no-click: {no_click_response.status_code} - {no_click_response.text}"
                            )

                    # Record history
                    history.append(
                        {
                            "iteration": i + 1,
                            "ad_id": ad_id,
                            "ad_name": self.ads[ad_id]["name"],
                            "is_clicked": is_clicked,
                            "selection_method": selection_method,
                        }
                    )

                    # Check experiment status every 100 requests
                    if (i + 1) % 100 == 0:
                        self.check_experiment_status()

                    # Update progress bar
                    pbar.update(1)

                    # Small delay to avoid API overload
                    time.sleep(0.01)

                except Exception as e:
                    logger.error(f"Error in simulation iteration {i}: {e}")

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
        self.check_experiment_status()

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
                            f"Experiment can be stopped. Winner: {winning_ad_name} ({winning_ad_id}) with {confidence:.2%} confidence"
                        )

                    # Get additional information for each ad
                    self.get_ads_statistics()

                return status
            else:
                logger.error(
                    f"Failed to check experiment status: {response.status_code} - {response.text}"
                )
                return {}

        except Exception as e:
            logger.error(f"Error checking experiment status: {e}")
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

                # Log statistics
                for ad in ads_stats:
                    ad_id = ad["id"]
                    if ad_id in self.ads:
                        logger.info(
                            f"Ad {ad['name']} ({ad_id}): "
                            f"Impressions={ad['impressions']}, "
                            f"Clicks={ad['clicks']}, "
                            f"CTR={ad['ctr']:.2%}, "
                            f"Probability being best={ad['probability_best']:.2%}"
                        )

                return ads_stats
            else:
                logger.error(
                    f"Failed to get ads statistics: {response.status_code} - {response.text}"
                )
                return []

        except Exception as e:
            logger.error(f"Error getting ads statistics: {e}")
            return []

    def visualize_results(
        self, history: List[Dict], output_file: Optional[str] = None
    ) -> None:
        """
        Visualize simulation results with advanced plots.

        Parameters
        ----------
        history : List[Dict]
            History of impressions and clicks
        output_file : Optional[str], optional
            Filename to save the visualization, by default None
        """
        if not history:
            logger.warning("No history data to visualize")
            return

        # Convert history to DataFrame
        df = pd.DataFrame(history)

        # Set Seaborn style for better aesthetics
        sns.set(style="whitegrid")

        # Create a color palette for consistent colors across plots
        ad_names = sorted(df["ad_name"].unique())
        num_ads = len(ad_names)
        colors = sns.color_palette("viridis", num_ads)
        color_dict = dict(zip(ad_names, colors))

        # Create a figure with 5 subplots
        fig, axs = plt.subplots(
            4, 1, figsize=(14, 20), gridspec_kw={"height_ratios": [1, 1, 1, 1.5]}
        )
        fig.suptitle(
            "Thompson Sampling Simulation Results",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # 1. Cumulative Impressions
        for i, ad_name in enumerate(ad_names):
            ad_df = df[df["ad_name"] == ad_name]
            cumulative_impressions = np.arange(1, len(ad_df) + 1)
            axs[0].plot(
                ad_df["iteration"],
                cumulative_impressions,
                color=color_dict[ad_name],
                label=f"{ad_name} ({len(ad_df)} total)",
            )

        axs[0].set_title("Cumulative Impressions by Ad", fontsize=16, pad=15)
        axs[0].set_ylabel("Number of Impressions", fontsize=12)
        axs[0].legend(loc="upper left", frameon=True, framealpha=0.9)
        axs[0].grid(True, alpha=0.3)

        # 2. Rolling CTR with confidence intervals
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
                    linewidth=2,
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

        axs[1].set_title(
            f"Rolling Average CTR with 95% Confidence Intervals (Window: {window_size})",
            fontsize=16,
            pad=15,
        )
        axs[1].set_ylabel("Click-Through Rate", fontsize=12)
        axs[1].legend(loc="best", frameon=True, framealpha=0.9)
        axs[1].set_ylim(bottom=0)
        axs[1].yaxis.set_major_formatter(PercentFormatter(1.0))
        axs[1].grid(True, alpha=0.3)

        # 3. CTR Convergence Plot (cumulative CTR over time)
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

                # Plot with log scale for x-axis to better show early convergence
                axs[2].plot(
                    ad_df["iteration"],
                    ad_df["cumulative_ctr"],
                    label=f"{ad_name}",
                    color=color_dict[ad_name],
                    linewidth=2,
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
            )

        axs[2].set_title("Cumulative CTR Convergence Over Time", fontsize=16, pad=15)
        axs[2].set_ylabel("Cumulative CTR", fontsize=12)
        axs[2].set_xlim(left=0)
        axs[2].set_ylim(bottom=0)
        axs[2].yaxis.set_major_formatter(PercentFormatter(1.0))
        axs[2].legend(loc="lower right", frameon=True, framealpha=0.9)
        axs[2].grid(True, alpha=0.3)

        # 4. Ad Selection Distribution Over Time (heatmap)
        # Divide the timeline into intervals
        intervals = 20
        max_iteration = df["iteration"].max()
        interval_size = max_iteration // intervals

        # Create a matrix for the heatmap
        heatmap_data = np.zeros((num_ads, intervals))
        interval_labels = []

        for i in range(intervals):
            start = i * interval_size + 1
            end = (i + 1) * interval_size if i < intervals - 1 else max_iteration
            interval_labels.append(f"{start}-{end}")

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
        cbar.set_label("Proportion of Impressions", rotation=270, labelpad=20)

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
                )

        axs[3].set_title("Ad Selection Distribution Over Time", fontsize=16, pad=15)
        axs[3].set_xlabel("Iteration Intervals", fontsize=12)
        axs[3].set_ylabel("Advertisement", fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.08)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Results visualization saved to {output_file}")

        plt.show()

    def run_simulation(self) -> None:
        """
        Run complete simulation based on configuration.

        This method orchestrates the entire simulation workflow including
        service reset, experiment configuration, ad creation, traffic simulation,
        and results visualization.
        """
        logger.info("Starting simulation")

        # Reset service
        if self.config.get("reset_service", True):
            logger.info("Resetting service")
            self.reset_service()

        # Configure experiment
        logger.info("Configuring experiment")
        self.configure_experiment(
            self.config.get("min_samples", 1000),
            self.config.get("confidence_threshold", 0.95),
        )

        # Create ads
        logger.info("Creating ads")
        self.create_ads(self.config["ads"])

        # Simulate traffic
        logger.info("Simulating traffic")
        ads_stats, history = self.simulate_traffic(
            self.config.get("total_requests", 10000),
            self.config.get("respect_winner", False),
        )

        # Visualize results
        if self.config.get("visualize", True):
            logger.info("Visualizing results")
            self.visualize_results(
                history, self.config.get("output_file", "simulation_results.png")
            )

        logger.info("Simulation completed")

        # Print final statistics
        logger.info("=== Final Statistics ===")
        for ad_id, ad_info in ads_stats.items():
            logger.info(
                f"Ad {ad_info['name']} ({ad_id}): "
                f"True CTR={ad_info['ctr']:.2%}, "
                f"Measured CTR={ad_info['actual_ctr']:.2%}, "
                f"Impressions={ad_info['impressions']}, "
                f"Clicks={ad_info['clicks']}"
            )


def main():
    """
    Main function to run the simulation with command-line arguments.

    Parses command-line arguments, sets up configuration, and runs the simulator.
    """
    parser = argparse.ArgumentParser(description="Thompson Sampling Service Simulator")
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
        "--output",
        type=str,
        default="simulation_results.png",
        help="Output file for visualization",
    )
    parser.add_argument(
        "--no-visualization", action="store_true", help="Disable visualization"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            config = {}
    else:
        # Use default configuration
        config = {
            "reset_service": True,
            "min_samples": args.min_samples,
            "confidence_threshold": args.confidence_threshold,
            "total_requests": args.total_requests,
            "respect_winner": True,
            "visualize": not args.no_visualization,
            "output_file": args.output,
            "ads": [
                {
                    "name": "Ad A - Low CTR",
                    "content": "This is advertisement A with low CTR",
                    "ctr": 0.05,  # 5% CTR
                },
                {
                    "name": "Ad B - Medium CTR",
                    "content": "This is advertisement B with medium CTR",
                    "ctr": 0.10,  # 10% CTR
                },
                {
                    "name": "Ad C - High CTR",
                    "content": "This is advertisement C with high CTR",
                    "ctr": 0.15,  # 15% CTR
                },
            ],
        }

    # Run simulation
    simulator = ThompsonSamplingSimulator(args.base_url, config)
    simulator.run_simulation()


if __name__ == "__main__":
    main()
