import asyncio
import logging
import logging.handlers
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from redis.exceptions import BusyLoadingError, ConnectionError

# --- Logging Configuration ---
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")  # Default log directory
LOG_FILE_API = os.path.join(LOG_DIR, "api.log")
MAX_LOG_SIZE_MB = 10  # Max size in MB before rotation
BACKUP_COUNT = 5  # Number of backup log files to keep


def setup_logging(log_file_path: str) -> None:
    """
    Configures file-based rotating logging and stream logging.

    Parameters
    ----------
    log_file_path : str
        The full path to the log file.
    """
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s"
    )
    log_level = logging.INFO

    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Rotating File Handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_formatter)

    # Stream Handler (for console output, useful with Docker logs)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Set library log levels if needed (e.g., reduce verbosity)
    logging.getLogger("uvicorn.error").propagate = False
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn").addHandler(file_handler)
    logging.getLogger("uvicorn").addHandler(stream_handler)


# Call logging setup early
setup_logging(LOG_FILE_API)
logger = logging.getLogger(__name__)  # Get logger after setup


# Models
class Advertisement(BaseModel):
    """Advertisement data model for creating new ads."""

    id: Optional[str] = None
    name: str
    content: str


class AdvertisementStats(BaseModel):
    """Model for advertisement statistics."""

    id: str
    name: str
    impressions: int = 0
    clicks: int = 0
    alpha: float = 1.0  # Prior alpha (successes + 1)
    beta: float = 1.0  # Prior beta (failures + 1)
    ctr: float = 0.0  # Click-through rate
    probability_best: float = 0.0  # Probability of being the best variant


class SuccessEvent(BaseModel):
    """Model for recording a success event (click)."""

    ad_id: str


class ExperimentConfig(BaseModel):
    """Configuration for the experiment stopping criteria."""

    min_samples: int = Field(
        1000,
        description="Minimum number of impressions per variant before considering stopping",
    )
    confidence_threshold: float = Field(
        0.95, description="Probability threshold to determine the winning variant"
    )
    simulation_count: int = Field(
        10000, description="Number of simulations for Monte Carlo estimation"
    )
    warmup_impressions: int = Field(
        100, description="Number of impressions per ad during warmup phase"
    )


class ExperimentStatus(BaseModel):
    """Status of the experiment, including whether it can be stopped."""

    can_stop: bool
    winning_ad: Optional[AdvertisementStats] = None
    confidence: float
    min_samples_reached: bool
    total_impressions: int
    in_warmup_phase: bool
    recommendation: str


# Redis connection
async def get_redis() -> AsyncGenerator[redis.Redis, None]:
    """
    Get Redis connection from pool.

    This function is designed to be used with FastAPI's dependency injection
    to automatically close the connection when done.

    Yields:
        redis.Redis: Redis connection from the pool
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    try:
        yield r  # Yields the Redis connection to the caller
    finally:
        await r.aclose()  # Ensures connection is closed after use


# Lifespan event to initialize Redis connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application")
    # Initialize experiment config with default values if not exists
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Connect to Redis with retry mechanism for BusyLoadingError
    r = None
    max_retries = 5
    retry_delay = 1.0  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            r = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await r.ping()  # Test the connection
            logger.info(f"Successfully connected to Redis after {attempt} attempt(s)")
            break
        except BusyLoadingError:
            if attempt < max_retries:
                logger.warning(
                    f"Redis is loading dataset. Retrying in {retry_delay}s (attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(
                    "Failed to connect to Redis: Redis is still loading dataset after maximum retries"
                )
                raise
        except ConnectionError as e:
            if attempt < max_retries:
                logger.warning(
                    f"Redis connection error: {e}. Retrying in {retry_delay}s (attempt {attempt}/{max_retries})"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(
                    f"Failed to connect to Redis after {max_retries} attempts: {e}"
                )
                raise

    if r is None:
        logger.error("Failed to establish Redis connection")
        raise RuntimeError("Failed to establish Redis connection")

    # Set default experiment config if not exists
    if not await r.exists("experiment_config"):
        default_config = ExperimentConfig()
        await r.hset(
            "experiment_config",
            mapping={
                "min_samples": str(default_config.min_samples),
                "confidence_threshold": str(default_config.confidence_threshold),
                "simulation_count": str(default_config.simulation_count),
                "warmup_impressions": str(default_config.warmup_impressions),
            },
        )

    await r.aclose()
    yield
    # Shutdown
    logger.info("Shutting down the application")


# Create FastAPI application
app = FastAPI(
    title="Thompson Sampling Ad Service with Warmup",
    description="A microservice that uses Thompson Sampling with warmup phase for advertisement testing",
    version="1.0.0",
    lifespan=lifespan,
)


# Helper functions for winner persistence
async def get_stored_winner(r: redis.Redis) -> Optional[Dict]:
    """
    Get the stored experiment winner from Redis.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Optional[Dict]: Winner information or None if no winner is stored
    """
    winner_data = await r.hgetall("experiment:winner")

    if not winner_data:
        return None

    return {
        "ad_id": winner_data.get("ad_id"),
        "name": winner_data.get("name"),
        "confidence": float(winner_data.get("confidence", 0)),
        "timestamp": int(winner_data.get("timestamp", 0)),
        "impressions": int(winner_data.get("impressions", 0)),
        "clicks": int(winner_data.get("clicks", 0)),
        "ctr": float(winner_data.get("ctr", 0)),
    }


async def set_experiment_winner(
    r: redis.Redis, winning_ad: AdvertisementStats, confidence: float
) -> bool:
    """
    Store the winning advertisement in Redis.

    Args:
        r (redis.Redis): Redis connection
        winning_ad (AdvertisementStats): The winning advertisement
        confidence (float): Confidence level

    Returns:
        bool: True if winner was set, False otherwise
    """
    if not winning_ad:
        return False

    # Get the current timestamp
    timestamp = int(time.time())

    # Store winner information in Redis
    await r.hset(
        "experiment:winner",
        mapping={
            "ad_id": winning_ad.id,
            "name": winning_ad.name,
            "confidence": str(confidence),
            "timestamp": str(timestamp),
            "impressions": str(winning_ad.impressions),
            "clicks": str(winning_ad.clicks),
            "ctr": str(winning_ad.ctr),
        },
    )

    logger.info(
        f"Set experiment winner: {winning_ad.id} with confidence {confidence:.2f}"
    )
    return True


# Helper functions
async def get_all_ads(r: redis.Redis) -> List[AdvertisementStats]:
    """
    Get all advertisements with their statistics.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        List[AdvertisementStats]: List of all advertisements with statistics
    """
    ad_ids = await r.smembers("ads")
    ads = []

    for ad_id in ad_ids:
        ad_data = await r.hgetall(f"ad:{ad_id}")
        if ad_data:
            try:
                impressions = int(ad_data.get("impressions", 0))
                clicks = int(ad_data.get("clicks", 0))
                ctr = clicks / impressions if impressions > 0 else 0

                ad = AdvertisementStats(
                    id=ad_id,
                    name=ad_data.get("name", "Unknown"),
                    impressions=impressions,
                    clicks=clicks,
                    alpha=float(ad_data.get("alpha", 1.0)),
                    beta=float(ad_data.get("beta", 1.0)),
                    ctr=ctr,
                    probability_best=float(ad_data.get("probability_best", 0.0)),
                )
                ads.append(ad)
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing ad {ad_id}: {e}")

    return ads


async def calculate_probability_being_best(
    ads: List[AdvertisementStats], simulation_count: int = 10000
) -> Dict[str, float]:
    """
    Calculate the probability of each advertisement being the best using Monte Carlo simulation.

    Args:
        ads (List[AdvertisementStats]): List of advertisements with their statistics
        simulation_count (int): Number of simulations to run

    Returns:
        Dict[str, float]: Dictionary mapping ad_id to probability of being best
    """
    if not ads:
        return {}

    # Initialize counters for each ad
    best_counts = {ad.id: 0 for ad in ads}

    # Run Monte Carlo simulation
    for _ in range(simulation_count):
        max_sample = -float("inf")
        best_ad_id = None

        # For each simulation, sample from beta distribution for each ad
        for ad in ads:
            # Sample CTR from beta distribution
            sample = np.random.beta(ad.alpha, ad.beta)

            if sample > max_sample:
                max_sample = sample
                best_ad_id = ad.id

        # Increment counter for the best ad in this simulation
        if best_ad_id is not None:
            best_counts[best_ad_id] += 1

    # Calculate probabilities
    probabilities = {
        ad_id: count / simulation_count for ad_id, count in best_counts.items()
    }
    return probabilities


async def get_experiment_config(r: redis.Redis) -> ExperimentConfig:
    """
    Get experiment configuration from Redis.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        ExperimentConfig: Experiment configuration
    """
    config_data = await r.hgetall("experiment_config")
    if not config_data:
        # Use default config
        return ExperimentConfig()

    return ExperimentConfig(
        min_samples=int(config_data.get("min_samples", 1000)),
        confidence_threshold=float(config_data.get("confidence_threshold", 0.95)),
        simulation_count=int(config_data.get("simulation_count", 10000)),
        warmup_impressions=int(config_data.get("warmup_impressions", 100)),
    )


async def check_experiment_status(r: redis.Redis) -> ExperimentStatus:
    """
    Check if the experiment can be stopped based on configured criteria.
    If criteria are met, store the winner in Redis.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        ExperimentStatus: Status of the experiment
    """
    # First check if we already have a winner stored
    stored_winner = await get_stored_winner(r)

    if stored_winner:
        # We have a previously determined winner
        ad_id = stored_winner["ad_id"]

        # Get ad data
        ad_data = await r.hgetall(f"ad:{ad_id}")
        if not ad_data:
            logger.error(f"Stored winner ad {ad_id} not found in database")
            # Continue with normal experiment status check
        else:
            impressions = int(ad_data.get("impressions", 0))
            clicks = int(ad_data.get("clicks", 0))
            ctr = clicks / impressions if impressions > 0 else 0

            winning_ad = AdvertisementStats(
                id=ad_id,
                name=ad_data.get("name", stored_winner["name"]),
                impressions=impressions,
                clicks=clicks,
                alpha=float(ad_data.get("alpha", 1.0)),
                beta=float(ad_data.get("beta", 1.0)),
                ctr=ctr,
                probability_best=float(
                    ad_data.get("probability_best", 1.0)
                ),  # Winner has 100% probability
            )

            return ExperimentStatus(
                can_stop=True,
                winning_ad=winning_ad,
                confidence=stored_winner["confidence"],
                min_samples_reached=True,
                total_impressions=impressions,
                in_warmup_phase=False,
                recommendation=f"Experiment already concluded. Advertisement '{winning_ad.name}' is the winner with {stored_winner['confidence']:.1%} confidence.",
            )

    # Get ads and config
    ads = await get_all_ads(r)
    config = await get_experiment_config(r)

    if not ads:
        return ExperimentStatus(
            can_stop=False,
            confidence=0.0,
            min_samples_reached=False,
            total_impressions=0,
            in_warmup_phase=False,
            recommendation="No advertisements available for testing",
        )

    # Check if in warmup phase
    in_warmup_phase = any(ad.impressions < config.warmup_impressions for ad in ads)

    # Check if all ads have minimum sample size
    total_impressions = sum(ad.impressions for ad in ads)
    min_samples_reached = all(ad.impressions >= config.min_samples for ad in ads)

    # Calculate probability of each ad being the best
    probabilities = await calculate_probability_being_best(ads, config.simulation_count)

    # Update probabilities in Redis
    for ad_id, prob in probabilities.items():
        await r.hset(f"ad:{ad_id}", "probability_best", str(prob))

    # Find ad with highest probability
    best_ad_id = max(probabilities, key=probabilities.get) if probabilities else None
    best_probability = probabilities.get(best_ad_id, 0) if best_ad_id else 0

    # Update ad objects with probabilities
    for ad in ads:
        ad.probability_best = probabilities.get(ad.id, 0.0)

    # Get the winning ad
    winning_ad = next((ad for ad in ads if ad.id == best_ad_id), None)

    # Determine if experiment can be stopped
    # Must be out of warmup phase, have minimum samples, and meet confidence threshold
    can_stop = (
        (not in_warmup_phase)
        and min_samples_reached
        and best_probability >= config.confidence_threshold
    )

    # If experiment can be stopped, store the winner
    if can_stop and winning_ad:
        await set_experiment_winner(r, winning_ad, best_probability)

    # Generate recommendation
    if in_warmup_phase:
        remaining_warmup = sum(
            max(0, config.warmup_impressions - ad.impressions) for ad in ads
        )
        recommendation = f"In warmup phase. Need approximately {remaining_warmup} more impressions to complete warmup."
    elif can_stop:
        recommendation = f"Experiment can be stopped. Advertisement '{winning_ad.name}' is the winner with {best_probability:.1%} confidence."
    elif not min_samples_reached:
        remaining_impressions = sum(
            max(0, config.min_samples - ad.impressions) for ad in ads
        )
        recommendation = f"Continue testing. Need approximately {remaining_impressions} more impressions to reach minimum sample size."
    else:
        recommendation = f"Continue testing. Best advertisement has {best_probability:.1%} confidence, below threshold of {config.confidence_threshold:.1%}."

    return ExperimentStatus(
        can_stop=can_stop,
        winning_ad=winning_ad,
        confidence=best_probability,
        min_samples_reached=min_samples_reached,
        total_impressions=total_impressions,
        in_warmup_phase=in_warmup_phase,
        recommendation=recommendation,
    )


# Routes
@app.post("/ads", status_code=status.HTTP_201_CREATED)
async def create_ad(ad: Advertisement, r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Create a new advertisement.

    Args:
        ad (Advertisement): Advertisement data
        r (redis.Redis): Redis connection

    Returns:
        Dict: Created advertisement data
    """
    # Generate ID if not provided
    if not ad.id:
        ad.id = f"ad_{int(await r.incr('ad_counter'))}"

    # Check if ad already exists
    exists = await r.sismember("ads", ad.id)
    if exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Advertisement with ID {ad.id} already exists",
        )

    # Add to Redis
    await r.sadd("ads", ad.id)
    await r.hset(
        f"ad:{ad.id}",
        mapping={
            "name": ad.name,
            "content": ad.content,
            "impressions": 0,
            "clicks": 0,
            "alpha": 1.0,  # Prior
            "beta": 1.0,  # Prior
            "probability_best": 0.0,  # Initial probability of being best
        },
    )

    logger.info(f"Created new advertisement: {ad.id}")
    return {
        "id": ad.id,
        "name": ad.name,
        "content": ad.content,
        "message": "Advertisement created successfully",
    }


@app.get("/ads/select")
async def select_ad(
    respect_winner: bool = Query(
        False,
        description="Whether to always select the winning ad if experiment can be stopped",
    ),
    r: redis.Redis = Depends(get_redis),
) -> Dict:
    """
    Select an advertisement using Thompson Sampling or return the winning ad if experiment can be stopped.
    During warmup phase, ads are selected uniformly to ensure each gets minimum exposure.

    Args:
        respect_winner (bool): Whether to return the winning ad if experiment can be stopped
        r (redis.Redis): Redis connection

    Returns:
        Dict: Selected advertisement with content
    """
    # Get all ads
    ads = await get_all_ads(r)

    if not ads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No advertisements available"
        )

    # First check if we should respect the winner and if a winner exists
    if respect_winner:
        stored_winner = await get_stored_winner(r)

        if stored_winner:
            # We have a stored winner, return it
            ad_id = stored_winner["ad_id"]

            # Get ad content
            content = await r.hget(f"ad:{ad_id}", "content")

            # Record impression
            await r.hincrby(f"ad:{ad_id}", "impressions", 1)
            await r.hincrbyfloat(
                f"ad:{ad_id}", "beta", 1.0
            )  # Assume no click initially

            logger.info(f"Selected stored winner advertisement: {ad_id}")
            return {
                "id": ad_id,
                "name": stored_winner["name"],
                "content": content,
                "selection_method": "winning_ad",
            }

        # No stored winner, check if experiment can be stopped now
        experiment_status = await check_experiment_status(r)
        if experiment_status.can_stop and experiment_status.winning_ad:
            # Return the winning ad
            selected_ad = experiment_status.winning_ad

            # Record impression
            await r.hincrby(f"ad:{selected_ad.id}", "impressions", 1)
            await r.hincrbyfloat(
                f"ad:{selected_ad.id}", "beta", 1.0
            )  # Assume no click initially

            # Get content
            content = await r.hget(f"ad:{selected_ad.id}", "content")

            logger.info(f"Selected winning advertisement: {selected_ad.id}")
            return {
                "id": selected_ad.id,
                "name": selected_ad.name,
                "content": content,
                "selection_method": "winning_ad",
            }

    # Get config for warmup check
    config = await get_experiment_config(r)

    # Check if we're in warmup phase for any ad
    ads_in_warmup = [ad for ad in ads if ad.impressions < config.warmup_impressions]

    if ads_in_warmup:
        # Select ad with least impressions during warmup to ensure even distribution
        selected_ad = min(ads_in_warmup, key=lambda a: a.impressions)

        # Record impression
        await r.hincrby(f"ad:{selected_ad.id}", "impressions", 1)
        await r.hincrbyfloat(
            f"ad:{selected_ad.id}", "beta", 1.0
        )  # Assume no click initially

        # Get content
        content = await r.hget(f"ad:{selected_ad.id}", "content")

        logger.info(f"Selected advertisement (warmup phase): {selected_ad.id}")
        return {
            "id": selected_ad.id,
            "name": selected_ad.name,
            "content": content,
            "selection_method": "warmup_phase",
        }

    # Otherwise use Thompson Sampling - sample from beta distribution for each ad
    max_sample = -1
    selected_ad = None

    for ad in ads:
        # Sample from beta distribution
        sample = np.random.beta(ad.alpha, ad.beta)
        if sample > max_sample:
            max_sample = sample
            selected_ad = ad

    if not selected_ad:
        selected_ad = ads[0]  # Fallback

    # Record impression
    await r.hincrby(f"ad:{selected_ad.id}", "impressions", 1)
    await r.hincrbyfloat(
        f"ad:{selected_ad.id}", "beta", 1.0
    )  # Assume no click initially

    # Get content
    content = await r.hget(f"ad:{selected_ad.id}", "content")

    logger.info(f"Selected advertisement (thompson sampling): {selected_ad.id}")
    return {
        "id": selected_ad.id,
        "name": selected_ad.name,
        "content": content,
        "selection_method": "thompson_sampling",
    }


@app.post("/ads/click")
async def record_click(
    event: SuccessEvent, r: redis.Redis = Depends(get_redis)
) -> Dict:
    """
    Record a click (success) for an advertisement.

    Args:
        event (SuccessEvent): Event data with ad_id
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    # Check if ad exists
    exists = await r.sismember("ads", event.ad_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement with ID {event.ad_id} not found",
        )

    # Update clicks and alpha
    await r.hincrby(f"ad:{event.ad_id}", "clicks", 1)
    await r.hincrbyfloat(f"ad:{event.ad_id}", "alpha", 1.0)

    logger.info(f"Recorded click for advertisement: {event.ad_id}")
    return {"message": f"Click recorded for advertisement {event.ad_id}"}


@app.post("/ads/impression/{ad_id}")
async def record_impression(ad_id: str, r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Manually record an impression for an advertisement.

    Args:
        ad_id (str): Advertisement ID
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    # Check if ad exists
    exists = await r.sismember("ads", ad_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement with ID {ad_id} not found",
        )

    # Update impression and beta (failure)
    await r.hincrby(f"ad:{ad_id}", "impressions", 1)
    await r.hincrbyfloat(f"ad:{ad_id}", "beta", 1.0)

    logger.info(f"Recorded impression for advertisement: {ad_id}")
    return {"message": f"Impression recorded for advertisement {ad_id}"}


@app.post("/ads/no-click/{ad_id}")
async def record_no_click(ad_id: str, r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Record explicitly that an ad was shown but not clicked.

    Args:
        ad_id (str): Advertisement ID
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    # Check if ad exists
    exists = await r.sismember("ads", ad_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement with ID {ad_id} not found",
        )

    # Update beta (failure) only
    await r.hincrbyfloat(f"ad:{ad_id}", "beta", 1.0)

    logger.info(f"Recorded no-click for advertisement: {ad_id}")
    return {"message": f"No-click recorded for advertisement {ad_id}"}


@app.get("/ads")
async def get_ads(r: redis.Redis = Depends(get_redis)) -> List[AdvertisementStats]:
    """
    Get all advertisements with statistics.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        List[AdvertisementStats]: List of all advertisements with statistics
    """
    return await get_all_ads(r)


@app.get("/ads/{ad_id}")
async def get_ad(ad_id: str, r: redis.Redis = Depends(get_redis)) -> AdvertisementStats:
    """
    Get a specific advertisement with statistics.

    Args:
        ad_id (str): Advertisement ID
        r (redis.Redis): Redis connection

    Returns:
        AdvertisementStats: Advertisement with statistics
    """
    exists = await r.sismember("ads", ad_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement with ID {ad_id} not found",
        )

    ad_data = await r.hgetall(f"ad:{ad_id}")
    if not ad_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement data for ID {ad_id} not found",
        )

    impressions = int(ad_data.get("impressions", 0))
    clicks = int(ad_data.get("clicks", 0))
    ctr = clicks / impressions if impressions > 0 else 0

    return AdvertisementStats(
        id=ad_id,
        name=ad_data.get("name", "Unknown"),
        impressions=impressions,
        clicks=clicks,
        alpha=float(ad_data.get("alpha", 1.0)),
        beta=float(ad_data.get("beta", 1.0)),
        ctr=ctr,
        probability_best=float(ad_data.get("probability_best", 0.0)),
    )


@app.get("/experiment/status")
async def get_experiment_status_endpoint(
    r: redis.Redis = Depends(get_redis),
) -> ExperimentStatus:
    """
    Get the current status of the experiment and check if it can be stopped.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        ExperimentStatus: Status of the experiment
    """
    return await check_experiment_status(r)


@app.put("/experiment/config")
async def update_experiment_config(
    config: ExperimentConfig, r: redis.Redis = Depends(get_redis)
) -> Dict:
    """
    Update the experiment configuration.

    Args:
        config (ExperimentConfig): New configuration
        r (redis.Redis): Redis connection

    Returns:
        Dict: Updated configuration
    """
    await r.hset(
        "experiment_config",
        mapping={
            "min_samples": str(config.min_samples),
            "confidence_threshold": str(config.confidence_threshold),
            "simulation_count": str(config.simulation_count),
            "warmup_impressions": str(config.warmup_impressions),
        },
    )

    logger.info(f"Updated experiment configuration: {config}")
    return {
        "message": "Experiment configuration updated successfully",
        "config": config,
    }


@app.get("/experiment/config", response_model=ExperimentConfig)
async def get_experiment_config_endpoint(
    r: redis.Redis = Depends(get_redis),
) -> ExperimentConfig:
    """
    Retrieve the current experiment configuration settings.

    Fetches the configuration values stored in Redis under the 'experiment_config' hash.
    If the configuration doesn't exist in Redis, it returns the default settings
    defined in the ExperimentConfig model.

    Parameters
    ----------
    r : redis.Redis
        An asynchronous Redis connection instance obtained via dependency injection.

    Returns
    -------
    ExperimentConfig
        The current experiment configuration settings.
    """
    logger.info("Received request to get experiment configuration.")
    # Reuse the existing helper function which handles fetching from Redis and defaults
    config = await get_experiment_config(r)
    logger.info(f"Returning current experiment configuration: {config}")
    return config


@app.delete("/ads/{ad_id}")
async def delete_ad(ad_id: str, r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Delete an advertisement.

    Args:
        ad_id (str): Advertisement ID
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    exists = await r.sismember("ads", ad_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Advertisement with ID {ad_id} not found",
        )

    # Remove from set and delete hash
    await r.srem("ads", ad_id)
    await r.delete(f"ad:{ad_id}")

    logger.info(f"Deleted advertisement: {ad_id}")
    return {"message": f"Advertisement {ad_id} deleted successfully"}


@app.delete("/ads")
async def reset_all_ads(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Delete all advertisements (for testing purposes).

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    # Get all ad IDs
    ad_ids = await r.smembers("ads")

    # Delete each ad hash
    for ad_id in ad_ids:
        await r.delete(f"ad:{ad_id}")

    # Clear the set
    await r.delete("ads")

    # Reset counter
    await r.set("ad_counter", 0)

    # Also reset any stored winner
    await r.delete("experiment:winner")

    logger.info("Reset all advertisements")
    return {"message": "All advertisements deleted successfully"}


@app.get("/experiment/winner")
async def get_experiment_winner_endpoint(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Get the winning advertisement if experiment can be stopped.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Winning advertisement data or error message
    """
    # First check if we have a stored winner
    stored_winner = await get_stored_winner(r)

    if stored_winner:
        # We have a previously determined winner
        ad_id = stored_winner["ad_id"]

        # Check if ad still exists
        exists = await r.sismember("ads", ad_id)
        if not exists:
            # Ad has been deleted, but we still have a winner
            return {
                "id": ad_id,
                "name": stored_winner["name"],
                "content": "Advertisement content not available (ad deleted)",
                "confidence": stored_winner["confidence"],
                "ctr": stored_winner["ctr"],
                "impressions": stored_winner["impressions"],
                "clicks": stored_winner["clicks"],
                "warning": "The winning ad has been deleted from the system",
            }

        # Get content
        content = await r.hget(f"ad:{ad_id}", "content")
        if content is None:
            content = "Content not available"

        return {
            "id": ad_id,
            "name": stored_winner["name"],
            "content": content,
            "confidence": stored_winner["confidence"],
            "ctr": stored_winner["ctr"],
            "impressions": stored_winner["impressions"],
            "clicks": stored_winner["clicks"],
        }

    # No stored winner, check experiment status
    status = await check_experiment_status(r)

    if not status.can_stop:
        detail_message = (
            f"Experiment cannot be stopped yet. {status.recommendation} "
            f"In warmup: {status.in_warmup_phase}, Min samples reached: {status.min_samples_reached}, "
            f"Total impressions: {status.total_impressions}, Best confidence: {status.confidence:.2f}"
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail_message)

    if not status.winning_ad:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No winning advertisement found",
        )

    # Get content
    content = await r.hget(f"ad:{status.winning_ad.id}", "content")
    if content is None:
        content = "Content not available"

    return {
        "id": status.winning_ad.id,
        "name": status.winning_ad.name,
        "content": content,
        "confidence": status.confidence,
        "ctr": status.winning_ad.ctr,
        "impressions": status.winning_ad.impressions,
        "clicks": status.winning_ad.clicks,
    }


@app.delete("/experiment/winner/reset")
async def reset_experiment_winner(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Reset (delete) the stored experiment winner.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Confirmation message
    """
    # Check if winner exists before deleting
    winner_data = await r.hgetall("experiment:winner")
    if not winner_data:
        # No winner found, return appropriate message
        return {"message": "No experiment winner exists to reset"}

    # Delete the winner
    await r.delete("experiment:winner")

    logger.info("Reset experiment winner")
    return {"message": "Experiment winner reset successfully"}


@app.get("/experiment/warmup/status")
async def get_warmup_status(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Get the current status of the warmup phase.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Warmup phase status details
    """
    ads = await get_all_ads(r)
    config = await get_experiment_config(r)

    if not ads:
        return {
            "in_warmup_phase": False,
            "message": "No advertisements available",
            "ads_status": [],
        }

    ads_status = []
    for ad in ads:
        remaining = max(0, config.warmup_impressions - ad.impressions)
        complete = ad.impressions >= config.warmup_impressions

        ads_status.append(
            {
                "id": ad.id,
                "name": ad.name,
                "impressions": ad.impressions,
                "required_warmup": config.warmup_impressions,
                "remaining": remaining,
                "warmup_complete": complete,
            }
        )

    in_warmup = any(not status["warmup_complete"] for status in ads_status)
    total_remaining = sum(status["remaining"] for status in ads_status)

    return {
        "in_warmup_phase": in_warmup,
        "total_warmup_remaining": total_remaining,
        "warmup_impressions_per_ad": config.warmup_impressions,
        "ads_status": ads_status,
        "message": (
            "Warmup phase complete"
            if not in_warmup
            else f"Warmup phase in progress. {total_remaining} impressions remaining."
        ),
    }


# Health check
@app.get("/health")
async def health_check(r: redis.Redis = Depends(get_redis)) -> Dict[str, str]:
    """
    Health check endpoint.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict[str, str]: Health status

    Raises:
        HTTPException: If the health check fails (e.g., Redis connection error).
    """
    try:
        await r.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "redis": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
