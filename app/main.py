import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


class ExperimentStatus(BaseModel):
    """Status of the experiment, including whether it can be stopped."""

    can_stop: bool
    winning_ad: Optional[AdvertisementStats] = None
    confidence: float
    min_samples_reached: bool
    total_impressions: int
    recommendation: str


# Redis connection
async def get_redis() -> redis.Redis:
    """
    Get Redis connection from pool.

    Returns:
        redis.Redis: Redis connection
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    try:
        yield r
    finally:
        await r.aclose()


# Lifespan event to initialize Redis connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application")
    # Initialize experiment config with default values if not exists
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)

    # Set default experiment config if not exists
    if not await r.exists("experiment_config"):
        default_config = ExperimentConfig()
        await r.hset(
            "experiment_config",
            mapping={
                "min_samples": str(default_config.min_samples),
                "confidence_threshold": str(default_config.confidence_threshold),
                "simulation_count": str(default_config.simulation_count),
            },
        )

    await r.aclose()
    yield
    # Shutdown
    logger.info("Shutting down the application")


# Create FastAPI application
app = FastAPI(
    title="Thompson Sampling Ad Service",
    description="A microservice that uses Thompson Sampling for advertisement testing",
    version="1.0.0",
    lifespan=lifespan,
)


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
    )


async def check_experiment_status(r: redis.Redis) -> ExperimentStatus:
    """
    Check if the experiment can be stopped based on configured criteria.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        ExperimentStatus: Status of the experiment
    """
    # Get ads and config
    ads = await get_all_ads(r)
    config = await get_experiment_config(r)

    if not ads:
        return ExperimentStatus(
            can_stop=False,
            confidence=0.0,
            min_samples_reached=False,
            total_impressions=0,
            recommendation="No advertisements available for testing",
        )

    # Check if all ads have minimum sample size
    total_impressions = sum(ad.impressions for ad in ads)
    min_samples_reached = all(ad.impressions >= config.min_samples for ad in ads)

    # Calculate probability of each ad being the best
    probabilities = await calculate_probability_being_best(ads, config.simulation_count)

    # Update probabilities in Redis
    for ad_id, prob in probabilities.items():
        await r.hset(f"ad:{ad_id}", "probability_best", str(prob))

    # Find ad with highest probability
    best_ad_id = max(probabilities, key=probabilities.get)
    best_probability = probabilities[best_ad_id]

    # Update ad objects with probabilities
    for ad in ads:
        ad.probability_best = probabilities.get(ad.id, 0.0)

    # Get the winning ad
    winning_ad = next((ad for ad in ads if ad.id == best_ad_id), None)

    # Determine if experiment can be stopped
    can_stop = min_samples_reached and best_probability >= config.confidence_threshold

    # Generate recommendation
    if can_stop:
        recommendation = f"Experiment can be stopped. Advertisement '{winning_ad.name}' is the winner with {best_probability:.1%} confidence."
    elif not min_samples_reached:
        remaining_impressions = max(
            0, config.min_samples * len(ads) - total_impressions
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

    Args:
        respect_winner (bool): Whether to return the winning ad if experiment can be stopped
        r (redis.Redis): Redis connection

    Returns:
        Dict: Selected advertisement with content
    """
    # Check if experiment can be stopped and we should respect the winner
    if respect_winner:
        experiment_status = await check_experiment_status(r)
        if experiment_status.can_stop and experiment_status.winning_ad:
            # Return the winning ad
            selected_ad = experiment_status.winning_ad

            # Record impression
            await r.hincrby(f"ad:{selected_ad.id}", "impressions", 1)

            # Get content
            content = await r.hget(f"ad:{selected_ad.id}", "content")

            logger.info(f"Selected winning advertisement: {selected_ad.id}")
            return {
                "id": selected_ad.id,
                "name": selected_ad.name,
                "content": content,
                "selection_method": "winning_ad",
            }

    # Otherwise use Thompson Sampling
    ads = await get_all_ads(r)

    if not ads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No advertisements available"
        )

    # Thompson Sampling - sample from beta distribution for each ad
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

    # Get content
    content = await r.hget(f"ad:{selected_ad.id}", "content")

    logger.info(f"Selected advertisement: {selected_ad.id}")
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
async def get_experiment_status(
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


@app.get("/experiment/winner")
async def get_experiment_winner(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Get the winning advertisement if experiment can be stopped.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Winning advertisement data or error message
    """
    status = await check_experiment_status(r)

    if not status.can_stop:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment cannot be stopped yet. {status.recommendation}",
        )

    if not status.winning_ad:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No winning advertisement found",
        )

    # Get content
    content = await r.hget(f"ad:{status.winning_ad.id}", "content")

    return {
        "id": status.winning_ad.id,
        "name": status.winning_ad.name,
        "content": content,
        "confidence": status.confidence,
        "ctr": status.winning_ad.ctr,
        "impressions": status.winning_ad.impressions,
        "clicks": status.winning_ad.clicks,
    }


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
        },
    )

    logger.info(f"Updated experiment configuration: {config}")
    return {
        "message": "Experiment configuration updated successfully",
        "config": config,
    }


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

    logger.info("Reset all advertisements")
    return {"message": "All advertisements deleted successfully"}


# Health check
@app.get("/health")
async def health_check(r: redis.Redis = Depends(get_redis)) -> Dict:
    """
    Health check endpoint.

    Args:
        r (redis.Redis): Redis connection

    Returns:
        Dict: Health status
    """
    try:
        # Check Redis connection
        await r.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "redis": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
