# Thompson Sampling Ad Service

A high-performance microservice for optimizing advertisement selection using Thompson Sampling - a Bayesian approach to the multi-armed bandit problem. This service dynamically allocates traffic to better-performing ad variants based on real-time click-through rates.

## Features

- **Bayesian A/B Testing**: Uses Thompson Sampling to efficiently balance exploration and exploitation of ad variants
- **Real-time Optimization**: Continuously learns and adapts traffic allocation based on user interactions
- **Early Experiment Termination**: Automatically detects winning variants with statistical confidence
- **RESTful API**: Comprehensive endpoints for managing experiments and collecting metrics
- **Simulation Tools**: Includes tools for traffic simulation and visualization
- **Redis Backend**: Fast, in-memory storage for high throughput

## Technology Stack

- **Python 3.8+**
- **FastAPI**: Modern, high-performance web framework
- **Redis**: In-memory data storage
- **NumPy**: For statistical computations
- **Uvicorn**: ASGI server
- **Matplotlib & Seaborn**: For data visualization in the simulation tool

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:yourusername/thompson-sampling-service.git
cd thompson-sampling-service

# Build and start the services
docker-compose up -d
```

The service uses Docker Compose with the following configuration:

#### Docker Compose Services

- **app**: The main Thompson Sampling service
  - Exposes port 8000 for the API
  - Connects to the Redis instance
  - Includes health checks for reliability
  - Environment variables:
    - `REDIS_URL`: Connection string for Redis
    - `LOG_LEVEL`: Logging verbosity level

- **redis**: In-memory database for storing ad statistics
  - Uses Redis 7 Alpine for minimal footprint
  - Persists data using AOF (Append Only File)
  - Exposes port 6379 (if you need to connect directly)
  - Includes health checks

#### Persistent Storage

Data is stored in a named volume (`thompson-redis-data`) to ensure your experiment data survives container restarts.

## Quick Start

1. **Create Advertisements**

```bash
# Create test advertisements
curl -X POST http://localhost:8000/ads \
  -H "Content-Type: application/json" \
  -d '{"name": "Ad A", "content": "Try our new product!"}'

curl -X POST http://localhost:8000/ads \
  -H "Content-Type: application/json" \
  -d '{"name": "Ad B", "content": "Limited time offer!"}'

curl -X POST http://localhost:8000/ads \
  -H "Content-Type: application/json" \
  -d '{"name": "Ad C", "content": "Free shipping today!"}'
```

2. **Configure Experiment Parameters**

```bash
curl -X PUT http://localhost:8000/experiment/config \
  -H "Content-Type: application/json" \
  -d '{"min_samples": 1000, "confidence_threshold": 0.95, "simulation_count": 10000}'
```

3. **Select Advertisements**

```bash
# Select an ad (Thompson Sampling)
curl http://localhost:8000/ads/select
```

4. **Record User Interactions**

```bash
# Record a click
curl -X POST http://localhost:8000/ads/click \
  -H "Content-Type: application/json" \
  -d '{"ad_id": "ad_1"}'
```

5. **Check Experiment Status**

```bash
# Get experiment status
curl http://localhost:8000/experiment/status
```

## How It Works

### Thompson Sampling Algorithm

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem:

1. Each ad variant is modeled with a Beta distribution (Beta(α, β))
2. Initially, all ads start with α=1, β=1 (uniform distribution)
3. When an ad is clicked, its α parameter increments by 1
4. When an ad is not clicked, its β parameter increments by 1
5. To select an ad, we:
   - Sample a random value from each ad's Beta distribution
   - Select the ad with the highest sampled value
6. Over time, the distribution for better-performing ads shifts to the right, increasing their selection probability

### Monte Carlo Simulation

This service uses Monte Carlo simulation to determine the probability of each ad being the best variant:

1. Samples are drawn from each ad's Beta distribution thousands of times
2. For each simulation, the ad with the highest sampled value is considered "best"
3. The probability is calculated as: (times an ad was best) / (total simulations)
4. When an ad's probability exceeds the confidence threshold (default 95%), it can be declared the winner

## API Reference

### Advertisements

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ads` | POST | Create a new advertisement |
| `/ads` | GET | Get all advertisements with statistics |
| `/ads/{ad_id}` | GET | Get a specific advertisement |
| `/ads/{ad_id}` | DELETE | Delete an advertisement |
| `/ads/select` | GET | Select an advertisement using Thompson Sampling |
| `/ads/click` | POST | Record a click for an advertisement |
| `/ads/impression/{ad_id}` | POST | Manually record an impression |
| `/ads/no-click/{ad_id}` | POST | Record a no-click event |

### Experiment Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/experiment/config` | PUT | Update experiment configuration |
| `/experiment/status` | GET | Get experiment status |
| `/experiment/winner` | GET | Get winning advertisement if available |

### Miscellaneous

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/ads` | DELETE | Reset all advertisements (testing only) |

## Simulation Tool

The service includes a comprehensive simulation tool to test performance and visualize results:

```bash
# Run a basic simulation
python simulation.py --base-url http://localhost:8000 --total-requests 10000

# Run with custom parameters
python simulation.py --base-url http://localhost:8000 \
  --total-requests 50000 \
  --min-samples 2000 \
  --confidence-threshold 0.98 \
  --output "my_simulation_results.png"

# Use a custom configuration file
python simulation.py --config my_simulation_config.json
```

The simulation tool generates detailed visualizations showing:
- Cumulative impressions by ad
- Rolling CTR with confidence intervals
- CTR convergence over time
- Ad selection distribution

## Contributing
Author: [Timofey Tkachenko](https://linktr.ee/timofey_tkachenko)