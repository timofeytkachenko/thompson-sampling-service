# Thompson Sampling Ad Service

A high-performance microservice for optimizing advertisement selection using Thompson Sampling - a Bayesian approach to the multi-armed bandit problem. This service dynamically allocates traffic to better-performing ad variants based on real-time click-through rates and includes a warmup phase to ensure fair initial exposure.

## Features

- **Bayesian A/B Testing**: Uses Thompson Sampling to efficiently balance exploration and exploitation of ad variants
- **Warmup Phase**: Ensures each ad variant receives minimum exposure before optimization begins
- **Real-time Optimization**: Continuously learns and adapts traffic allocation based on user interactions
- **Early Experiment Termination**: Automatically detects winning variants with statistical confidence
- **Winner Respect**: Option to always serve the winning variant once determined
- **RESTful API**: Comprehensive endpoints for managing experiments and collecting metrics
- **Admin Panel**: Streamlit-based web interface for experiment visualization and management
- **Advanced Simulation Tool**: Includes tools for traffic simulation and visualization with detailed phase analysis
- **Redis Backend**: Fast, in-memory storage for high throughput

## Technology Stack

- **Python 3.11+**
- **FastAPI**: Modern, high-performance web framework
- **Redis Async**: In-memory data storage with async support
- **NumPy**: For statistical computations
- **Uvicorn**: ASGI server
- **Streamlit**: For the admin dashboard UI
- **Plotly & Pandas**: For interactive data visualization
- **Matplotlib & Seaborn**: For data visualization in the simulation tool

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone git@github.com:timofeytkachenko/thompson-sampling-service.git
cd thompson-sampling-service

# Build and start the services
docker-compose up -d
```

With this setup:
- FastAPI service is available at http://localhost:8000
- Admin panel is available at http://localhost:8501

The service uses Docker Compose with the following configuration:

#### Docker Compose Services

- **api**: The main Thompson Sampling service
  - Exposes port 8000 for the API
  - Connects to the Redis instance
  - Includes health checks for reliability
  - Environment variables:
    - `REDIS_URL`: Connection string for Redis
    - `LOG_LEVEL`: Logging verbosity level

- **admin**: The Streamlit admin dashboard
  - Exposes port 8501 for the web interface
  - Connects to the API service
  - Includes health checks
  - Environment variables:
    - `API_BASE_URL`: URL to connect to the API service

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
  -d '{"min_samples": 1000, "confidence_threshold": 0.95, "simulation_count": 10000, "warmup_impressions": 100}'
```

3. **Select Advertisements**

```bash
# Select an ad (Thompson Sampling with warmup phase)
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

### Thompson Sampling with Warmup Phase

The service implements Thompson Sampling with an initial warmup phase to ensure fair evaluation:

1. **Warmup Phase**: Each ad receives a minimum number of impressions (configurable) to build initial statistics
2. **Thompson Sampling Algorithm**:
   - Each ad variant is modeled with a Beta distribution (Beta(α, β))
   - Initially, all ads start with α=1, β=1 (uniform distribution)
   - When an ad is clicked, its α parameter increments by 1
   - When an ad is not clicked, its β parameter increments by 1
   - To select an ad, we:
     - Sample a random value from each ad's Beta distribution
     - Select the ad with the highest sampled value
3. **Winner Detection**: When an ad's probability of being best exceeds the confidence threshold, it can be declared the winner
4. **Winner Respect**: If configured, only the winning ad will be served after it's determined

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

| Endpoint | Method | Description                                |
|----------|--------|--------------------------------------------|
| `/experiment/config` | PUT    | Update experiment configuration            |
| `/experiment/config` | GET    | Get experiment configuration               |
| `/experiment/status` | GET    | Get experiment status                      |
| `/experiment/winner` | GET    | Get winning advertisement if available     |
| `/experiment/winner/reset` | DELETE | Reset the stored experiment winner         |
| `/experiment/warmup/status` | GET    | Get the current status of the warmup phase |

### Miscellaneous

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/ads` | DELETE | Reset all advertisements (testing only) |

## Admin Panel

The service includes a Streamlit-based admin panel for visualizing and managing ad experiments through an intuitive web interface.

### Admin Panel Features

- **Real-time Monitoring**: View current experiment status, warmup phase, and ad performance
- **Interactive Visualizations**: See ad performance metrics with rich, interactive charts:
  - Impression distribution
  - Click-through rate comparison
  - Probability of being the best ad
  - Beta distribution parameters visualization
- **Ad Management**: Create new ads and delete existing ones through the UI
- **Experiment Configuration**: Adjust experiment parameters like minimum samples, confidence threshold, and warmup impressions
- **Reset Options**: Reset the experiment winner and all ads for starting fresh experiments

### Admin Panel Access

The admin panel runs as a separate service within the Docker Compose setup:

- **URL**: http://localhost:8501
- **Running Process**: The Streamlit server runs in a dedicated container named `thompson-admin-panel`
- **Auto-startup**: The admin panel automatically starts when you run `docker-compose up -d`
- **Connection to API**: The admin panel connects to the API service at http://api:8000 within the Docker network
- **No Additional Setup Required**: The admin panel is ready to use as soon as the containers are running

### Accessing the Admin Panel

With the Docker Compose setup, the admin panel is automatically started and available at http://localhost:8501.

### Admin Panel UI

The admin panel provides an intuitive dashboard with:
- Experiment status overview with key metrics
- Ad performance statistics and visualizations
- Warmup phase monitoring with progress bars
- Controls for managing ads and experiment configuration

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
  --warmup 500 \
  --output-prefix "my_simulation_results" \
  --respect-winner true

# Use a custom configuration file
python simulation.py --config my_simulation_config.json
```

### Simulation Features

The enhanced simulator provides detailed analysis and visualizations of the experiment:

- **Phase Analysis**: Tracks warmup, testing, and post-winner phases
- **Selection Method Distribution**: Monitors how ads are selected throughout the experiment
- **Multiple Visualizations**:
  - Cumulative impressions by ad
  - Rolling CTR with confidence intervals
  - CTR convergence over time
  - Ad selection distribution
  - Probability evolution charts
  - Experiment status tracking
- **Detailed Reports**: Generates comprehensive logs and statistics
- **File Outputs**: Saves charts, raw data, and summary information

All results are saved to the `my_results` directory with subdirectories for logs, charts, and data.

### Simulation Configuration

You can configure the simulation with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--base-url` | URL of the Thompson Sampling service | http://localhost:8000 |
| `--total-requests` | Total number of requests to simulate | 10000 |
| `--min-samples` | Minimum samples before stopping | 1000 |
| `--confidence-threshold` | Confidence threshold for winner | 0.95 |
| `--warmup` | Impressions per ad during warmup | 500 |
| `--respect-winner` | Whether to show only winner after determination | true |
| `--ads` | Number of test ads to create (3-5) | 3 |
| `--check-interval` | Frequency of status checks | 100 |
| `--output-prefix` | Prefix for output files | thompson |
| `--no-visualization` | Disable visualization generation | false |
| `--config` | Path to JSON configuration file | None |

## Contributing
Author: [Timofey Tkachenko](https://linktr.ee/timofey_tkachenko)
