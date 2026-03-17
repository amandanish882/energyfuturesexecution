"""Central configuration for the Energy Commodities Trading Platform.

Defines filesystem paths, external API credentials, database connection
parameters, logging settings, and product-level defaults consumed by
every module in the project. API keys are resolved from environment
variables first, then from a ``.env`` file in the project root.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
PLOT_DIR = PROJECT_ROOT / "output" / "plots"

# EIA API
EIA_API_KEY = os.environ.get("EIA_API_KEY", None)

def get_eia_api_key():
    """Retrieve the EIA API key from the environment or a .env file.

    Checks the module-level ``EIA_API_KEY`` constant first (populated
    from the ``EIA_API_KEY`` environment variable at import time). If
    that is falsy, reads the ``.env`` file in the project root and
    parses the first line matching ``EIA_API_KEY=...``, stripping
    surrounding quotes.

    Returns:
        The EIA API key string if found in the environment or ``.env``
        file, otherwise ``None``.
    """
    if EIA_API_KEY:
        return EIA_API_KEY
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("EIA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

# Databento API
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", None)

def get_databento_api_key():
    """Retrieve the Databento API key from the environment or a .env file.

    Checks the module-level ``DATABENTO_API_KEY`` constant first
    (populated from the ``DATABENTO_API_KEY`` environment variable at
    import time). If that is falsy, reads the ``.env`` file in the
    project root and parses the first line matching
    ``DATABENTO_API_KEY=...``, stripping surrounding quotes.

    Returns:
        The Databento API key string if found in the environment or
        ``.env`` file, otherwise ``None``.
    """
    if DATABENTO_API_KEY:
        return DATABENTO_API_KEY
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

# KDB+
KDB_HOST = os.environ.get("KDB_HOST", "localhost")
KDB_PORT = int(os.environ.get("KDB_PORT", "5000"))

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# Defaults
DEFAULT_VALUATION_DATE = "2026-03-09"
BUMP_SIZE_USD = 1.0

# Energy products
ENERGY_PRODUCTS = {
    "CL": "WTI Crude Oil",
    "HO": "Heating Oil",
    "RB": "RBOB Gasoline",
    "NG": "Natural Gas",
}
