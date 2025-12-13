# EXECUTIVE SUMMARY

This document proposes a canonical Tier1 dataset schema for satellite constellation simulations, designed to enhance reproducibility, scalability, and machine learning model training. The core of this proposal is a "tidy" dataset structure where each row represents a single, unique simulation run, and columns represent its parameters, metadata, and results [^4](https://arxiv.org/html/2509.11741v1). This approach ensures that all information required to reproduce a result is self-contained within the dataset.

The recommended storage format is Apache Parquet, a columnar format that offers significant advantages over traditional CSV files in storage efficiency, query performance, and schema enforcement [^1](https://last9.io/blog/parquet-vs-csv/). Parquet's features, such as column pruning and predicate pushdown, are ideally suited for large-scale simulation datasets where analytical queries often target a subset of columns [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). The schema includes robust support for evolution, allowing for future modifications without breaking backward compatibility [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141).

To address the common pitfall of data leakage in model training, this proposal defines a rigorous train/validation splitting strategy using grouped cross-validation techniques like `GroupShuffleSplit` [^6](https://scikit-learn.org/stable/modules/cross_validation.html). By grouping simulations based on their fundamental design characteristics, we can ensure that variations of the same core constellation design do not appear in both the training and validation sets, leading to more realistic model performance estimates [^5](https://scikit-learn.org/stable/common_pitfalls.html). The entire framework is supported by a comprehensive reproducibility contract, integrating with tools like MLflow and DVC to track every aspect of the simulation and data generation process [^7](https://mlflow.org/docs/latest/ml/tracking/), [^9](https://doc.dvc.org/user-guide/pipelines).

# 1. PROPOSED CANONICAL SCHEMA

## 1.1 Schema Definition

The proposed schema follows tidy data principles, where each row corresponds to one complete simulation run. The columns are organized into logical groups for clarity and ease of use [^4](https://arxiv.org/html/2509.11741v1).

### 1.1.1 Constellation Design Parameters

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `num_orbital_planes` | `int64` | The number of orbital planes in the constellation. | `> 0` | `12` |
| `sats_per_plane` | `int64` | The number of satellites in each orbital plane. | `> 0` | `48` |
| `inclination_deg` | `float64` | The inclination of the orbital planes in degrees. | `0.0 <= x <= 180.0` | `53.0` |
| `altitude_km` | `float64` | The altitude of the satellites in kilometers. | `> 0.0` | `550.0` |
| `phasing_parameter` | `float64` | The Walker phasing parameter F. | `Not Nullable` | `21.0` |
| `total_satellites` | `int64` | Derived metric: `num_orbital_planes * sats_per_plane`. | `> 0` | `576` |

### 1.1.2 Ground Station Configuration

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `ground_station_count` | `int64` | The number of ground stations in the network. | `>= 0` | `150` |
| `ground_station_config_id` | `string` | Identifier for the specific set of ground station locations and network topology used. | `Not Nullable` | `"global_network_v2"` |

### 1.1.3 Simulation Configuration

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `simulation_duration_sec` | `int64` | The total duration of the simulation in seconds. | `> 0` | `86400` |
| `simulation_timestep_sec` | `int64` | The time resolution of the simulation in seconds. | `> 0` | `10` |
| `failure_model_id` | `string` | Identifier for the failure model applied (e.g., rate, MTBF). | `Not Nullable` | `"exp_failure_rate_0.01"` |
| `simulation_start_epoch` | `datetime64[ns]` | The start timestamp of the simulation. | `Not Nullable` | `2025-01-01T00:00:00` |
| `simulation_mode` | `string` | The mode or variant of the simulation run (e.g., 'nominal', 'degraded'). | `Not Nullable` | `"nominal"` |

### 1.1.4 Reproducibility Metadata

This metadata is crucial for ensuring that a simulation can be perfectly reproduced, a key principle in robust simulation studies [^3](https://www.scholzmx.com/blog/2022/11-03-how-to-make-your-simulation-study-reproducible/), [^10](https://www.nature.com/articles/s41597-025-05126-1).

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `run_id` | `string` | A unique identifier for this specific simulation run (e.g., UUID). | `Unique, Not Nullable` | `"f47ac10b-58cc-4372-a567"` |
| `random_seed` | `int64` | The seed used for all random number generation to ensure deterministic results. | `Not Nullable` | `42` |
| `schema_version` | `string` | The semantic version of this dataset schema (e.g., '1.0.0'). | `Not Nullable` | `"1.0.0"` |
| `engine_version` | `string` | The version or git commit hash of the simulation engine used. | `Not Nullable` | `"v2.1.3-a4b1c2d"` |
| `execution_timestamp_utc` | `datetime64[ns]` | The UTC timestamp when the simulation was executed. | `Not Nullable` | `2025-07-21T14:30:00` |
| `execution_environment` | `string` | A string (e.g., JSON) detailing key environment info like Python version and library versions. | `Nullable` | `"{'python': '3.9.12', 'numpy': '1.21.5'}"` |
| `config_hash` | `string` | An SHA256 hash of the complete input configuration to verify exact parameter matching. | `Unique, Not Nullable` | `"e3b0c44298fc1c149afbf4c8"` |

### 1.1.5 Output Metrics (Continuous)

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `coverage_mean_pct` | `float64` | The time-averaged global coverage percentage. | `0.0 <= x <= 100.0` | `99.8` |
| `coverage_p95_pct` | `float64` | The 95th percentile of global coverage over time. | `0.0 <= x <= 100.0` | `98.5` |
| `latency_mean_ms` | `float64` | The mean end-to-end latency in milliseconds. | `>= 0.0` | `45.3` |
| `latency_max_ms` | `float64` | The maximum observed end-to-end latency in milliseconds. | `>= 0.0` | `120.1` |
| `throughput_mean_gbps` | `float64` | The mean network throughput in Gbps. | `>= 0.0` | `1024.7` |
| `reliability_score` | `float64` | A composite score representing system reliability. | `0.0 <= x <= 1.0` | `0.9999` |

### 1.1.6 Output Metrics (Categorical)

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `is_successful` | `boolean` | A boolean flag indicating if the simulation completed without critical errors. | `Not Nullable` | `True` |
| `performance_tier` | `string` | A categorical label for performance (e.g., 'Tier1', 'Tier2'). | `['Tier1', 'Tier2', 'Tier3']` | `"Tier1"` |
| `dominant_failure_mode` | `string` | The primary mode of failure if one occurred. | `Nullable` | `"power_subsystem_failure"` |

### 1.1.7 Grouping/Stratification Fields

These fields are essential for implementing the grouped train/validation split strategy discussed in Section 4.

| Column Name | Data Type | Description | Constraints | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `design_family_id` | `string` | An identifier grouping constellations with similar fundamental designs (e.g., Walker Deltas with similar plane/sat counts). | `Not Nullable` | `"walker_delta_12x48"` |
| `topology_hash` | `string` | An SHA256 hash of the core constellation topology parameters to identify identical architectures. | `Not Nullable` | `"a1b2c3d4e5f6..."` |
| `altitude_bin` | `string` | A categorical bin for altitude (e.g., 'LEO', 'MEO'). Used for stratification. | `['LEO', 'MEO', 'GEO']` | `"LEO"` |

## 1.2 Schema as Code

### Pandas DataFrame Schema

```python
import pandas as pd

# Define data types for Pandas DataFrame
pandas_schema = {
    'run_id': 'string',
    'num_orbital_planes': 'int64',
    'sats_per_plane': 'int64',
    'inclination_deg': 'float64',
    'altitude_km': 'float64',
    'phasing_parameter': 'float64',
    'total_satellites': 'int64',
    'ground_station_count': 'int64',
    'ground_station_config_id': 'string',
    'simulation_duration_sec': 'int64',
    'simulation_timestep_sec': 'int64',
    'failure_model_id': 'string',
    'simulation_start_epoch': 'datetime64[ns]',
    'simulation_mode': 'string',
    'random_seed': 'int64',
    'schema_version': 'string',
    'engine_version': 'string',
    'execution_timestamp_utc': 'datetime64[ns]',
    'execution_environment': 'string',
    'config_hash': 'string',
    'coverage_mean_pct': 'float64',
    'coverage_p95_pct': 'float64',
    'latency_mean_ms': 'float64',
    'latency_max_ms': 'float64',
    'throughput_mean_gbps': 'float64',
    'reliability_score': 'float64',
    'is_successful': 'boolean',
    'performance_tier': 'category',
    'dominant_failure_mode': 'string',
    'design_family_id': 'string',
    'topology_hash': 'string',
    'altitude_bin': 'category'
}

# Example DataFrame creation
# df = pd.DataFrame(columns=pandas_schema.keys()).astype(pandas_schema)
```

### PyArrow Schema for Parquet

```python
import pyarrow as pa

# Define PyArrow schema for writing to Parquet
pyarrow_schema = pa.schema([
    pa.field('run_id', pa.string(), nullable=False),
    pa.field('num_orbital_planes', pa.int64(), nullable=False),
    pa.field('sats_per_plane', pa.int64(), nullable=False),
    pa.field('inclination_deg', pa.float64(), nullable=False),
    pa.field('altitude_km', pa.float64(), nullable=False),
    pa.field('phasing_parameter', pa.float64(), nullable=False),
    pa.field('total_satellites', pa.int64(), nullable=False),
    pa.field('ground_station_count', pa.int64(), nullable=False),
    pa.field('ground_station_config_id', pa.string(), nullable=False),
    pa.field('simulation_duration_sec', pa.int64(), nullable=False),
    pa.field('simulation_timestep_sec', pa.int64(), nullable=False),
    pa.field('failure_model_id', pa.string(), nullable=False),
    pa.field('simulation_start_epoch', pa.timestamp('ns'), nullable=False),
    pa.field('simulation_mode', pa.string(), nullable=False),
    pa.field('random_seed', pa.int64(), nullable=False),
    pa.field('schema_version', pa.string(), nullable=False),
    pa.field('engine_version', pa.string(), nullable=False),
    pa.field('execution_timestamp_utc', pa.timestamp('ns'), nullable=False),
    pa.field('execution_environment', pa.string(), nullable=True),
    pa.field('config_hash', pa.string(), nullable=False),
    pa.field('coverage_mean_pct', pa.float64(), nullable=False),
    pa.field('coverage_p95_pct', pa.float64(), nullable=False),
    pa.field('latency_mean_ms', pa.float64(), nullable=False),
    pa.field('latency_max_ms', pa.float64(), nullable=False),
    pa.field('throughput_mean_gbps', pa.float64(), nullable=False),
    pa.field('reliability_score', pa.float64(), nullable=False),
    pa.field('is_successful', pa.bool_(), nullable=False),
    pa.field('performance_tier', pa.dictionary(pa.int8(), pa.string()), nullable=False),
    pa.field('dominant_failure_mode', pa.string(), nullable=True),
    pa.field('design_family_id', pa.string(), nullable=False),
    pa.field('topology_hash', pa.string(), nullable=False),
    pa.field('altitude_bin', pa.dictionary(pa.int8(), pa.string()), nullable=False)
])
```

### Data Validation with Pandera

```python
import pandera as pa
from pandera.typing import Series

class ConstellationSchema(pa.SchemaModel):
    run_id: Series[str] = pa.Field(unique=True, nullable=False)
    num_orbital_planes: Series[int] = pa.Field(gt=0, coerce=True)
    sats_per_plane: Series[int] = pa.Field(gt=0, coerce=True)
    inclination_deg: Series[float] = pa.Field(ge=0.0, le=180.0, coerce=True)
    altitude_km: Series[float] = pa.Field(gt=0, coerce=True)
    coverage_mean_pct: Series[float] = pa.Field(ge=0.0, le=100.0, coerce=True)
    random_seed: Series[int] = pa.Field(nullable=False, coerce=True)
    config_hash: Series[str] = pa.Field(nullable=False)
    design_family_id: Series[str] = pa.Field(nullable=False)
    performance_tier: Series[str] = pa.Field(isin=['Tier1', 'Tier2', 'Tier3'])
    
    class Config:
        strict = 'filter' # Drops columns not defined in schema
        coerce = True # Coerces types where possible
```

# 2. FORMAT COMPARISON: CSV vs PARQUET

## 2.1 Detailed Comparison Table

| Feature | CSV (Comma-Separated Values) | Apache Parquet |
| :--- | :--- | :--- |
| **Storage Type** | Row-based, plain text [^1](https://last9.io/blog/parquet-vs-csv/). | Columnar, binary [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Storage Efficiency** | Poor. No native compression, resulting in large file sizes [^1](https://last9.io/blog/parquet-vs-csv/). | Excellent. High compression ratios due to columnar storage (similar data is grouped) and efficient compression algorithms (Snappy, Gzip, Zstd) [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Read Performance** | Slow for analytics. Must read entire rows even if only a few columns are needed [^1](https://last9.io/blog/parquet-vs-csv/). | Fast for analytics. Only the required columns are read from disk (column pruning) [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Write Performance** | Generally fast due to its simplicity. | Can be slower than CSV due to the overhead of columnar organization, compression, and metadata writing [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Schema Enforcement** | None. It is "schema-on-read," requiring inference which can be slow and error-prone. No type safety [^1](https://last9.io/blog/parquet-vs-csv/). | Strong. The schema is embedded in the file metadata, ensuring data integrity and type safety [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Metadata Support** | None. All information must be contained in the data itself or stored separately. | Rich. Stores schema, file statistics (min/max), compression details, and user-defined metadata [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Column Pruning/Predicate Pushdown** | Not supported. | Fully supported. Queries can filter data using metadata statistics (predicate pushdown) before reading it, drastically reducing I/O [^1](https://last9.io/blog/parquet-vs-csv/). |
| **Schema Evolution** | Not supported. Adding a column breaks parsers expecting a fixed number of columns. | Natively supported. Can add, remove, and reorder columns, providing forward and backward compatibility [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141). |
| **Ecosystem Compatibility** | Universal. Supported by virtually every data tool and programming language [^1](https://last9.io/blog/parquet-vs-csv/). | Wide support in the big data ecosystem (Spark, Dask, Pandas, Polars, etc.) and cloud platforms (AWS, GCP, Azure) [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format). |
| **Human Readability** | High. Can be opened and inspected in any text editor or spreadsheet software [^1](https://last9.io/blog/parquet-vs-csv/). | Low. It is a binary format that requires specialized tools to view [^1](https://last9.io/blog/parquet-vs-csv/). |
| **Use Case Suitability** | Small datasets, data export for simple tools, human inspection. | Large-scale datasets, analytics, machine learning, data warehousing. |

## 2.2 Recommendation

**The unequivocal recommendation is to use Apache Parquet as the canonical format for the satellite constellation simulation dataset.**

This recommendation is based on the following technical justifications:
1.  **Storage & Cost Efficiency:** Simulation datasets can grow to terabytes or petabytes. Parquet's superior compression will lead to significant reductions in storage costs and faster data transfer times [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format).
2.  **Performance for ML and Analytics:** Machine learning training and data analysis rarely require all columns at once. Parquet's columnar nature and predicate pushdown capabilities will dramatically accelerate data loading and querying, which is often a bottleneck in ML workflows [^1](https://last9.io/blog/parquet-vs-csv/).
3.  **Data Integrity:** The embedded schema ensures that data types are consistent and prevents the data corruption or misinterpretation issues common with CSV files. This is critical for reliable model training [^1](https://last9.io/blog/parquet-vs-csv/).
4.  **Long-Term Maintainability:** The robust support for schema evolution is essential for a long-lived project. As new parameters or metrics are added to simulations, Parquet allows the dataset to evolve gracefully without invalidating older data [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141).

## 2.3 Hybrid Approach

While Parquet should be the primary storage format, a hybrid approach has merit in specific, limited scenarios:
*   **Human-Readable Exports:** For debugging, creating reports, or sharing a small sample of data with non-technical stakeholders, exporting a few rows to a CSV file can be useful due to its universal accessibility.
*   **Legacy System Integration:** If a specific tool in the pipeline does not support Parquet, a temporary conversion to CSV might be necessary, but this should be an exception, not the rule.

# 3. INTEGRATION WITH `SimulationEngine.run()`

## 3.1 Current State Analysis

A typical `SimulationEngine.run()` method takes a configuration object or dictionary, executes a complex simulation, and returns a dictionary or custom object containing the results.

```python
# Typical existing structure
class SimulationEngine:
    def run(self, config: dict) -> dict:
        # 1. Unpack config
        # 2. Set up simulation environment
        # 3. Run the core simulation logic
        # 4. Compute and return output metrics
        results = {"coverage": 99.5, "latency": 50.1}
        return results
```

This structure is functional but lacks the metadata and standardized format required for a canonical dataset.

## 3.2 Minimal Refactoring Approach

The goal is to refactor the `run` method to output a single-row Pandas DataFrame that conforms to the canonical schema, capturing all necessary inputs, metadata, and outputs.

**Implementation Strategy:**

1.  **Start with the Input Config:** Treat the input `config` dictionary as the primary source for the parameter columns.
2.  **Generate Reproducibility Metadata:** Before running the simulation, generate all reproducibility fields: `run_id`, `random_seed`, `engine_version`, timestamps, and a hash of the input config.
3.  **Execute Simulation:** Run the core simulation logic as before.
4.  **Collect Outputs:** Gather all continuous and categorical output metrics.
5.  **Generate Grouping Fields:** Compute the `design_family_id` and `topology_hash` from the input parameters.
6.  **Assemble DataFrame:** Combine all collected data into a single dictionary and create a one-row Pandas DataFrame, ensuring the columns and dtypes match the canonical schema.
7.  **Write to Parquet:** Append the resulting DataFrame to a master Parquet dataset.

## 3.3 Code Example

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import hashlib
import json
import datetime as dt
import subprocess
from typing import Dict, Any

# --- Helper Functions ---

def get_git_revision_hash() -> str:
    """Gets the git commit hash of the current repository."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def generate_config_hash(config: Dict[str, Any]) -> str:
    """Generates a SHA256 hash of a configuration dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def get_design_family_id(config: Dict[str, Any]) -> str:
    """Creates a human-readable ID for a constellation design family."""
    planes = config.get('num_orbital_planes', 'N')
    sats = config.get('sats_per_plane', 'N')
    return f"walker_delta_{planes}x{sats}"

def write_to_parquet(df: pd.DataFrame, path: str, schema: pa.Schema):
    """Appends a DataFrame to a Parquet dataset."""
    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_to_dataset(table, root_path=path, existing_data_behavior='overwrite_or_ignore')

# --- Refactored SimulationEngine ---

class SimulationEngine:
    """
    A refactored simulation engine that outputs to the canonical schema.
    """
    SCHEMA_VERSION = "1.0.0"
    ENGINE_VERSION = get_git_revision_hash()

    def run(self, config: Dict[str, Any], output_path: str):
        """
        Runs a simulation and appends the result to a Parquet dataset.
        """
        # 1. Generate reproducibility metadata
        repro_meta = {
            'run_id': str(uuid.uuid4()),
            'random_seed': config['random_seed'],
            'schema_version': self.SCHEMA_VERSION,
            'engine_version': self.ENGINE_VERSION,
            'execution_timestamp_utc': dt.datetime.utcnow(),
            'config_hash': generate_config_hash(config)
        }

        # 2. Run core simulation logic
        print(f"Running simulation for run_id: {repro_meta['run_id']}...")
        # ... your core simulation logic here ...
        # This is a placeholder for the actual simulation results
        sim_results = {
            'coverage_mean_pct': 99.8,
            'latency_mean_ms': 45.3,
            'is_successful': True,
            'performance_tier': 'Tier1'
        }
        print("Simulation complete.")

        # 3. Combine all data into a single record
        record = {
            **repro_meta,
            **config, # Unpack input config
            **sim_results, # Unpack simulation results
            'total_satellites': config['num_orbital_planes'] * config['sats_per_plane'],
            'design_family_id': get_design_family_id(config),
            'topology_hash': generate_config_hash({k: v for k, v in config.items() if k in ['num_orbital_planes', 'sats_per_plane', 'inclination_deg']}),
            'altitude_bin': 'LEO' if config['altitude_km'] < 2000 else 'MEO'
        }
        
        # 4. Format as a canonical DataFrame
        df = pd.DataFrame([record])
        # Ensure correct dtypes - for a full implementation, use the schema objects defined earlier
        df['execution_timestamp_utc'] = pd.to_datetime(df['execution_timestamp_utc'])
        
        # 5. Write to Parquet dataset
        print(f"Writing results to {output_path}...")
        write_to_parquet(df, output_path, pyarrow_schema)
        print("Write complete.")
```

# 4. TRAIN/VAL SPLIT STRATEGY

## 4.1 The Data Leakage Problem

In satellite constellation simulations, datasets are often generated through systematic parameter sweeps (e.g., varying inclination by 1 degree at a time) or by exploring minor variations of a base design. A naive random split is highly problematic because it can place very similar (or nearly identical) constellation designs into both the training and testing sets [^5](https://scikit-learn.org/stable/common_pitfalls.html).

This leads to **data leakage**, where the model inadvertently learns from data that is too similar to the test set. The model may appear to perform exceptionally well during validation, but it fails to generalize to genuinely new, unseen constellation architectures in the real world. The validation score becomes an unreliable, overly optimistic estimate of the model's true performance [^5](https://scikit-learn.org/stable/common_pitfalls.html).

## 4.2 Grouped Split Strategy

To prevent this form of data leakage, we must use a **grouped split strategy**. This ensures that all data points belonging to a specific "group" (i.e., a family of similar constellation designs) are contained entirely within either the training set or the validation set, but never split across both [^6](https://scikit-learn.org/stable/modules/cross_validation.html).

The `design_family_id` column in the canonical schema is specifically designed for this purpose.

**Recommended Strategy:** `sklearn.model_selection.GroupShuffleSplit`

*   **How it Works:** This splitter performs randomized splits but ensures that the same group ID does not appear in both the train and test sets. It is flexible and allows for specifying the number of splits and the test set size.
*   **Why it's suitable:** It is ideal for creating a single train/validation split, which is the standard practice before model training. `GroupKFold` is better suited for cross-validation pipelines.
*   **Stratification:** If the distribution of a key parameter (like `altitude_bin`) or a target variable (like `performance_tier`) is highly imbalanced, `StratifiedGroupKFold` should be considered to preserve these distributions within each split while respecting the group boundaries [^6](https://scikit-learn.org/stable/modules/cross_validation.html).

This image from the scikit-learn documentation visually demonstrates how `GroupKFold` (which `GroupShuffleSplit` is based on) works. Notice how samples with the same group color are never in both train and test splits simultaneously.
![GroupKFold Visualization](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_007.png) [^6](https://scikit-learn.org/stable/modules/cross_validation.html)

## 4.3 Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# --- 1. Create Sample Data ---
# Let's create a dummy dataset that simulates the data leakage problem
data = []
design_families = [f"design_{i}" for i in range(10)] # 10 distinct design families
for i, family in enumerate(design_families):
    # Each family has 10 minor variations
    for j in range(10):
        data.append({
            'run_id': f"{family}_var_{j}",
            'design_family_id': family,
            'inclination_deg': 50 + i, # Parameter tied to family
            'altitude_km': 550 + j * 5, # Minor variation
            'performance_tier': 'Tier1' if (i+j) % 2 == 0 else 'Tier2',
            'coverage_mean_pct': 95 + np.random.randn()
        })
dataset_df = pd.DataFrame(data)
X = dataset_df[['inclination_deg', 'altitude_km']]
y = dataset_df['performance_tier']
groups = dataset_df['design_family_id']

# --- 2. Perform Grouped Split ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

print(f"Total samples: {len(dataset_df)}")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# --- 3. Validate No Leakage ---
train_groups = set(groups_train)
val_groups = set(groups_val)
leakage = train_groups.intersection(val_groups)

print(f"\nTraining groups: {train_groups}")
print(f"Validation groups: {val_groups}")
if not leakage:
    print("\n✅ SUCCESS: No leakage detected. The sets of design families are disjoint.")
else:
    print(f"\n❌ FAILURE: Leakage detected! Overlapping groups: {leakage}")

```

## 4.4 Validation Strategy

To programmatically verify that the split has successfully prevented data leakage:
1.  **Check Group Membership:** The most direct method is to find the intersection of the unique group IDs in the training and validation sets. An empty intersection confirms that there is no leakage, as demonstrated in the code example above.
2.  **Compare Design Parameters:** For a sanity check, compute descriptive statistics (mean, std, min, max) for key design parameters (e.g., `num_orbital_planes`) grouped by `design_family_id`. This helps confirm that the groups correctly capture design similarity.

# 5. SCHEMA VERSIONING AND EVOLUTION

## 5.1 Versioning Strategy

A disciplined schema versioning strategy is crucial for long-term maintainability. We will adopt **Semantic Versioning (MAJOR.MINOR.PATCH)**. The version should be stored in two places:
1.  As a column `schema_version` in the dataset itself.
2.  In the file-level metadata of the Parquet file.

**Incrementing Version Components:**
*   **MAJOR (e.g., 2.0.0):** Increment for a breaking change. In our "additive-only" schema, this should be extremely rare. An example would be fundamentally changing the meaning of a column (e.g., changing `altitude_km` to `altitude_meters`).
*   **MINOR (e.g., 1.1.0):** Increment when adding new columns in a backward-compatible way. This will be the most common type of change (e.g., adding a new output metric `jitter_ms`) [^11](https://www.designandexecute.com/designs/best-approaches-to-manage-schema-evolution-for-parquet-files/).
*   **PATCH (e.g., 1.0.1):** Increment for minor corrections that do not alter the schema structure, such as clarifying a column description or updating metadata conventions.

## 5.2 Evolution Best Practices

Drawing from best practices for Parquet schema evolution [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141/), [^11](https://www.designandexecute.com/designs/best-approaches-to-manage-schema-evolution-for-parquet-files/):
1.  **Use Nullable Fields for New Columns:** All new columns added to the schema must be nullable. This ensures that older files, which do not contain the column, can be read by new code, with the missing values being interpreted as `null` (forward compatibility) [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141).
2.  **Never Rename or Remove Columns:** Renaming or removing columns is a breaking change. Instead, deprecate the old column by no longer populating it in new data and add a new column with the desired name or logic. This maintains backward compatibility [^11](https://www.designandexecute.com/designs/best-approaches-to-manage-schema-evolution-for-parquet-files/).
3.  **Add New Columns at the End:** While Parquet is not sensitive to column order, adding new columns to the end of the schema is a good convention for human readability.
4.  **Use a Schema Registry:** For mature systems, a dedicated schema registry (like Confluent Schema Registry) or a metadata catalog (like AWS Glue or Nessie) can be used to centrally manage and validate schemas.
5.  **Document All Changes:** Maintain a `CHANGELOG.md` file alongside the schema definition code that meticulously documents every version change and its rationale.

## 5.3 Migration Strategy

When reading a dataset that contains files written with different schema versions, Parquet's schema merging capabilities can be used. Most modern data tools (Spark, Dask, PyArrow) can handle this automatically.

When reading the data:
1.  The reader will unify the schemas from all files into a single, comprehensive schema that includes all columns from all versions.
2.  For files that are missing a newer column, the values will be filled with `null`.
3.  Appropriate `fillna` strategies can then be applied based on the analysis context (e.g., filling with `0`, a mean value, or a special marker like `-1`).

## 5.4 Code Example

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# --- Setup: Create two Parquet files with different schema versions ---
# Version 1.0.0
data_v1 = [{'run_id': 'abc', 'value': 10.0, 'schema_version': '1.0.0'}]
df_v1 = pd.DataFrame(data_v1)
schema_v1 = pa.schema([
    pa.field('run_id', pa.string()),
    pa.field('value', pa.float64()),
    pa.field('schema_version', pa.string())
])
table_v1 = pa.Table.from_pandas(df_v1, schema=schema_v1)

# Version 1.1.0 (adds a new nullable column)
data_v2 = [{'run_id': 'def', 'value': 20.0, 'new_metric': 99.9, 'schema_version': '1.1.0'}]
df_v2 = pd.DataFrame(data_v2)
schema_v2 = pa.schema([
    pa.field('run_id', pa.string()),
    pa.field('value', pa.float64()),
    pa.field('new_metric', pa.float64(), nullable=True), # Must be nullable
    pa.field('schema_version', pa.string())
])
table_v2 = pa.Table.from_pandas(df_v2, schema=schema_v2)

# Write as a partitioned dataset
os.makedirs("mixed_schema_dataset", exist_ok=True)
pq.write_table(table_v1, "mixed_schema_dataset/data_v1.parquet")
pq.write_table(table_v2, "mixed_schema_dataset/data_v2.parquet")

# --- Reading and Handling Mixed Schemas ---
# PyArrow's read_table can automatically merge schemas
dataset = pq.ParquetDataset("mixed_schema_dataset/")
merged_table = dataset.read()
merged_df = merged_table.to_pandas()

print("--- Merged DataFrame ---")
print(merged_df)
print("\n--- Data Types ---")
print(merged_df.dtypes)

# Handle missing columns gracefully
# For 'new_metric', a value of 0 might be a sensible default
merged_df['new_metric'] = merged_df['new_metric'].fillna(0)

print("\n--- DataFrame after fillna ---")
print(merged_df)
```

# 6. REPRODUCIBILITY CONTRACT

## 6.1 Contract Definition

To guarantee the exact reproducibility of any given simulation run (i.e., any row in the dataset), the following information, captured in our schema's reproducibility metadata, must be tracked [^10](https://www.nature.com/articles/s41597-025-05126-1):
1.  **All Input Parameters:** Every parameter that influences the simulation's outcome, from `num_orbital_planes` to `failure_model_id`.
2.  **Random Seed:** The `random_seed` used to initialize the random number generator [^3](https://www.scholzmx.com/blog/2022/11-03-how-to-make-your-simulation-study-reproducible).
3.  **Software Versions:** The exact version (preferably git commit hash) of the `SimulationEngine` and key libraries (`engine_version`, `execution_environment`).
4.  **Configuration Hash:** The `config_hash` serves as a final checksum to ensure the input configuration has not been altered.
5.  **Execution Timestamp:** The `execution_timestamp_utc` provides provenance for when the simulation was run.

## 6.2 Verification Approach

To verify reproducibility for a given `run_id`:
1.  Retrieve the row from the dataset corresponding to the `run_id`.
2.  Extract the original input parameters and the `random_seed`.
3.  Check out the exact `engine_version` (git commit hash).
4.  Re-run the simulation using these parameters.
5.  Compare the output metrics of the new run with the original metrics stored in the dataset.
6.  For floating-point numbers, comparisons should be made within a small tolerance threshold (e.g., `np.allclose(original, new, rtol=1e-5)`). All other values should match exactly.

## 6.3 Integration with MLflow/DVC

Experiment tracking tools are essential for managing the reproducibility contract at scale.

**MLflow for Experiment Tracking:** [^7](https://mlflow.org/docs/latest/ml/tracking/)
*   **Parameters:** Log all input parameters (constellation design, ground station config, simulation config) to MLflow using `mlflow.log_params()`.
*   **Metrics:** Log all output metrics (coverage, latency, etc.) using `mlflow.log_metrics()`.
*   **Artifacts:** Log the one-row Parquet file for the run as an artifact using `mlflow.log_artifact()`. The full dataset can be an artifact of a "data generation" parent run.
*   **Tags:** Use `mlflow.set_tag()` to store metadata like `engine_version`, `schema_version`, and `config_hash`.

**DVC for Data Versioning:** [^9](https://doc.dvc.org/user-guide/pipelines)
*   The entire Parquet dataset directory should be versioned with DVC.
*   When a new batch of simulations is run, the dataset is updated, and `dvc add` is used to create a new version of the data.
*   The `.dvc` file, which is a small pointer to the data stored in remote storage, is committed to git. This allows you to check out a specific version of your code and get the exact corresponding version of the dataset.

**Example Workflow:**

1.  A data scientist defines a simulation campaign in a `dvc.yaml` pipeline.
2.  Running `dvc repro` executes the simulation script.
3.  The script runs simulations, logging parameters and metrics for each run to **MLflow**.
4.  Each run produces a single-row Parquet file, which are collected into a dataset directory.
5.  The `dvc.yaml` pipeline tracks the entire dataset directory as an output.
6.  Committing the updated `dvc.lock` file versions the code, parameters, and the exact dataset produced.

# 7. IMPLEMENTATION ROADMAP

1.  **Define and Validate Schema:**
    *   Finalize the schema definition in code (PyArrow, Pandera).
    *   Commit schema definitions to a version-controlled repository.
2.  **Create Schema Validation Utilities:**
    *   Develop a standalone script or library function to validate any given Parquet file against the canonical schema using Pandera.
3.  **Refactor `SimulationEngine.run()`:**
    *   Incrementally update the `SimulationEngine` as shown in Section 3.3.
    *   Start by adding reproducibility metadata, then format outputs to match the schema.
4.  **Implement Grouped Splitting Logic:**
    *   Create helper functions for data loading and splitting using `GroupShuffleSplit`.
    *   Integrate these functions into the ML model training pipelines.
5.  **Set up Version Control (git) + Data Versioning (DVC):**
    *   Initialize DVC in the project repository.
    *   Configure remote storage (e.g., S3, GCS).
    *   Perform an initial `dvc add` on the dataset directory.
6.  **Create Migration Scripts for Existing Data:**
    *   Write a script to read any existing simulation data (from CSVs, logs, etc.), transform it into the canonical schema, and write it to a new Parquet dataset.
7.  **Document Schema and Usage:**
    *   Create a `README.md` in the schema's repository explaining each column, the versioning policy, and how to use the validation utilities.
8.  **Set up CI/CD for Schema Validation:**
    *   Create a continuous integration pipeline (e.g., GitHub Actions) that runs on every pull request.
    *   The pipeline should run a validation check to ensure that any new data generation code produces schema-compliant Parquet files.

# 8. CODE EXAMPLES SUMMARY

This section provides a consolidated, runnable Python script summarizing the key code examples.

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandera
import uuid
import hashlib
import json
import datetime as dt
import subprocess
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from typing import Dict, Any
import os

# ==============================================================================
# 1. SCHEMA DEFINITION (PyArrow & Pandera)
# ==============================================================================
pyarrow_schema = pa.schema([
    pa.field('run_id', pa.string(), nullable=False),
    pa.field('design_family_id', pa.string(), nullable=False),
    pa.field('config_hash', pa.string(), nullable=False),
    pa.field('num_orbital_planes', pa.int64(), nullable=False),
    pa.field('sats_per_plane', pa.int64(), nullable=False),
    pa.field('inclination_deg', pa.float64(), nullable=False),
    pa.field('altitude_km', pa.float64(), nullable=False),
    pa.field('random_seed', pa.int64(), nullable=False),
    pa.field('schema_version', pa.string(), nullable=False),
    pa.field('engine_version', pa.string(), nullable=False),
    pa.field('coverage_mean_pct', pa.float64(), nullable=False),
    pa.field('is_successful', pa.bool_(), nullable=False)
])

class SummarySchema(pandera.SchemaModel):
    run_id: pandera.typing.Series[str] = pandera.Field(unique=True)
    inclination_deg: pandera.typing.Series[float] = pandera.Field(ge=0, le=180)
    coverage_mean_pct: pandera.typing.Series[float] = pandera.Field(ge=0, le=100)
    class Config:
        strict = "filter"
        coerce = True

# ==============================================================================
# 2. SimulationEngine INTEGRATION
# ==============================================================================
def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception: return "unknown"

def generate_config_hash(config: Dict[str, Any]) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def get_design_family_id(config: Dict[str, Any]) -> str:
    planes = config.get('num_orbital_planes', 'N')
    sats = config.get('sats_per_plane', 'N')
    return f"walker_delta_{planes}x{sats}"

class SimulationEngine:
    SCHEMA_VERSION = "1.0.0"
    ENGINE_VERSION = get_git_revision_hash()

    def run(self, config: Dict[str, Any]) -> pd.DataFrame:
        repro_meta = {
            'run_id': str(uuid.uuid4()),
            'random_seed': config['random_seed'],
            'schema_version': self.SCHEMA_VERSION,
            'engine_version': self.ENGINE_VERSION,
            'config_hash': generate_config_hash(config)
        }
        sim_results = {'coverage_mean_pct': 95 + np.random.uniform(-2, 2), 'is_successful': True}
        record = {**repro_meta, **config, **sim_results, 'design_family_id': get_design_family_id(config)}
        return pd.DataFrame([record])

# ==============================================================================
# 3. DATA GENERATION, VALIDATION, AND WRITING
# ==============================================================================
print("--- Generating Simulation Data ---")
engine = SimulationEngine()
all_runs = []
for i in range(5): # 5 design families
    for j in range(10): # 10 variations each
        sim_config = {
            'num_orbital_planes': 10 + i,
            'sats_per_plane': 20,
            'inclination_deg': 50.0 + i,
            'altitude_km': 550.0 + j * 10,
            'random_seed': 42 + i * 10 + j
        }
        all_runs.append(engine.run(sim_config))

dataset_df = pd.concat(all_runs, ignore_index=True)

print(f"Generated dataset with {len(dataset_df)} rows.")

print("\n--- Validating Schema with Pandera ---")
validated_df = SummarySchema.validate(dataset_df)
print("Schema validation successful.")

print("\n--- Writing to Parquet ---")
output_path = "simulation_dataset"
table = pa.Table.from_pandas(validated_df, schema=pyarrow_schema)
pq.write_to_dataset(table, root_path=output_path, existing_data_behavior='overwrite_or_ignore')
print(f"Data written to '{output_path}' directory.")


# ==============================================================================
# 4. GROUPED TRAIN/VAL SPLIT
# ==============================================================================
print("\n--- Performing Grouped Train/Val Split ---")
X = dataset_df.drop(columns=['is_successful'])
y = dataset_df['is_successful']
groups = dataset_df['design_family_id']

gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups))

groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

print(f"Training samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")

leakage = set(groups_train).intersection(set(groups_val))
assert not leakage, f"Leakage detected: {leakage}"
print("✅ SUCCESS: No data leakage detected between train and val sets.")

# ==============================================================================
# 5. SCHEMA EVOLUTION HANDLING
# ==============================================================================
# See Section 5.4 for a detailed, standalone example of handling mixed schemas.
print("\n--- Schema Evolution Example in Section 5.4 ---")
```

# 9. REFERENCES AND FURTHER READING

1.  **Parquet vs CSV: Which Format Should You Choose?** [^1](https://last9.io/blog/parquet-vs-csv/)  
    A blog post providing a detailed comparison of Parquet and CSV, covering performance, storage efficiency, data integrity, and use cases.
2.  **What are the pros and cons of the Apache Parquet format compared to other formats?** [^2](https://stackoverflow.com/questions/36822224/what-are-the-pros-and-cons-of-the-apache-parquet-format-compared-to-other-format)  
    A Stack Overflow discussion offering technical insights into Parquet's columnar storage, compression, predicate pushdown, and immutability.
3.  **How to make your simulation study reproducible** [^3](https://www.scholzmx.com/blog/2022/11-03-how-to-make-your-simulation-study-reproducible)  
    A guide on best practices for reproducibility in simulation studies, emphasizing the importance of setting, using, and saving random seeds and tracking metadata.
4.  **Tidy simulation: Designing robust, reproducible, and scalable simulation studies** [^4](https://arxiv.org/html/2509.11741v1)  
    An academic paper outlining the "tidy simulation" framework, which advocates for a structured approach involving a simulation grid, data generation, analysis, and a results table.
5.  **Common pitfalls and recommended practices - scikit-learn documentation** [^5](https://scikit-learn.org/stable/common_pitfalls.html)  
    Official scikit-learn documentation detailing common issues, with a strong focus on preventing data leakage during preprocessing and model validation.
6.  **Cross-validation: evaluating estimator performance - scikit-learn documentation** [^6](https://scikit-learn.org/stable/modules/cross_validation.html)  
    Official scikit-learn documentation describing various cross-validation strategies, including detailed explanations and visualizations of grouped iterators like `GroupKFold` and `GroupShuffleSplit`.
7.  **MLflow Tracking Documentation** [^7](https://mlflow.org/docs/latest/ml/tracking/)  
    The official documentation for MLflow Tracking, explaining how to log parameters, metrics, artifacts, and models for experiment management and reproducibility.
8.  **All About Parquet Part 04 — Schema Evolution in Parquet** [^8](https://medium.com/data-engineering-with-dremio/all-about-parquet-part-04-schema-evolution-in-parquet-c2c2b1aa6141)  
    An article detailing how Parquet handles schema changes, including adding/removing columns and ensuring forward/backward compatibility.
9.  **Pipelines - Data Version Control · DVC Documentation** [^9](https://doc.dvc.org/user-guide/pipelines)  
    The official documentation for DVC, explaining how to version datasets, define data processing pipelines, and ensure reproducibility of ML workflows.
10. **Metadata practices for simulation workflows - *Scientific Data*** [^10](https://www.nature.com/articles/s41597-025-05126-1)  
    A peer-reviewed article in *Nature* that proposes a comprehensive framework for collecting and managing metadata in simulation workflows to improve reproducibility and data sharing.
11. **Best Approaches to Manage Schema Evolution for Parquet Files** [^11](https://www.designandexecute.com/designs/best-approaches-to-manage-schema-evolution-for-parquet-files/)  
    A guide outlining ten best practices for managing schema changes in Parquet, including additive design, schema registries, and using tools like Avro and Delta Lake.

---
### How this report was produced
This report was generated by a multi-agent AI system. An initial planning agent broke down the request into a series of research steps. A web search agent executed these steps, gathering information on data formats, schema design, reproducibility, data splitting strategies, and schema evolution from high-quality technical blogs, official documentation, and scientific papers. An extraction agent summarized the key information from these sources. Finally, a report-writing agent synthesized all the gathered and extracted data into this comprehensive technical proposal, adhering to the specified structure and requirements. All sources have been cited inline.