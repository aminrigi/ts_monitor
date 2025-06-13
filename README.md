# TSMonitor

**TSMonitor** is a lightweight and flexible Python class for monitoring univariate time series data. It provides functionality for seasonal-trend decomposition, outlier detection, and change point detection using state-of-the-art algorithms.

## Features

- STL & MSTL decomposition (supports multiple seasonalities)
- Residual-based outlier detection via quantile thresholds
- Change point detection using PELT (via `skchange`)
- Interactive and insightful visualizations (STL components, deseasonalized series, change points)
- Robust handling of missing data

## Data Requirements

TSMonitor expects your time series data in a `pandas.DataFrame` with the following structure:

### Required Columns

| Column | Type            | Description                           |
|--------|------------------|---------------------------------------|
| `ds`   | `datetime64[ns]` | Timestamps with **daily frequency**   |
| `y`    | `float` or `int` | Numeric values for the time series    |


## Missing Value Strategy

TSMonitor includes built-in logic to handle missing values in daily time series data based on the use case that I needed it.

### Rules Applied:

1. **Ensure daily frequency**:
   - The class reindexes the series to fill in any missing calendar dates.

2. **Weekends (Saturday & Sunday)**:
   - Missing values are automatically filled with `0`, assuming no activity.

3. **Weekdays (Monday to Friday)**:
   - If a value is missing:
     - If both the same weekday from the **previous and next week** are available, fill with their **average**.
     - If only one is available, use that value.
     - If neither is available, use the **maximum** of the nearest available previous or next values.

4. **High Missing Rate Handling**:
   - If more than **5%** of the data is missing, the system raises a `ValueError` to encourage better data preprocessing.



## Why Use PELT?

[PELT (Pruned Exact Linear Time)](https://arxiv.org/pdf/2003.06222) is a fast and statistically robust algorithm for detecting change points in time series.

### ✅ Benefits of PELT:
- **Linear time complexity**: Efficient even on long time series
- **Statistically grounded**: Supports BIC/AIC-style penalty selection
- **Flexible**: Compatible with various cost functions (`GaussianCost`, `L2Cost`, etc.)
- **Accurate**: Finds the optimal number and position of change points

### How TSMonitor uses it:
- Detects change points in **deseasonalized series** (trend + residuals)
- Uses [`skchange`](https://github.com/florentfav/skchange) for fast implementation
- Adjustable sensitivity via the `penalty` parameter
- Used here to detect change points in the deseasonalized series (trend + residuals) — a best practice in monitoring tasks.


> Tip: Lower penalty values detect more frequent changes; higher values yield fewer but more confident breakpoints.


## Usage

```python
import pandas as pd
from ts_monitor import TSMonitor

# Step 1: Load your data
df = pd.read_csv("your_timeseries.csv")  # Must contain 'ds' and 'y' columns

# Step 2: Create the monitor
monitor = TSMonitor(ts_id="ts_001", ts_df=df)

# Step 3: STL or MSTL decomposition
monitor.decompose_stl(period=7)  # For weekly seasonality
# or use multiple seasonalities:
# monitor.decompose_mstl(periods=(7, 365))

# Step 4: Outlier Detection
monitor.detect_outliers(lower_quantile=0.02, upper_quantile=0.98)
monitor.plot_outliers()

# Step 5: Change Point Detection (PELT)
monitor.detect_change_points_pelt(method="deseasoned")
monitor.plot_deseasoned(show_change_points=True)

# Step 6: STL Components Plot
monitor.plot_stl()

# Step 7: Original series with change points
monitor.plot_original(show_change_points=True)

# Step 8: Full analysis as DataFrame
result_df = monitor.get_full_analysis()
print(result_df.head())
```

## Full Analysis Output

After running decomposition, outlier detection, and change point detection, you can generate a complete DataFrame using:

```python
full_df = monitor.get_full_analysis()
```
### Columns Returned

| Column         | Type              | Description                                                                 |
|----------------|-------------------|-----------------------------------------------------------------------------|
| `ds`           | `datetime64[ns]`  | Date of the observation                                                     |
| `y`            | `float` or `int`  | Original time series value                                                  |
| `trend`        | `float`           | Smoothed long-term trend extracted via STL/MSTL                             |
| `season`       | `float`           | Seasonal component (e.g., weekly/yearly pattern)                            |
| `residuals`    | `float`           | Remaining noise: `y - (trend + season)`                                     |
| `outlier`      | `bool`            | Whether the residual is an outlier based on quantile thresholds             |
| `change_points`| `bool`            | Whether a structural change point was detected at this date                 |


