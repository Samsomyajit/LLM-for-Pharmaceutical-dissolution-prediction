import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Reference data
t_ref = [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360]
y_ref = [[0, 38, 53, 64, 70, 79, 83, 89, 92, 98, 100, 100, 100]]  # Single reference batch

# Test data
t_test = [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300]
test_groups = {
    "ZS": [0, 16.2, 31.7, 45.9, 71.3, 87.4, 96.8, 99.7, 100.0, 100.0, 100.0, 100.0],
    "ZS-CoT": [0, 22.3, 41.7, 58.9, 82.5, 96.1, 99.8, 100.0, 100.0, 100.0, 100.0, 100.0],
    "FS": [0, 16.2, 31.7, 45.9, 71.3, 87.4, 96.8, 99.7, 100.0, 100.0, 100.0, 100.0],
    "FS-CoT": [0, 28.4, 49.7, 65.2, 87.1, 98.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    "AdaRAG-ZS": [0, 18.0, 32.0, 45.0, 65.0, 78.0, 85.0, 92.0, 95.0, 98.0, 99.0, 100.0],
    "AdaRAG-ZS-CoT": [0, 8.6, 16.5, 23.1, 39.3, 51.2, 60.6, 75.3, 84.5, 93.2, 97.0, 99.3],
    "AdaRAG-FS": [0, 12.0, 24.0, 35.0, 55.0, 70.0, 82.0, 92.0, 96.0, 98.0, 99.0, 99.5],
    "AdaRAG-FS-CoT": [0, 15.3, 28.7, 38.9, 62.1, 77.4, 86.5, 94.8, 97.9, 99.6, 99.9, 100.0],
}

# 1. Create common time grid (union of reference/test times)
common_time = sorted(list(set(t_ref + t_test)))  # Includes all points up to 360

# 2. Interpolate reference profile to common time
ref_interp = np.interp(common_time, t_ref, y_ref[0])

# 3. Calculate metrics for each test group
results = []
for group_name, test_profile in test_groups.items():
    # Interpolate test profile to common time
    test_interp = np.interp(common_time, t_test, test_profile)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ref_interp, test_interp))
    r2 = r2_score(ref_interp, test_interp)
    pcc, _ = pearsonr(ref_interp, test_interp)

    results.append({
        "Group": group_name,
        "RMSE": round(rmse, 2),
        "R²": round(r2, 3),
        "PCC": round(pcc, 3)
    })

# Print results as a table
import pandas as pd

df = pd.DataFrame(results)
print(df)
