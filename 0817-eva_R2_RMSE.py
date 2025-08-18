import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# ====== Input Data ======
## Drug 7 ##
t_ref = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360]
y_ref = [[0, 38, 53, 64, 70, 79, 83, 89, 92, 98, 100]]

### Result
t_test = [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300]
test_groups = {
    "ZS": [0, 22.5, 38.9, 54.1, 78.6, 90.8, 97.2, 99.7, 100, 100, 100, 100],
    "ZS-CoT": [0, 23.1, 41.7, 58.3, 79.8, 88.6, 93.2, 96.5, 98.1, 99.3, 99.8, 100],
    "FS": [0, 23.2, 39.8, 55.1, 79.4, 91.2, 97.5, 99.9, 100.0, 100.0, 100.0, 100.0],
    "FS-CoT": [0, 18.2, 34.7, 49.1, 72.4, 79.8, 81.0, 81, 81, 81, 81, 81],
    "AdaRAG-ZS": [0, 25.8, 41.2, 53.1, 75.2, 86.9, 92.7, 97.4, 99.0, 99.8, 99.9, 100],
    "AdaRAG-ZS-CoT": [0, 9.5, 18.1, 25.9, 45.1, 59.4, 69.9, 83.4, 90.9, 97.5, 99.3, 99.8],
    "AdaRAG-FS": [0, 3.0, 5.9, 8.6, 16.9, 24.4, 31.7, 44.9, 56.4, 75.0, 86.4, 93.6],
    "AdaRAG-FS-CoT": [0, 12.1, 22.3, 30.5, 48.7, 61.2, 70.1, 82.4, 88.9, 94.6, 96.8, 97.5],
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
