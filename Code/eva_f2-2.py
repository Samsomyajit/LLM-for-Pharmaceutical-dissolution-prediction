import numpy as np


# Your provided function (slightly modified for clarity)
def calculate_f2(time_ref, reference_batches, time_test, test_profile):
    # Create common time grid
    common_time = sorted(list(set(time_ref + time_test)))

    # Interpolate reference batches to common time
    ref_interp = []
    for batch in reference_batches:
        interp_values = np.interp(common_time, time_ref, batch)
        ref_interp.append(interp_values)
    avg_ref = np.mean(ref_interp, axis=0)

    # Interpolate test profile to common time
    test_interp = np.interp(common_time, time_test, test_profile)

    # Find truncation point (first time ≥85% in either profile)
    trunc_idx = None
    for i, (r, t) in enumerate(zip(avg_ref, test_interp)):
        if r >= 85 or t >= 85:
            trunc_idx = i + 1  # Include the current time point
            break

    # Apply truncation and ensure ≥3 time points
    avg_trunc = avg_ref[:trunc_idx] if trunc_idx else avg_ref
    test_trunc = test_interp[:trunc_idx] if trunc_idx else test_interp
    if len(avg_trunc) < 3:
        avg_trunc = avg_ref[:3]
        test_trunc = test_interp[:3]

    # Calculate f2
    n = len(avg_trunc)
    sum_sq_diff = np.sum((avg_trunc - test_trunc) ** 2)
    f2 = 50 * np.log10(100 / np.sqrt(1 + (sum_sq_diff / n)))
    return np.round(f2, 1)


# Reference data
t_ref = [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360]
y_ref = [[0, 38, 53, 64, 70, 79, 83, 89, 92, 98, 100, 100, 100]]  # Single reference batch

# Test data (time points and profiles)
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

# Calculate f2 for each test group
f2_results = {}
for group, profile in test_groups.items():
    f2 = calculate_f2(t_ref, y_ref, t_test, profile)
    f2_results[group] = f2

# Print results
print("f2 Similarity Factors:")
for group, f2 in f2_results.items():
    print(f"{group}: {f2}")
