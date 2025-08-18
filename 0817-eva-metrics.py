import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

# ====== Input Data ======
## Drug 7 ##
t_ref = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360]
y_ref = [[0, 38, 53, 64, 70, 79, 83, 89, 92, 98, 100]]

### Result-0 by using 0814-AdapRAG.py
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

# ====== Functions ======
def align_profiles(t_ref, y_ref, t_test, y_test):
    return np.interp(t_test, t_ref, y_ref), np.array(y_test)

def calc_msd(ref, test):
    return np.mean((ref - test) ** 2)

def f2(ref, test):
    return 50 * np.log10(100 / np.sqrt(1 + np.mean((ref - test) ** 2)))

def bootstrap_f2(ref, test, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n_points = len(ref)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_points, n_points)
        boot_vals.append(f2(ref[idx], test[idx]))
    boot_vals = np.array(boot_vals)
    return np.mean(boot_vals), (np.percentile(boot_vals, 5), np.percentile(boot_vals, 95))

def bca_ci(ref, test, alpha=0.10, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    obs_stat = f2(ref, test)
    n_points = len(ref)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_points, n_points)
        boot_stats.append(f2(ref[idx], test[idx]))
    boot_stats = np.array(boot_stats)
    z0 = norm.ppf(np.mean(boot_stats < obs_stat))
    jack_stats = []
    for i in range(n_points):
        idx = np.delete(np.arange(n_points), i)
        jack_stats.append(f2(ref[idx], test[idx]))
    jack_stats = np.array(jack_stats)
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    alphas = [alpha / 2, 1 - alpha / 2]
    zs = [z0 + norm.ppf(a_) for a_ in alphas]
    adj_alpha = [norm.cdf(z0 + zs[i] / (1 - a * zs[i])) for i in range(2)]
    ci = np.percentile(boot_stats, [100 * adj_alpha[0], 100 * adj_alpha[1]])
    return obs_stat, tuple(ci)

# ====== Run calculations ======
results = []
for name, y_test_vals in test_groups.items():
    ref_aligned, test_aligned = align_profiles(t_ref, y_ref[0], t_test, y_test_vals)
    msd_val = calc_msd(ref_aligned, test_aligned)
    f2_val = f2(ref_aligned, test_aligned)
    f2_boot_mean, f2_boot_ci = bootstrap_f2(ref_aligned, test_aligned)
    f2_obs, f2_bca_ci = bca_ci(ref_aligned, test_aligned)
    results.append({
        "Group": name,
        "MSD": msd_val,
        "f2": f2_val,
        "boot_f2_mean": f2_boot_mean,
        "boot_f2_CI_low": f2_boot_ci[0],
        "boot_f2_CI_high": f2_boot_ci[1],
        "BCa_f2_CI_low": f2_bca_ci[0],
        "BCa_f2_CI_high": f2_bca_ci[1]
    })

df_results = pd.DataFrame(results)
pd.set_option("display.precision", 3)
print(df_results)

# ====== Plot table as image ======
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
tbl = ax.table(cellText=df_results.round(3).values,
               colLabels=df_results.columns,
               cellLoc='center',
               loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.2)
plt.tight_layout()
plt.show()
