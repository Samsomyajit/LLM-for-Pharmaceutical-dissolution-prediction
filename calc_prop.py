import numpy as np
import pandas as pd
from skimage import measure, morphology
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def process_mask(mask_path: str, output_dir: str, scale=1/60):
    """
    Process a single mask image and save analysis results.
    
    Args:
        mask_path (str): Path to the mask image (npy or tif).
        output_dir (str): Directory to save results.
        scale (float): Scale factor (pixels to micrometers).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mask - handle both .npy and image formats
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    else:
        # For TIF files from cellpose
        mask = plt.imread(mask_path)
        # Handle potential multi-channel or float32 masks
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.dtype != np.uint16 and mask.dtype != np.int32:
            mask = mask.astype(np.uint16)
    
    # Cellpose masks are already labeled - no need for measure.label(mask > 0)
    # Just verify it's a proper labeled image
    if mask.max() == 0:
        print(f"⚠️ Warning: No cells detected in mask {mask_path}")
        return
    
    # Get region properties directly from the labeled mask
    regions = measure.regionprops(mask)
    
    # Filter small particles (area < 5 pixels)
    valid_regions = [r for r in regions if r.area >= 5]
    if not valid_regions:
        print(f"⚠️ Warning: No valid particles found in mask {mask_path}")
        return
    
    # Calculate properties
    data = []
    for region in valid_regions:
        # Use Feret diameter as primary size metric
        diameter = calculate_feret_diameter(region) * scale
        data.append({
            "Feret Diameter (µm)": diameter,
            "Equivalent Diameter (µm)": calculate_equivalent_diameter(region) * scale,
            "Aspect Ratio": calculate_aspect_ratio(region),
            "Circularity": calculate_circularity(region),
            "Solidity": calculate_solidity(region),
            "Convexity": calculate_convexity(region)
        })
    
    df = pd.DataFrame(data)
    if df.empty:
        return
    
    # Save Excel
    excel_path = os.path.join(output_dir, "particle_analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Particles", index=False)
        d10, d50, d90, d99 = cumulative_percentiles(df["Feret Diameter (µm)"].values)
        pd.DataFrame([{
            "D10 (µm)": d10,
            "D50 (µm)": d50,
            "D90 (µm)": d90,
            "D99 (µm)": d99
        }]).to_excel(writer, sheet_name="D_Values", index=False)
    
    # Save JSON summary
    json_path = os.path.join(output_dir, "properties.json")
    with open(json_path, 'w') as f:
        json.dump({
            "d10": float(d10),
            "d50": float(d50),
            "d90": float(d90),
            "d99": float(d99),
            "particle_count": len(df),
            "scale_factor": scale,
            "mask_file": os.path.basename(mask_path)
        }, f)
    
    # Save visualizations
    save_visualizations(df, output_dir)

def save_visualizations(df, output_dir):
    """Save distribution visualizations for key properties"""
    plt.figure(figsize=(12, 8))
    
    # Feret Diameter distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df["Feret Diameter (µm)"], kde=True)
    plt.title("Feret Diameter Distribution")
    plt.xlabel("Diameter (µm)")
    
    # Equivalent Diameter distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df["Equivalent Diameter (µm)"], kde=True)
    plt.title("Equivalent Diameter Distribution")
    plt.xlabel("Diameter (µm)")
    
    # Aspect Ratio distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df["Aspect Ratio"], kde=True)
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("Aspect Ratio")
    
    # Circulariy distribution
    plt.subplot(2, 2, 4)
    sns.histplot(df["Circularity"], kde=True)
    plt.title("Circularity Distribution")
    plt.xlabel("Circularity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distributions.png"))
    plt.close()

# Geometry Calculations
def calculate_feret_diameter(region):
    """Calculate maximum Feret diameter"""
    minr, minc, maxr, maxc = region.bbox
    return np.hypot(maxr - minr, maxc - minc)

def calculate_equivalent_diameter(region):
    """Calculate diameter of equivalent area circle"""
    return np.sqrt(4 * region.area / np.pi)

def calculate_aspect_ratio(region):
    """Calculate aspect ratio (major_axis_length / minor_axis_length)"""
    if region.minor_axis_length == 0:
        return 0.0
    return region.major_axis_length / region.minor_axis_length

def calculate_circularity(region):
    """Calculate circularity (4π(area/perimeter²))"""
    if region.perimeter == 0:
        return 0.0
    return (4 * np.pi * region.area) / (region.perimeter ** 2)

def calculate_solidity(region):
    """Calculate solidity (area / convex_area)"""
    if region.convex_area == 0:
        return 0.0
    return region.area / region.convex_area

def calculate_convexity(region):
    """Calculate convexity (perimeter / convex_perimeter)"""
    convex_perimeter = measure.perimeter(region.convex_image)
    if convex_perimeter == 0:
        return 0.0
    return region.perimeter / convex_perimeter

def cumulative_percentiles(particle_sizes):
    """Calculate D10, D50, D90, D99 values"""
    sorted_sizes = np.sort(particle_sizes)
    n = len(sorted_sizes)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # Calculate indices for percentiles
    idx_10 = int(0.10 * n)
    idx_50 = int(0.50 * n)
    idx_90 = int(0.90 * n)
    idx_99 = int(0.99 * n)
    
    # Clamp indices to array bounds
    idx_10 = min(max(idx_10, 0), n-1)
    idx_50 = min(max(idx_50, 0), n-1)
    idx_90 = min(max(idx_90, 0), n-1)
    idx_99 = min(max(idx_99, 0), n-1)
    
    return (
        sorted_sizes[idx_10],
        sorted_sizes[idx_50],
        sorted_sizes[idx_90],
        sorted_sizes[idx_99]
    )

def process_directory(input_dir, output_base_dir, scale=1/60):
    """
    Process all mask files in a directory.
    
    Args:
        input_dir (str): Directory containing mask files.
        output_base_dir (str): Base directory for results.
        scale (float): Scale factor (pixels to micrometers).
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all mask files
    mask_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.npy', '.tif', '.tiff')):
                mask_files.append(os.path.join(root, file))
    
    # Process each mask
    results = []
    for mask_path in mask_files:
        print(f"Processing {mask_path}...")
        sample_name = Path(mask_path).stem
        output_dir = os.path.join(output_base_dir, sample_name)
        
        try:
            process_mask(mask_path, output_dir, scale)
            logger.info(f"✅ Analysis completed for {sample_name}")
        except Exception as e:
            logger.error(f"❌ Analysis failed for {sample_name}: {str(e)}")
    
    # Save overall summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(output_base_dir, "summary_results.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"Summary results saved to {summary_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mask_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "dissolution_prop_results"
        scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1/60
        process_mask(mask_path, output_dir, scale)
    else:
        print("Usage: python calc_prop.py <mask_path> [output_dir] [scale]")