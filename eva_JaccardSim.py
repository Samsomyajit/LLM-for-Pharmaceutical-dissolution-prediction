import re
import pandas as pd


def extract_content(text):
    """Extract the scientific content between markers"""
    if "----- Recipe -----" in text:
        return text.split("----- Recipe -----")[-1].split("Process finished")[0]
    if "Response:" in text:
        return text.split("Response:")[-1].split("Process finished")[0]
    return text


def preprocess(text):
    """Clean and tokenize technical content"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Keep words/numbers
    return set(text.split())


def jaccard_similarity(set1, set2):
    """Calculate Jaccard Index between two sets"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


# Initialize matrix and labels
files = [f'FS-{i}.txt' for i in range(1, 11)]
token_sets = []
matrix = [[0.0] * 10 for _ in range(10)]

# Process all files
for i in range(10):
    with open(files[i], 'r') as f:
        content = extract_content(f.read())
        token_sets.append(preprocess(content))

# Calculate pairwise similarities
for i in range(10):
    for j in range(10):
        matrix[i][j] = jaccard_similarity(token_sets[i], token_sets[j])

# Create and format DataFrame
labels = [f'FS{i + 1}' for i in range(10)]
df = pd.DataFrame(matrix, index=labels, columns=labels)

# Print formatted table
print("\n10×10 Jaccard Similarity Matrix:")
print(df.to_string(formatters={col: "{:.2f}".format for col in df.columns}))

# Optional: Save to CSV
# df.to_csv("jaccard_similarity_matrix.csv")

# Optional: Visualize with matplotlib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Jaccard Index'})
    #plt.title("Pharmaceutical Formulation Similarity Matrix")
    plt.show()

except ImportError:
    print("\nInstall matplotlib and seaborn for visualization:")
    print("pip install matplotlib seaborn")
