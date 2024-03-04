import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "C:/Users/bgiet/OneDrive/Documents/GitHub/ConflictingForces/ConflictingForces/sensitive_files/sensitive_files/ConflictingForcesSensitive/Merged_Child_subset.csv"
df = pd.read_csv(file_path)

# Teacher ratings
variables_of_interest = ["TC_reading", "TC_writing", "TC_comprehension",
                         "TC_maths", "TC_imagin_creat", "TC_oral_comm", "TC_prob_solving"]
df_subset = df[variables_of_interest]

# Labels

sns.set(style="whitegrid")
category_labels = ["Below average", "Average", "Above average"]

variable_labels = {
    "TC_comprehension": "Comprehension",
    "TC_imagin_creat": "Imagination/Creativity",
    "TC_oral_comm": "Oral communication",
    "TC_prob_solving": "Problem solving",
    "TC_reading": "Reading",
    "TC_writing": "Writing",
    "TC_maths": "Maths"
}

sns.set(style="whitegrid")

# Plot 1 - distribution teacher ratings
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_1 = ["TC_reading", "TC_writing", "TC_maths"]
for i, variable in enumerate(variables_1):
    row = i // 2
    col = i % 2
    sns.countplot(data=df_subset, x=variable, palette="pastel", ax=axes[row, col])
    axes[row, col].set_xticklabels(category_labels)  # Set the category labels
    axes[row, col].set_xlabel(f"{variable_labels[variable]} Ratings")  # Set the variable label
    axes[row, col].set_ylabel("Count")
    axes[row, col].set_title(f"Distribution of {variable_labels[variable]} Ratings")

# Hide the last plot in the last position
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# Plot 2 - distribution teacher ratings
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_2 = ["TC_comprehension", "TC_imagin_creat", "TC_oral_comm", "TC_prob_solving"]
for i, variable in enumerate(variables_2):
    sns.countplot(data=df_subset, x=variable, palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels)  # Set the category labels
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings")  # Set the variable label
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings")

plt.tight_layout()
plt.show()

# Correlation coefficients comparison:

from matplotlib import rcParams

data = pd.DataFrame({
    'Pair': ['Reading * Comprehension', 'Problem Solving * Maths', 'Reading * Writing',
             'Writing * Comprehension', 'Comprehension * Problem Solving', 'Comprehension * Maths',
             'Comprehension * Oral communication', 'Reading * Problem Solving',
             'Oral communication * Imagination/Creativity', 'Reading * Maths',
             'Writing * Problem Solving', 'Reading * Oral communication', 'Maths * Writing',
             'Comprehension * Imagination/Creativity', 'Problem Solving * Oral communication',
             'Problem Solving * Imagination/Creativity', 'Writing * Oral communication',
             'Writing * Imagination/Creativity', 'Reading * Imagination/Creativity',
             'Maths * Oral communication', 'Maths * Imagination/Creativity'],
    'Pearson': [0.7995, 0.7822, 0.7425, 0.7337, 0.6967, 0.6711, 0.6287, 0.6277, 0.623,
                0.6045, 0.6039, 0.5879, 0.5876, 0.5822, 0.581, 0.5542, 0.5468, 0.5397,
                0.5347, 0.5129, 0.5032],
    'Spearman': [0.804, 0.7836, 0.7392, 0.7383, 0.6936, 0.6718, 0.6386, 0.6264, 0.6299,
                 0.6095, 0.6072, 0.6031, 0.5911, 0.5843, 0.5871, 0.5538, 0.5557, 0.5452,
                 0.5422, 0.5177, 0.5031],
    'Kendall': [0.7798, 0.759, 0.7131, 0.7128, 0.6666, 0.6438, 0.6129, 0.5966, 0.6107,
                0.5782, 0.5754, 0.5745, 0.5596, 0.5603, 0.56, 0.5298, 0.5286, 0.5204,
                0.5164, 0.4929, 0.4806]
})

sns.set(style="whitegrid")
rcParams['font.size'] = 10

plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Pearson', y='Pair', color='red', label='Pearson')
sns.scatterplot(data=data, x='Spearman', y='Pair', color='blue', label='Spearman')
sns.scatterplot(data=data, x='Kendall', y='Pair', color='green', label='Kendall')

plt.xlabel('Correlation Coefficient')
plt.ylabel('Pair')
plt.legend(title='Correlation')

plt.show()