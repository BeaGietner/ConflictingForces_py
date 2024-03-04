import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "C:/Users/bgiet/OneDrive/Documents/GitHub/ConflictingForces/ConflictingForces/Merged_Child_subset.csv"
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
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

variables_1 = ["TC_reading", "TC_writing", "TC_maths"]
for i, variable in enumerate(variables_1):
    sns.countplot(data=df_subset, x=variable, palette="pastel", ax=axes[i])
    axes[i].set_xticklabels(category_labels)  # Set the category labels
    axes[i].set_xlabel(f"{variable_labels[variable]} Ratings")  # Set the variable label
    axes[i].set_ylabel("Count")
    axes[i].set_title(f"Distribution of {variable_labels[variable]} Ratings")

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