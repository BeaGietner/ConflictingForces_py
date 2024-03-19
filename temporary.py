import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lets_plot
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

file_path = "C:/Users/bgiet/OneDrive/Documents/GitHub/ConflictingForces/ConflictingForces/sensitive_files/ConflictingForcesSensitive/Merged_Child_subset.csv"
Merged_Child_subset = pd.read_csv(file_path)
df = pd.read_csv(file_path)

# Teacher ratings
variables_of_interest = ["TC_reading", "TC_writing", "TC_comprehension", 
                         "TC_maths", "TC_imagin_creat", "TC_oral_comm", "TC_prob_solving", "Gender_MF"]
df_subset = df[variables_of_interest]

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

variables_1 = ["TC_prob_solving", "TC_maths", "TC_comprehension"]
for i, variable in enumerate(variables_1):
    row = i // 2
    col = i % 2
    sns.countplot(data=df_subset, x=variable, palette="pastel", ax=axes[row, col])
    axes[row, col].set_xticklabels(category_labels)  
    axes[row, col].set_xlabel(f"{variable_labels[variable]} Ratings")  
    axes[row, col].set_ylabel("Count")
    axes[row, col].set_title(f"Distribution of {variable_labels[variable]} Ratings")

# Hide the last plot in the last position
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()



# Plot 2 - distribution teacher ratings 
fig, axes = plt.subplots(2, 2, figsize=(10, 10))


variables_2 = ["TC_reading", "TC_writing", "TC_oral_comm", "TC_imagin_creat"]
for i, variable in enumerate(variables_2):
    sns.countplot(data=df_subset, x=variable, palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels)  
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings") 
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings")

plt.tight_layout()
plt.show()

# Plot 3 - distribution teacher ratings by gender
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_3 = ["TC_prob_solving", "TC_maths", "TC_comprehension"]

for i, variable in enumerate(variables_3):
    sns.countplot(data=df_subset, x=variable, hue="Gender_MF", palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels) 
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings")  
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings by Gender")
    axes[i // 2, i % 2].legend(["Male", "Female"])

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


# Plot 4 - distribution teacher ratings by gender
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_4 = ["TC_reading", "TC_writing", "TC_oral_comm", "TC_imagin_creat"]

for i, variable in enumerate(variables_4):
    sns.countplot(data=df_subset, x=variable, hue="Gender_MF", palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels) 
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings")  
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings by Gender")
    axes[i // 2, i % 2].legend(["Male", "Female"])

plt.tight_layout()
plt.show()


# Dropping rows with missing values:

df_subset_complete = df_subset.dropna()
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_3 = ["TC_prob_solving", "TC_maths", "TC_comprehension"]
for i, variable in enumerate(variables_3):
    sns.countplot(data=df_subset_complete, x=variable, hue="Gender_MF", palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels) 
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings")  
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings by Gender")
    axes[i // 2, i % 2].legend(["Male", "Female"])

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

variables_4 = ["TC_reading", "TC_writing", "TC_oral_comm", "TC_imagin_creat"]
for i, variable in enumerate(variables_4):
    sns.countplot(data=df_subset_complete, x=variable, hue="Gender_MF", palette="pastel", ax=axes[i // 2, i % 2])
    axes[i // 2, i % 2].set_xticklabels(category_labels)  
    axes[i // 2, i % 2].set_xlabel(f"{variable_labels[variable]} Ratings") 
    axes[i // 2, i % 2].set_ylabel("Count")
    axes[i // 2, i % 2].set_title(f"Distribution of {variable_labels[variable]} Ratings by Gender")
    axes[i // 2, i % 2].legend(["Male", "Female"])

plt.tight_layout()
plt.show()


# Correlation coefficients comparison:
    

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


# Creating the dataframe (proportion of ratings in abilities by gender)

results = pd.DataFrame({
    'Gender': ['Male', 'Female'],
    'Reading_1': [0.1317409, 0.1119454],
    'Reading_2': [0.4039088, 0.4102389],
    'Reading_3': [0.4643503, 0.4778157],
    'Writing_1': [0.1682953, 0.1139932],
    'Writing_2': [0.4748462, 0.4829352],
    'Writing_3': [0.3568585, 0.4030717],
    'Comprehension_1': [0.1053203, 0.1061433],
    'Comprehension_2': [0.4491495, 0.4675768],
    'Comprehension_3': [0.4455302, 0.4262799],
    'Maths_1': [0.09627217, 0.12798635],
    'Maths_2': [0.45313066, 0.51945392],
    'Maths_3': [0.45059718, 0.35255973],
    'Imagination_Creativity_1': [0.06623236, 0.05426621],
    'Imagination_Creativity_2': [0.54578357, 0.53447099],
    'Imagination_Creativity_3': [0.38798408, 0.41126280],
    'Oral_Communication_1': [0.07636627, 0.05358362],
    'Oral_Communication_2': [0.50126674, 0.51774744],
    'Oral_Communication_3': [0.42236699, 0.42866894],
    'Problem_Solving_1': [0.1208831, 0.1464164],
    'Problem_Solving_2': [0.4983713, 0.5648464],
    'Problem_Solving_3': [0.3807456, 0.2887372]
})

results_long = results.melt(id_vars='Gender', var_name='Ability', value_name='Proportion')

results_long['Rating'] = results_long['Ability'].str.extract(r'_(\d)')
results_long['Rating'] = results_long['Rating'].map({'1': 'Below Average', '2': 'Average', '3': 'Above Average'})

results_long['Ability'] = results_long['Ability'].str.replace('_1', '').str.replace('_2', '').str.replace('_3', '')
results_long['Ability'] = results_long['Ability'].replace({
    'Reading': 'Reading',
    'Writing': 'Writing',
    'Comprehension': 'Comprehension',
    'Maths': 'Maths',
    'Imagination_Creativity': 'Imagination/Creativity',
    'Oral_Communication': 'Oral Communication',
    'Problem_Solving': 'Problem Solving'
})

pastel_blue = (0.6, 0.8, 1)
pastel_pink = (1, 0.6, 0.8)

# Separate plots for each proportion rating
proportions = results_long['Rating'].unique()
for prop in proportions:
    prop_data = results_long[results_long['Rating'] == prop]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=prop_data, x='Ability', y='Proportion', hue='Gender', palette=[pastel_blue, pastel_pink])
    plt.title(f"Proportion of {prop} Students")
    plt.xlabel("Ability")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()

# and 

results_long = results.melt(id_vars='Gender', var_name='Ability', value_name='Proportion')

results_long['Rating'] = results_long['Ability'].str.extract(r'_(\d)')
results_long['Rating'] = results_long['Rating'].map({'1': 'Below Average', '2': 'Average', '3': 'Above Average'})

results_long['Ability'] = results_long['Ability'].str.replace('_1', '').str.replace('_2', '').str.replace('_3', '')
results_long['Ability'] = results_long['Ability'].replace({
    'Reading': 'Reading',
    'Writing': 'Writing',
    'Comprehension': 'Comprehension',
    'Maths': 'Maths',
    'Imagination_Creativity': 'Imagination/Creativity',
    'Oral_Communication': 'Oral Communication',
    'Problem_Solving': 'Problem Solving'
})

# Separate plots for each ability
abilities = results_long['Ability'].unique()
for ability in abilities:
    ability_data = results_long[results_long['Ability'] == ability]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ability_data, x='Rating', y='Proportion', hue='Gender', palette=[pastel_blue, pastel_pink])
    plt.title(f"Proportion of Students in {ability}")
    plt.xlabel("Rating")
    plt.ylabel("Proportion")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.show()
    
# ------------------

Merged_Child_subset['Maths_f'] = pd.Categorical(Merged_Child_subset['Maths_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['Irish_f'] = pd.Categorical(Merged_Child_subset['Irish_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['History_f'] = pd.Categorical(Merged_Child_subset['History_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['Geography_f'] = pd.Categorical(Merged_Child_subset['Geography_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['Science_f'] = pd.Categorical(Merged_Child_subset['Science_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['English_f'] = pd.Categorical(Merged_Child_subset['English_f'], categories=['E or lower', 'D', 'C', 'B', 'A'], ordered=True)
Merged_Child_subset['Agreeable_W3_YP'] = Merged_Child_subset['Agreeable_W3_YP'].astype('category')
Merged_Child_subset['Conscientious_W3_YP'] = Merged_Child_subset['Conscientious_W3_YP'].astype('category')
Merged_Child_subset['Emo_Stability_W3_YP'] = Merged_Child_subset['Emo_Stability_W3_YP'].astype('category')
Merged_Child_subset['Extravert_W3_YP'] = Merged_Child_subset['Extravert_W3_YP'].astype('category')
Merged_Child_subset['Openness_W3_YP'] = Merged_Child_subset['Openness_W3_YP'].astype('category')


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.mosaicplot as sp

colors = {
    'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange', 'E or lower': 'purple',
    '1': 'lightcoral', '1.5': 'lightblue', '2': 'lightgreen', '2.5': 'lightsalmon',
    '3': 'lightseagreen', '3.5': 'lightpink', '4': 'lightyellow', '4.5': 'lightskyblue',
    '5': 'lightgreen', '5.5': 'lightcoral', '6': 'lightblue', '6.5': 'lightgreen', '7': 'lightpink'
}

complete_cases_test = Merged_Child_subset.dropna(subset=["Agreeable_W3_YP", "Conscientious_W3_YP", 
                                                         "Emo_Stability_W3_YP", "Extravert_W3_YP", 
                                                         "Openness_W3_YP", "Maths_f"])

plt.figure(figsize=(10, 8))
sp.mosaic(pd.crosstab(complete_cases_test["Maths_f"], complete_cases_test["Openness_W3_YP"]).stack(),
          title="Mosaic Plot of Openness_W3_YP",
          properties=colors)  
plt.show()

# PCA:
    
predictors = ["Agreeable_W2_PCG", "Conscientious_W2_PCG", "Emo_Stability_W2_PCG",
              "Extravert_W2_PCG", "Openness_W2_PCG",
              "Agreeable_W3_PCG", "Conscientious_W3_PCG", "Emo_Stability_W3_PCG",
              "Extravert_W3_PCG", "Openness_W3_PCG",
              "Agreeable_W3_SCG", "Conscientious_W3_SCG", "Emo_Stability_W3_SCG",
              "Extravert_W3_SCG", "Openness_W3_SCG",
              "Agreeable_W3_YP", "Conscientious_W3_YP", "Emo_Stability_W3_YP", 
              "Extravert_W3_YP", "Openness_W3_YP",
              "DEIS_binary", "Fee_paying",
              "Discussed_exams_PCG", "Discussed_friends_PCG", "Discussed_future_PCG",
              "Discussed_teachers_PCG", "Discussed_workload_PCG", "Discussed_subjects_PCG",
              "Discussed_exams_YP", "Discussed_friends_YP", "Discussed_future_YP",
              "Discussed_teachers_YP", "Discussed_workload_YP", "Discussed_subjects_YP",
              "PCG_Educ", "SCG_Educ",
              "TC_writing", "TC_reading", "TC_comprehension","TC_maths","TC_imagin_creat",
              "TC_oral_comm","TC_prob_solving"]


subset_df = df[predictors].copy()
subset_df.dropna(inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(subset_df)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

loadings = pca.components_
loadings_df = pd.DataFrame(loadings, columns=subset_df.columns)
print(loadings_df)
loadings_df.to_csv('loadings.csv', index=False)

sorted_loadings_df = loadings_df.apply(abs).sort_values(by=loadings_df.columns[0], ascending=False)
top_10_contributors = sorted_loadings_df.head(10)
print(top_10_contributors)

loadings_df = pd.read_csv('loadings.csv')

predictor_names = [
    "Agreeable_W2_PCG", "Conscientious_W2_PCG", "Emo_Stability_W2_PCG", "Extravert_W2_PCG", "Openness_W2_PCG",
    "Agreeable_W3_PCG", "Conscientious_W3_PCG", "Emo_Stability_W3_PCG", "Extravert_W3_PCG", "Openness_W3_PCG",
    "Agreeable_W3_SCG", "Conscientious_W3_SCG", "Emo_Stability_W3_SCG", "Extravert_W3_SCG", "Openness_W3_SCG",
    "Agreeable_W3_YP", "Conscientious_W3_YP", "Emo_Stability_W3_YP", "Extravert_W3_YP", "Openness_W3_YP",
    "DEIS_binary", "Fee_paying",
    "Discussed_exams_PCG", "Discussed_friends_PCG", "Discussed_future_PCG", "Discussed_teachers_PCG",
    "Discussed_workload_PCG", "Discussed_subjects_PCG",
    "Discussed_exams_YP", "Discussed_friends_YP", "Discussed_future_YP", "Discussed_teachers_YP",
    "Discussed_workload_YP", "Discussed_subjects_YP",
    "PCG_Educ", "SCG_Educ",
    "TC_writing", "TC_reading", "TC_comprehension", "TC_maths", "TC_imagin_creat",
    "TC_oral_comm", "TC_prob_solving"]
    
    
predictor_variables = loadings_df.columns
top_10_loadings = loadings_df.iloc[top_10_contributors.index]
max_loadings = top_10_loadings.abs().max(axis=1)
top_10_predictors = predictor_variables[max_loadings.idxmax()]
print("Top 10 Predictor Variables:")
print(top_10_predictors)

# First component

abs_loadings = loadings_df.abs()
sorted_loadings = abs_loadings.iloc[:, 0].sort_values(ascending=False)
top_10_contributors = sorted_loadings.head(10)

plt.figure(figsize=(10, 6))
top_10_contributors.plot(kind='barh')
plt.title('Top 10 Contributors for the First Principal Component')
plt.xlabel('Absolute Loadings')
plt.ylabel('Predictors')
plt.show()

plt.figure(figsize=(10, 6))
top_10_contributors.plot(kind='barh')
plt.xlabel('Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.title('Top 10 Contributors to First Principal Component')
plt.gca().invert_yaxis()  
plt.show()
# or
plt.figure(figsize=(10, 6))
top_10_contributors.plot(kind='barh')
plt.xlabel('Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.yticks(range(10), [predictor_names[idx] for idx in top_10_contributors.index])  
plt.title('Top 10 Contributors to First Principal Component')
plt.gca().invert_yaxis()  
plt.show()

# Second component:
    
abs_loadings = loadings_df.abs()
sorted_loadings = abs_loadings.iloc[:, 1].sort_values(ascending=False)  
top_10_contributors = sorted_loadings.head(10)

plt.figure(figsize=(10, 6))
top_10_contributors.plot(kind='barh')
plt.xlabel('Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.yticks(range(10), [predictor_names[idx] for idx in top_10_contributors.index])  
plt.title('Top 10 Contributors to Second Principal Component')
plt.gca().invert_yaxis()
plt.show()

# Cumulative ranking


cumulative_loadings = {i: 0 for i in range(len(predictor_names))}

for i in range(10):
    component_loadings = abs(loadings_df.iloc[:, i])
    for j in range(len(predictor_names)):
        cumulative_loadings[j] += component_loadings[j]


cumulative_loadings_df = pd.DataFrame.from_dict(cumulative_loadings, orient='index', columns=['Cumulative Loadings'])

sorted_cumulative_loadings_df = cumulative_loadings_df.sort_values(by='Cumulative Loadings', ascending=False)

plt.figure(figsize=(10, 8))
sorted_cumulative_loadings_df.head(20).plot(kind='barh', legend=False)
plt.xlabel('Cumulative Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.title('Top Contributors across First 10 Principal Components')
plt.gca().invert_yaxis() 
plt.show()

# 20 principal components

cumulative_loadings = {i: 0 for i in range(len(predictor_names))}

for i in range(20):
    component_loadings = abs(loadings_df.iloc[:, i])
    for j in range(len(predictor_names)):
        cumulative_loadings[j] += component_loadings[j]

cumulative_loadings_df = pd.DataFrame.from_dict(cumulative_loadings, orient='index', columns=['Cumulative Loadings'])

sorted_cumulative_loadings_df = cumulative_loadings_df.sort_values(by='Cumulative Loadings', ascending=False)

plt.figure(figsize=(10, 12))  
sorted_cumulative_loadings_df.head(20).plot(kind='barh', legend=False)
plt.xlabel('Cumulative Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.title('Top Contributors across First 20 Principal Components')
plt.gca().invert_yaxis() 
plt.show()

index_to_predictor = {i: predictor_names[i] for i in range(len(predictor_names))}
sorted_indices = [3, 7, 2, 6, 10, 8, 9, 18, 26, 17, 31, 21, 15, 33, 12, 27, 19, 37, 28, 30]

plt.figure(figsize=(8, 6))
plt.barh([index_to_predictor[idx] for idx in sorted_indices], cumulative_loadings_df.iloc[sorted_indices]['Cumulative Loadings'])
plt.xlabel('Cumulative Absolute Loadings')
plt.ylabel('Predictor Variables')
plt.title('Top Contributors across First 20 Principal Components')
plt.gca().invert_yaxis() 
plt.show()

# OLS OLR

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


subset_df = df[predictors].copy()
subset_df.dropna(inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(subset_df)


principal_components = pca.transform(scaled_data)
selected_principal_components = principal_components[:, :10]
ols_target_variable = df['Maths'].copy()
X_train_ols, X_test_ols, y_train_ols, y_test_ols = train_test_split(selected_principal_components, ols_target_variable, test_size=0.2, random_state=42)
ols_model = LinearRegression()
ols_model.fit(X_train_ols, y_train_ols)
ols_predictions = ols_model.predict(X_test_ols)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

predictors = ["Agreeable_W2_PCG", "Conscientious_W2_PCG", "Emo_Stability_W2_PCG",
              "Extravert_W2_PCG", "Openness_W2_PCG",
              "Agreeable_W3_PCG", "Conscientious_W3_PCG", "Emo_Stability_W3_PCG",
              "Extravert_W3_PCG", "Openness_W3_PCG",
              "Agreeable_W3_SCG", "Conscientious_W3_SCG", "Emo_Stability_W3_SCG",
              "Extravert_W3_SCG", "Openness_W3_SCG",
              "Agreeable_W3_YP", "Conscientious_W3_YP", "Emo_Stability_W3_YP", 
              "Extravert_W3_YP", "Openness_W3_YP",
              "DEIS_binary", "Fee_paying",
              "Discussed_exams_PCG", "Discussed_friends_PCG", "Discussed_future_PCG",
              "Discussed_teachers_PCG", "Discussed_workload_PCG", "Discussed_subjects_PCG",
              "Discussed_exams_YP", "Discussed_friends_YP", "Discussed_future_YP",
              "Discussed_teachers_YP", "Discussed_workload_YP", "Discussed_subjects_YP",
              "PCG_Educ", "SCG_Educ",
              "TC_writing", "TC_reading", "TC_comprehension","TC_maths","TC_imagin_creat",
              "TC_oral_comm","TC_prob_solving"]

data = Merged_Child_subset[predictors].dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA()
pca_result = pca.fit(scaled_data)

plt.plot(range(1, pca_result.n_components_ + 1), np.cumsum(pca_result.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

loadings = pca_result.components_.T

sorted_loadings_idx = np.argsort(np.abs(loadings[:, 0]))[::-1]
top_20_contributors = loadings[sorted_loadings_idx[:20], :]
variable_names = data.columns[sorted_loadings_idx[:20]]
for i, var_name in enumerate(variable_names):
    print(f"{var_name}: {top_20_contributors[:, i]}")
loadings_df = pd.DataFrame(top_20_contributors, index=variable_names, columns=[f"Dim.{i}" for i in range(1, top_20_contributors.shape[1] + 1)])
print(loadings_df)


# top cumulative loadings for all PCs
cumulative_abs_loadings = np.sum(abs_loadings, axis=0)
sorted_cumulative_loadings = np.sort(cumulative_abs_loadings)[::-1]
top_20_cumulative_loadings = sorted_cumulative_loadings[:20]
print(top_20_cumulative_loadings)
predictor_cumulative_loadings = dict(zip(predictors, cumulative_abs_loadings))
sorted_predictor_cumulative_loadings = {k: v for k, v in sorted(predictor_cumulative_loadings.items(), key=lambda item: item[1], reverse=True)}
print("Top 20 Predictor Variables and Their Cumulative Loadings:")
for i, (predictor, loading) in enumerate(sorted_predictor_cumulative_loadings.items(), start=1):
    print(f"{i}. {predictor}: {loading:.2f}")

# top 20 cumulative loadings 

cumulative_abs_loadings = np.sum(abs_loadings[:, :20], axis=0)
sorted_cumulative_loadings = np.sort(cumulative_abs_loadings)[::-1]
top_20_cumulative_loadings = sorted_cumulative_loadings[:20]
predictor_cumulative_loadings = dict(zip(predictors, cumulative_abs_loadings))
sorted_predictor_cumulative_loadings = {k: v for k, v in sorted(predictor_cumulative_loadings.items(), key=lambda item: item[1], reverse=True)}
print("Top 20 Predictor Variables and Their Cumulative Loadings for the First 20 Principal Components:")
for i, (predictor, loading) in enumerate(sorted_predictor_cumulative_loadings.items(), start=1):
    print(f"{i}. {predictor}: {loading:.2f}")
    
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.6)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA')
plt.show()

