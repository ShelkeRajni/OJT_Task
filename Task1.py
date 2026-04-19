#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print('Libraries loaded')


# In[12]:


df = pd.read_csv("school_data.csv")
print(f'Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns')


# In[13]:


df.head()


# In[14]:


print('Column Info')
df.info()


# In[15]:


df.describe().T


# In[16]:


# Data Understanding
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Shape of Dataset ---")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

# Identify target variable (example: 'Score' or 'Marks')
target = 'Score' if 'Score' in df.columns else df.columns[-1]
print("\nTarget Variable:", target)


# In[17]:


missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df[missing_df['Missing Count'] > 0]


# In[18]:


fig, ax = plt.subplots(figsize=(8, 4))
missing_pct[missing_pct > 0].plot(kind='bar', color='salmon', edgecolor='black', ax=ax)
ax.set_title('Missing Values (% per column)')
ax.set_ylabel('% Missing')
ax.set_xlabel('Column')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


# In[19]:


cols_with_missing = ['%_Math_Score', 'Teacher_Student_Ratio', 'Internet_Available', 'Parent_Literacy_Rate']

for col in cols_with_missing:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f'{col}: filled with median = {median_val:.2f}')

print('\nMissing values after imputation:')
print(df.isnull().sum().sum(), 'missing values remaining')


# In[20]:


dupes = df.duplicated().sum()
print(f'Duplicate rows found: {dupes}')

if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f'Duplicates removed. New shape: {df.shape}')
else:
    print('No duplicates — dataset is clean')


# In[21]:


num_cols = df.select_dtypes(include=np.number).columns.tolist()

print('Outlier counts per column (IQR method):')
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        print(f'  {col}: {outliers} outliers')


# In[22]:


#3. EDA – Exploratory Data Analysis


# In[25]:


# 3.1 Univariate Analysis – Score Distributions

score_cols = ['%_Math_Score', '%_Language_Score', '%_Science_Score']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#4C72B0', '#55A868', '#C44E52']

for ax, col, color in zip(axes, score_cols, colors):
    sns.histplot(df[col], kde=True, ax=ax, color=color, bins=30)
    ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.5, label=f'Mean: {df[col].mean():.1f}')
    ax.set_title(col.replace('%_', '').replace('_Score', ' Score'))
    ax.set_xlabel('Score (%)')
    ax.legend()

plt.suptitle('Distribution of Academic Scores', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# In[26]:


other_num = ['Teacher_Student_Ratio', 'Avg_Teacher_Experience_Years',
             'Teacher_Trainings_Attended', '%_Marginalized_Students',
             '%_Students_With_Home_Internet', 'Parent_Literacy_Rate']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, col in zip(axes, other_num):
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue', bins=25)
    ax.set_title(col.replace('_', ' '))
    ax.set_xlabel('')

plt.suptitle('Distribution of Feature Variables', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[27]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['Urban_Rural'].value_counts().plot(kind='bar', ax=axes[0], color=['#4C72B0','#C44E52'], edgecolor='black')
axes[0].set_title('Urban vs Rural School Distribution')
axes[0].set_xlabel('Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

df['District'].value_counts().plot(kind='bar', ax=axes[1], color='teal', edgecolor='black')
axes[1].set_title('Schools per District')
axes[1].set_xlabel('District')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()


# In[28]:


infra_cols = ['Electricity_Available', 'Internet_Available', 'Functional_Toilets', 'Library_Available']

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
labels = ['No', 'Yes']

for ax, col in zip(axes, infra_cols):
    counts = df[col].value_counts().sort_index()
    ax.pie(counts, labels=labels[:len(counts)], autopct='%1.1f%%',
           colors=['#FF6B6B', '#6BCB77'], startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(col.replace('_', ' '))

plt.suptitle('Infrastructure Availability', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# In[29]:


#3.2 Bivariate Analysis
# Scores by Urban vs Rural
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, col in zip(axes, score_cols):
    sns.boxplot(data=df, x='Urban_Rural', y=col, ax=ax, palette='Set2')
    ax.set_title(f'{col.replace("%_","").replace("_Score"," Score")} by Urban/Rural')
    ax.set_xlabel('')

plt.suptitle('Academic Scores: Urban vs Rural', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[30]:


# Scores by District
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, col in zip(axes, score_cols):
    district_means = df.groupby('District')[col].mean().sort_values(ascending=False)
    district_means.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'Avg {col.replace("%_","").replace("_Score"," Score")} by District')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('Average Academic Scores by District', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[31]:


# Parent Literacy Rate vs Math Score
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, score_cols):
    sns.scatterplot(data=df.sample(500, random_state=42),
                    x='Parent_Literacy_Rate', y=col, ax=ax,
                    hue='Urban_Rural', alpha=0.6, palette='Set1')
    # Regression line
    m, b = np.polyfit(df['Parent_Literacy_Rate'], df[col], 1)
    x_line = np.linspace(df['Parent_Literacy_Rate'].min(), df['Parent_Literacy_Rate'].max(), 100)
    ax.plot(x_line, m*x_line + b, 'k--', linewidth=1.5)
    ax.set_title(f'Parent Literacy vs {col.replace("%_","").replace("_Score"," Score")}')

plt.suptitle('Parent Literacy Rate vs Academic Scores', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[32]:


# Teacher Experience vs Scores
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, score_cols):
    sns.scatterplot(data=df.sample(500, random_state=1),
                    x='Avg_Teacher_Experience_Years', y=col,
                    ax=ax, alpha=0.5, color='teal')
    ax.set_title(f'Teacher Exp vs {col.replace("%_","").replace("_Score"," Score")}')

plt.suptitle('Teacher Experience vs Academic Scores', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[33]:


# Infrastructure impact on Math Score (grouped bar)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, col in zip(axes, infra_cols):
    grp = df.groupby(col)[score_cols].mean().rename(columns=lambda x: x.replace('%_','').replace('_Score',''))
    grp.index = ['No', 'Yes']
    grp.plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_title(f'Avg Scores by {col.replace("_"," ")}')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='lower right')

plt.suptitle('Infrastructure Availability vs Academic Performance', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# In[34]:


#3.3 Correlation Analysis

# Full correlation heatmap
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax)
ax.set_title('Correlation Heatmap (All Numeric Features)', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# In[35]:


# Top correlations with Math Score
math_corr = corr['%_Math_Score'].drop('%_Math_Score').sort_values(key=abs, ascending=False)
print('Top correlations with %_Math_Score:')
print(math_corr.to_string())

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#C44E52' if v < 0 else '#55A868' for v in math_corr]
math_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
ax.set_title('Feature Correlations with %_Math_Score', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()


# In[36]:


# Pairplot for key variables
key_vars = ['%_Math_Score', '%_Language_Score', '%_Science_Score',
            'Parent_Literacy_Rate', '%_Students_With_Home_Internet', 'Urban_Rural']

g = sns.pairplot(df[key_vars].sample(300, random_state=7),
                 hue='Urban_Rural', palette='Set1',
                 plot_kws={'alpha': 0.5})
g.fig.suptitle('Pairplot: Key Academic & Socioeconomic Variables', y=1.02, fontsize=13, fontweight='bold')
plt.show()


# In[37]:


#4. Insights
# --- Insight 1: Urban vs Rural performance gap ---
gap = df.groupby('Urban_Rural')[score_cols].mean()
print('=== INSIGHT 1: Urban vs Rural Mean Scores ===')
print(gap.round(2))
print()

urban_math = df[df['Urban_Rural']=='Urban']['%_Math_Score']
rural_math = df[df['Urban_Rural']=='Rural']['%_Math_Score']
t, p = stats.ttest_ind(urban_math, rural_math)
print(f'T-test (Math): t={t:.3f}, p={p:.4f}')
print('→ Statistically significant!' if p < 0.05 else '→ Not significant')


# In[38]:


# --- Insight 2: Parental literacy strongly influences scores ---
r_math, _ = stats.pearsonr(df['Parent_Literacy_Rate'], df['%_Math_Score'])
r_lang, _ = stats.pearsonr(df['Parent_Literacy_Rate'], df['%_Language_Score'])
r_sci,  _ = stats.pearsonr(df['Parent_Literacy_Rate'], df['%_Science_Score'])

print('=== INSIGHT 2: Pearson Correlation – Parent Literacy vs Scores ===')
print(f'  Math    : r = {r_math:.3f}')
print(f'  Language: r = {r_lang:.3f}')
print(f'  Science : r = {r_sci:.3f}')
print('→ Strong positive correlation — parental literacy is a key factor')


# In[39]:


# --- Insight 3: Internet at home vs academic performance ---
df['Home_Internet_Group'] = pd.cut(df['%_Students_With_Home_Internet'],
                                    bins=[0,20,50,100],
                                    labels=['Low (0-20%)', 'Medium (20-50%)', 'High (50%+)'])

internet_scores = df.groupby('Home_Internet_Group')[score_cols].mean().round(2)
print('=== INSIGHT 3: Home Internet Access vs Avg Scores ===')
print(internet_scores)


# In[40]:


# --- Insight 4: Infrastructure schools perform better ---
for infra in infra_cols:
    yes_score = df[df[infra]==1]['%_Math_Score'].mean()
    no_score  = df[df[infra]==0]['%_Math_Score'].mean()
    print(f'{infra:30s} → With: {yes_score:.2f}  |  Without: {no_score:.2f}  |  Diff: {yes_score - no_score:.2f}')

print('\n=== INSIGHT 4: Schools with ALL infrastructure → better scores ===')


# In[41]:


# --- Insight 5: Teacher training matters ---
df['Training_Group'] = pd.cut(df['Teacher_Trainings_Attended'], bins=3, labels=['Low', 'Medium', 'High'])
training_scores = df.groupby('Training_Group')[score_cols].mean().round(2)
print('=== INSIGHT 5: Teacher Training Attended vs Avg Scores ===')
print(training_scores)

r_train, _ = stats.pearsonr(df['Teacher_Trainings_Attended'], df['%_Math_Score'])
print(f'\nCorrelation (trainings vs math score): r = {r_train:.3f}')


# In[42]:


# --- Insight 6: Marginalized students & score impact ---
r_marg, _ = stats.pearsonr(df['%_Marginalized_Students'], df['%_Math_Score'])
print('=== INSIGHT 6: % Marginalized Students vs Math Score ===')
print(f'Correlation: r = {r_marg:.3f}')

fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df.sample(400, random_state=5),
                x='%_Marginalized_Students', y='%_Math_Score',
                hue='Urban_Rural', alpha=0.6, palette='Set2', ax=ax)
ax.set_title('% Marginalized Students vs Math Score', fontweight='bold')
plt.tight_layout()
plt.show()


# In[44]:


# Fix unrealistic values
df = df[(df['%_Math_Score'] <= 100) & (df['Parent_Literacy_Rate'] >= 0) & (df['Parent_Literacy_Rate'] <= 100)]


# In[ ]:


#| No. | Analysis Area                 | Key Result                                          | Conclusion                                                                   |
| --- | ----------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- |
| 1   | Urban vs Rural Performance    | p-value = **0.7727** (> 0.05)                       | No statistically significant difference between Urban and Rural schools      |
| 2   | Subject Correlation           | Math–Language: **0.836**<br>Math–Science: **0.770** | Strong positive relationship → students perform consistently across subjects |
| 3   | Parental Literacy             | Correlation ≈ **0 (0.012, 0.005, 0.004)**           | No meaningful impact on student scores                                       |
| 4   | Home Internet Access          | Low internet group has slightly higher scores       | Internet access does not improve performance (very weak relationship)        |
| 5   | Infrastructure Impact         | Score differences ≈ **±0.5 (very small)**           | Infrastructure has minimal or no effect on scores                            |
| 6   | Teacher Training & Experience | Correlation ≈ **0 (0.007)**                         | No strong linear relationship with performance                               |
| 7   | Marginalized Students         | Correlation ≈ **0 (0.005)**                         | No significant impact on academic scores                                     |
| 8   | Data Quality Issues           | Invalid values (scores >100, literacy <0 or >100)   | Data cleaning needed; results may be slightly affected                       |


# In[45]:


#6.  Feature Engineering + Basic Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


df_model = df.copy()

df_model['Avg_Score'] = df_model[score_cols].mean(axis=1)

df_model['Infrastructure_Score'] = df_model[infra_cols].sum(axis=1)

le = LabelEncoder()
df_model['Urban_Rural_enc'] = le.fit_transform(df_model['Urban_Rural'])
df_model['District_enc']    = le.fit_transform(df_model['District'])

print('Feature engineering complete')
print('New features: Avg_Score, Infrastructure_Score, Urban_Rural_enc, District_enc')


# In[47]:


# --- Basic Random Forest Model to predict Math Score ---
feature_cols = [
    'Teacher_Student_Ratio', 'Avg_Teacher_Experience_Years',
    'Teacher_Trainings_Attended', 'Electricity_Available',
    'Internet_Available', 'Functional_Toilets', 'Library_Available',
    '%_Marginalized_Students', '%_Students_With_Home_Internet',
    'Parent_Literacy_Rate', 'Urban_Rural_enc', 'District_enc',
    'Infrastructure_Score'
]

X = df_model[feature_cols]
y = df_model['%_Math_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f'Random Forest Results:')
print(f'  MAE : {mae:.3f}')
print(f'  R²  : {r2:.3f}')


# In[48]:


# Feature Importance Plot
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
importances.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Feature Importances – Random Forest (Math Score)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()


# In[49]:


# Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.4, color='teal', edgecolors='white', linewidth=0.3)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
ax.set_xlabel('Actual Math Score')
ax.set_ylabel('Predicted Math Score')
ax.set_title(f'Actual vs Predicted  |  R² = {r2:.3f}', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:




