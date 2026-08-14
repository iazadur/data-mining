import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Configure Matplotlib styling
plt.style.use('default')
sns.set_theme(style='whitegrid')

img_dir = 'images'
os.makedirs(img_dir, exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('retail_sales_dataset.csv')
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Data Cleaning & Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['DayOfWeek'] = df['Date'].dt.day_name()

missing_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print("Missing values per column:\n", missing_counts)
print("Duplicate rows count:", duplicate_count)

# Summary statistics
num_summary = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].describe()
print("\nNumerical Summary:\n", num_summary)

# Visualizations

# Chart 1: Bar Chart - Total Sales Revenue by Product Category
plt.figure(figsize=(8, 5))
cat_sales = df.groupby('Product Category')['Total Amount'].sum().reset_index()
sns.barplot(data=cat_sales, x='Product Category', y='Total Amount', palette='viridis')
plt.title('Chart 1: Total Sales Revenue by Product Category', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
for index, row in cat_sales.iterrows():
    plt.text(index, row['Total Amount'] + 2000, f"${row['Total Amount']:,}", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '01_bar_sales_by_category.png'), dpi=300)
plt.close()

# Chart 2: Line Chart - Monthly Sales Trend (2023)
plt.figure(figsize=(10, 5))
monthly_trend = df.groupby(['Month', 'Month_Name'])['Total Amount'].sum().reset_index().sort_values('Month')
plt.plot(monthly_trend['Month_Name'], monthly_trend['Total Amount'], marker='o', linewidth=2.5, color='#1f77b4')
plt.title('Chart 2: Monthly Sales Trend (2023)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
for i, txt in enumerate(monthly_trend['Total Amount']):
    plt.annotate(f"${txt:,}", (monthly_trend['Month_Name'].iloc[i], monthly_trend['Total Amount'].iloc[i]+500), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '02_line_monthly_sales.png'), dpi=300)
plt.close()

# Chart 3: Pie Chart - Gender Share of Transactions
plt.figure(figsize=(6, 6))
gender_counts = df['Gender'].value_counts()
colors = ['#ff9999', '#66b3ff']
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0), textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('Chart 3: Gender Distribution of Transactions', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '03_pie_gender_distribution.png'), dpi=300)
plt.close()

# Chart 4: Histogram - Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, bins=15, color='#2ca02c')
plt.title('Chart 4: Customer Age Distribution', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency (Customer Count)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '04_histogram_age_distribution.png'), dpi=300)
plt.close()

# Chart 5: Box Plot - Total Amount by Product Category
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Product Category', y='Total Amount', palette='Set2')
plt.title('Chart 5: Outlier Detection - Total Amount by Product Category', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Total Amount ($)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '05_boxplot_total_amount_by_category.png'), dpi=300)
plt.close()

# Chart 6: Scatter Plot - Price per Unit vs Total Amount by Gender
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Price per Unit', y='Total Amount', hue='Gender', style='Gender', s=70, alpha=0.8, palette={'Female': '#e377c2', 'Male': '#1f77b4'})
plt.title('Chart 6: Price per Unit vs. Total Amount by Gender', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Price per Unit ($)', fontsize=12)
plt.ylabel('Total Amount ($)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '06_scatterplot_price_vs_total.png'), dpi=300)
plt.close()

# Chart 7: Heatmap - Correlation Matrix
plt.figure(figsize=(7, 5))
corr = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, cbar=True)
plt.title('Chart 7: Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '07_heatmap_correlation.png'), dpi=300)
plt.close()

# Chart 8: Count Plot - Product Category split by Gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Product Category', hue='Gender', palette='Accent')
plt.title('Chart 8: Transaction Count per Category by Gender', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Transaction Count', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '08_countplot_category_by_gender.png'), dpi=300)
plt.close()

# DATA MINING TASK 1: CLASSIFICATION
# Target: Product Category based on Gender, Age, Quantity, Price per Unit, Total Amount
le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Product Category'])

X_clf = df[['Gender_Encoded', 'Age', 'Quantity', 'Price per Unit', 'Total Amount']]
y_clf = df['Category_Encoded']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Decision Tree Model
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_clf.fit(X_train_c_scaled, y_train_c)
y_pred_dt = dt_clf.predict(X_test_c_scaled)

# Random Forest Model
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
rf_clf.fit(X_train_c_scaled, y_train_c)
y_pred_rf = rf_clf.predict(X_test_c_scaled)

# Metrics
dt_acc = accuracy_score(y_test_c, y_pred_dt)
dt_prec = precision_score(y_test_c, y_pred_dt, average='weighted')
dt_rec = recall_score(y_test_c, y_pred_dt, average='weighted')
dt_f1 = f1_score(y_test_c, y_pred_dt, average='weighted')

rf_acc = accuracy_score(y_test_c, y_pred_rf)
rf_prec = precision_score(y_test_c, y_pred_rf, average='weighted')
rf_rec = recall_score(y_test_c, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test_c, y_pred_rf, average='weighted')

cm_rf = confusion_matrix(y_test_c, y_pred_rf)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le_category.classes_, yticklabels=le_category.classes_)
plt.title('Chart 9: Random Forest Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Category', fontsize=12)
plt.ylabel('Actual Category', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '09_confusion_matrix.png'), dpi=300)
plt.close()

# DATA MINING TASK 2: CLUSTERING (K-Means)
X_cluster = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']]
scaler_cl = StandardScaler()
X_cluster_scaled = scaler_cl.fit_transform(X_cluster)

wcss = []
silhouette_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_cluster_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, cluster_labels))

# Plot Elbow and Silhouette
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia / WCSS', color='#1f77b4', fontsize=12)
ax1.plot(K_range, wcss, color='#1f77b4', marker='o', linewidth=2, label='Inertia (WCSS)')
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='#ff7f0e', fontsize=12)
ax2.plot(K_range, silhouette_scores, color='#ff7f0e', marker='s', linestyle='--', linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

plt.title('Chart 10: K-Means Elbow Curve & Silhouette Scores', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig(os.path.join(img_dir, '10_kmeans_elbow.png'), dpi=300)
plt.close()

# Fit Optimal K=4
kmeans_opt = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans_opt.fit_predict(X_cluster_scaled)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=70, alpha=0.8)
plt.title('Chart 11: K-Means Customer Clusters (PCA Projection)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '11_kmeans_clusters_pca.png'), dpi=300)
plt.close()

# DATA MINING TASK 3: PREDICTION / REGRESSION
# Multiple Linear Regression to predict Total Amount from Age, Quantity, Price per Unit
X_reg = df[['Age', 'Quantity', 'Price per Unit', 'Gender_Encoded']]
y_reg = df['Total Amount']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

reg_r2 = r2_score(y_test_r, y_pred_r)
reg_mae = mean_absolute_error(y_test_r, y_pred_r)
reg_mse = mean_squared_error(y_test_r, y_pred_r)
reg_rmse = np.sqrt(reg_mse)

plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_r, color='#2ca02c', alpha=0.7, edgecolors='k')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', linewidth=2)
plt.title('Chart 12: Linear Regression - Actual vs Predicted Total Amount', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Total Amount ($)', fontsize=12)
plt.ylabel('Predicted Total Amount ($)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '12_regression_actual_vs_pred.png'), dpi=300)
plt.close()

# Save metrics summary file for markdown generation
summary_text = f"""--- PREPROCESSING & EDA METRICS ---
Shape: {df.shape}
Null Count: {missing_counts.to_dict()}
Duplicate Rows: {duplicate_count}
Gender Distribution:
{df['Gender'].value_counts().to_dict()}

Category Sales:
{cat_sales.to_dict(orient='records')}

--- CLASSIFICATION RESULTS ---
Decision Tree:
  Accuracy: {dt_acc:.4f}
  Precision: {dt_prec:.4f}
  Recall: {dt_rec:.4f}
  F1-Score: {dt_f1:.4f}

Random Forest:
  Accuracy: {rf_acc:.4f}
  Precision: {rf_prec:.4f}
  Recall: {rf_rec:.4f}
  F1-Score: {rf_f1:.4f}

--- CLUSTERING RESULTS ---
Optimal K: 4
Silhouette Scores: {dict(zip(K_range, [round(s, 4) for s in silhouette_scores]))}
Cluster Profiles:
{df.groupby('Cluster')[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].mean().to_dict()}

--- REGRESSION RESULTS ---
Linear Regression:
  R2 Score: {reg_r2:.4f}
  MAE: {reg_mae:.4f}
  MSE: {reg_mse:.4f}
  RMSE: {reg_rmse:.4f}
  Coefficients: {dict(zip(X_reg.columns, [round(c, 4) for c in reg_model.coef_]))}
  Intercept: {reg_model.intercept_:.4f}
"""

with open('analysis_summary.txt', 'w') as f:
    f.write(summary_text)

print("Analysis & Visualizations Completed Successfully!")
