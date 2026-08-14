import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.decomposition import PCA

# Configure Matplotlib styling
plt.style.use('default')
sns.set_theme(style='whitegrid')

img_dir = 'images'
os.makedirs(img_dir, exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('E-commerce Dataset.csv')
print("Dataset Shape:", df.shape)

# Clean missing values in critical columns
df['Aging'] = df['Aging'].fillna(df['Aging'].median())
df['Gender'] = df['Gender'].fillna('Unknown')
df['Customer_Login_type'] = df['Customer_Login_type'].fillna('Guest')

missing_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

# Encode Categoricals
le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Product_Category'])

le_priority = LabelEncoder()
df['Priority_Encoded'] = le_priority.fit_transform(df['Order_Priority'])

# Visualizations

# Chart 1: Bar Chart - Sales Revenue by Product Category
plt.figure(figsize=(8.5, 4.5))
cat_sales = df.groupby('Product_Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
sns.barplot(data=cat_sales, x='Product_Category', y='Sales', palette='viridis')
plt.title('Chart 1: Total Sales Revenue by Product Category', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Product Category', fontsize=11)
plt.ylabel('Total Sales ($)', fontsize=11)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '01_bar_sales_by_category.png'), dpi=300)
plt.close()

# Chart 2: Pie Chart - Order Priority Share
plt.figure(figsize=(5.5, 5.5))
priority_counts = df['Order_Priority'].value_counts()
plt.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=140, colors=['#e53e3e', '#dd6b20', '#3182ce', '#38a169'], explode=(0.05, 0, 0, 0))
plt.title('Chart 2: Order Priority Share Breakdown', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '02_pie_order_priority.png'), dpi=300)
plt.close()

# Chart 3: Line Chart - Monthly Sales & Profit Trend (2018)
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Month'] = df['Order_Date'].dt.month
df['Month_Name'] = df['Order_Date'].dt.strftime('%b')

monthly_trend = df.groupby(['Month', 'Month_Name'])[['Sales', 'Profit']].sum().reset_index().sort_values('Month')

plt.figure(figsize=(9, 4.5))
plt.plot(monthly_trend['Month_Name'], monthly_trend['Sales'], marker='o', linewidth=2.5, color='#3182ce', label='Sales ($)')
plt.plot(monthly_trend['Month_Name'], monthly_trend['Profit'], marker='s', linewidth=2.5, color='#38a169', label='Profit ($)')
plt.title('Chart 3: 2018 Monthly Sales & Profit Performance', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Month', fontsize=11)
plt.ylabel('Amount ($)', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '03_line_monthly_trend.png'), dpi=300)
plt.close()

# Chart 4: Histogram - Profit Distribution
plt.figure(figsize=(8, 4.5))
sns.histplot(df['Profit'], kde=True, bins=30, color='#38a169')
plt.title('Chart 4: Transaction Profit Distribution', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Profit ($)', fontsize=11)
plt.ylabel('Transaction Count', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '04_histogram_profit_distribution.png'), dpi=300)
plt.close()

# Chart 5: Box Plot - Profit Outliers by Product Category
plt.figure(figsize=(9, 4.5))
sns.boxplot(data=df, x='Product_Category', y='Profit', palette='Set2')
plt.title('Chart 5: Profit Spread & Outlier Analysis by Category', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Product Category', fontsize=11)
plt.ylabel('Profit ($)', fontsize=11)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '05_boxplot_profit_by_category.png'), dpi=300)
plt.close()

# Chart 6: Scatter Plot - Sales vs Profit by Gender
plt.figure(figsize=(8, 4.5))
sns.scatterplot(data=df.sample(2000, random_state=42), x='Sales', y='Profit', hue='Gender', s=50, alpha=0.7, palette={'Female': '#e53e3e', 'Male': '#3182ce', 'Unknown': '#a0aec0'})
plt.title('Chart 6: Sales vs. Profit Scatter Plot', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Sales ($)', fontsize=11)
plt.ylabel('Profit ($)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '06_scatterplot_sales_vs_profit.png'), dpi=300)
plt.close()

# Chart 7: Heatmap - Correlation Matrix
plt.figure(figsize=(7, 5))
num_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Aging']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Chart 7: E-commerce Numerical Feature Correlations', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '07_heatmap_correlation.png'), dpi=300)
plt.close()

# Chart 8: Count Plot - Payment Method by Device Type
plt.figure(figsize=(8.5, 4.5))
sns.countplot(data=df, x='Payment_method', hue='Device_Type', palette='Accent')
plt.title('Chart 8: Payment Method Choice by Device Type', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Payment Method', fontsize=11)
plt.ylabel('Transaction Count', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '08_countplot_payment_by_device.png'), dpi=300)
plt.close()

# DATA MINING TASK 1: CLASSIFICATION (Predict Order Priority)
X_clf = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Category_Encoded', 'Gender_Encoded']]
y_clf = df['Priority_Encoded']

# Sample 10,000 for fast ML training
df_sample = df.sample(10000, random_state=42)
X_clf_smp = df_sample[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Category_Encoded', 'Gender_Encoded']]
y_clf_smp = df_sample['Priority_Encoded']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf_smp, y_clf_smp, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c_s = scaler_c.fit_transform(X_train_c)
X_test_c_s = scaler_c.transform(X_test_c)

dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_c_s, y_train_c)
y_pred_dt = dt.predict(X_test_c_s)

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
rf.fit(X_train_c_s, y_train_c)
y_pred_rf = rf.predict(X_test_c_s)

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
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le_priority.classes_, yticklabels=le_priority.classes_)
plt.title('Chart 9: Random Forest Order Priority Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Priority', fontsize=11)
plt.ylabel('Actual Priority', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '09_confusion_matrix.png'), dpi=300)
plt.close()

# DATA MINING TASK 2: CLUSTERING (K-Means)
X_cluster = df_sample[['Sales', 'Quantity', 'Profit', 'Shipping_Cost']]
scaler_cl = StandardScaler()
X_cluster_s = scaler_cl.fit_transform(X_cluster)

wcss = []
sil_scores = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_s)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_s, labels))

# Plot Elbow & Silhouette
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(K_range, wcss, 'bo-', linewidth=2, label='Inertia (WCSS)')
ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia (WCSS)', color='b', fontsize=11)
ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, 'rs--', linewidth=2, label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r', fontsize=11)
plt.title('Chart 10: K-Means Elbow Curve & Silhouette Scores', fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
plt.savefig(os.path.join(img_dir, '10_kmeans_elbow.png'), dpi=300)
plt.close()

# Fit K=4
kmeans_opt = KMeans(n_clusters=4, random_state=42, n_init=10)
df_sample['Cluster'] = kmeans_opt.fit_predict(X_cluster_s)

# PCA 2D Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_s)
df_sample['PCA1'] = X_pca[:, 0]
df_sample['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_sample, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=60, alpha=0.8)
plt.title('Chart 11: E-commerce Order Clusters (2D PCA Projection)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '11_kmeans_clusters_pca.png'), dpi=300)
plt.close()

# DATA MINING TASK 3: PREDICTION / REGRESSION (Predict Profit)
X_reg = df_sample[['Sales', 'Quantity', 'Discount', 'Shipping_Cost']]
y_reg = df_sample['Profit']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

reg_r2 = r2_score(y_test_r, y_pred_r)
reg_mae = mean_absolute_error(y_test_r, y_pred_r)
reg_mse = mean_squared_error(y_test_r, y_pred_r)
reg_rmse = np.sqrt(reg_mse)

plt.figure(figsize=(8, 5))
plt.scatter(y_test_r, y_pred_r, color='#38a169', alpha=0.6, edgecolors='k')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', linewidth=2)
plt.title('Chart 12: Multiple Linear Regression - Actual vs Predicted Profit', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Actual Profit ($)', fontsize=11)
plt.ylabel('Predicted Profit ($)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, '12_regression_actual_vs_pred.png'), dpi=300)
plt.close()

# Save metrics summary file for markdown generation
summary_text = f"""--- PREPROCESSING & EDA METRICS ---
Dataset Shape: {df.shape}
Null Count: {missing_counts.to_dict()}
Duplicate Rows: {duplicate_count}

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
Silhouette Scores: {dict(zip(K_range, [round(s, 4) for s in sil_scores]))}
Cluster Profiles:
{df_sample.groupby('Cluster')[['Sales', 'Quantity', 'Profit', 'Shipping_Cost']].mean().to_dict()}

--- REGRESSION RESULTS ---
Multiple Linear Regression:
  R2 Score: {reg_r2:.4f}
  MAE: {reg_mae:.4f}
  MSE: {reg_mse:.4f}
  RMSE: {reg_rmse:.4f}
  Coefficients: {dict(zip(X_reg.columns, [round(c, 4) for c in reg.coef_]))}
  Intercept: {reg.intercept_:.4f}
"""

with open('analysis_summary.txt', 'w') as f:
    f.write(summary_text)

print("Rakib Analysis Assets & Metrics Generated Successfully!")
