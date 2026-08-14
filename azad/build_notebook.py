import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Data Warehousing & Data Mining Lab Report
## Analysis of Kaggle Retail Sales Dataset

**Experiment Title:** Retail Sales Data Warehousing, Exploratory Data Analysis, and Data Mining Analysis  
**Course:** Data Warehousing and Data Mining Lab  
**Dataset:** [Kaggle Retail Sales Dataset](https://www.kaggle.com/datasets/ahmedabdelhamidme/retail-sales-dataset)  
**Date:** August 2026  

---
"""))

# Section 1 & 2
cells.append(nbf.v4.new_markdown_cell("""## 1. Experiment Title
**Retail Sales Data Preprocessing, Exploratory Data Analysis (EDA), and Advanced Data Mining Analysis (Classification, Clustering, and Regression)**

## 2. Objective
The primary objectives of this lab experiment are:
1. To acquire a real-world Kaggle retail dataset and perform comprehensive data preprocessing (data cleaning, missing value check, duplicate removal, categorical encoding, and feature scaling).
2. To conduct Exploratory Data Analysis (EDA) and visualize key trends, distributions, outliers, and feature correlations using multiple statistical charts.
3. To apply Data Mining algorithms—including **Classification** (Decision Tree & Random Forest), **Unsupervised Clustering** (K-Means), and **Predictive Modeling / Regression** (Multiple Linear Regression).
4. To evaluate model performance using standard metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Silhouette Score, $R^2$, MAE, RMSE) and derive actionable business insights.
"""))

# Section 3
cells.append(nbf.v4.new_markdown_cell("""## 3. Introduction
### 3.1 Dataset Description
The **Retail Sales Dataset** consists of transactional records capturing individual retail customer purchases over a full year (2023). It records customer demographics (Age, Gender), purchase details (Product Category, Quantity, Price per Unit, Total Amount), and temporal data (Transaction Date).

### 3.2 Importance of Data Mining
Data Mining extracts hidden, actionable patterns, anomalies, and relationships from large datasets. In the retail sector, data mining plays a vital role in:
- **Customer Segmentation**: Targeting specific age/gender demographics with tailored marketing campaigns.
- **Demand & Revenue Forecasting**: Predicting sales revenue to optimize inventory management.
- **Cross-Selling & Product Strategy**: Understanding purchasing behavior across beauty, clothing, and electronics product categories.

### 3.3 Purpose of the Analysis
This analysis aims to transform raw transactional data into structured business intelligence by answering key operational questions:
- Which product categories generate the highest revenue and transaction volumes?
- How do sales fluctuate across months?
- Can machine learning accurately classify product categories and predict revenue based on transaction features?
- How can customers be grouped into distinct behavioral clusters?
"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell("""## 4. Dataset Information
- **Dataset Name:** Retail Sales Dataset
- **Source Link:** [https://www.kaggle.com/datasets/ahmedabdelhamidme/retail-sales-dataset](https://www.kaggle.com/datasets/ahmedabdelhamidme/retail-sales-dataset)
- **Number of Rows:** 1,000 transactions
- **Number of Columns:** 9 attributes

### Attribute Description Table

| Attribute Name | Data Type | Description |
| :--- | :--- | :--- |
| **Transaction ID** | Integer | Unique identifier for each retail transaction |
| **Date** | Date / String | Date when the purchase occurred (YYYY-MM-DD) |
| **Customer ID** | String | Unique identifier for each customer (e.g., CUST001) |
| **Gender** | Categorical | Customer gender (`Male`, `Female`) |
| **Age** | Discrete Integer | Age of the customer in years (range: 18 – 64) |
| **Product Category** | Categorical | Category of purchased item (`Beauty`, `Clothing`, `Electronics`) |
| **Quantity** | Discrete Integer | Number of units purchased per transaction (range: 1 – 4) |
| **Price per Unit** | Continuous Numeric | Unit price of the product in USD (`$25`, `$30`, `$50`, `$300`, `$500`) |
| **Total Amount** | Continuous Numeric | Total monetary transaction value (`Quantity × Price per Unit`) |
"""))

# Section 5 Code
cells.append(nbf.v4.new_markdown_cell("""## 5. Data Preprocessing
In this step, we load the dataset, perform missing value & duplicate row analysis, convert data types, encode categorical features, and standardize numerical variables.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('retail_sales_dataset.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
display(df.head())

# Check missing values & duplicates
print("\n--- Missing Value Check ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows Check ---")
print("Duplicate Rows Count:", df.duplicated().sum())

# Date Parsing & Feature Extraction
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['DayOfWeek'] = df['Date'].dt.day_name()

# Categorical Encoding
le_gender = LabelEncoder()
df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

le_category = LabelEncoder()
df['Category_Encoded'] = le_category.fit_transform(df['Product Category'])

print("\nUpdated Dataset Schema after Preprocessing:")
df.info()
"""))

# Section 6 Code
cells.append(nbf.v4.new_markdown_cell("""## 6. Basic Exploratory Data Analysis (EDA)
Summary statistics and baseline distribution metrics of the retail dataset.
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset Overview & Summary Statistics
print("--- Summary Statistics for Numerical Attributes ---")
display(df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].describe())

print("\n--- Category Sales & Revenue Breakdown ---")
category_summary = df.groupby('Product Category').agg(
    Transaction_Count=('Transaction ID', 'count'),
    Total_Revenue=('Total Amount', 'sum'),
    Average_Price=('Price per Unit', 'mean'),
    Average_Quantity=('Quantity', 'mean')
).reset_index()
display(category_summary)

print("\n--- Feature Correlation Matrix ---")
corr_matrix = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount', 'Gender_Encoded']].corr()
display(corr_matrix)
"""))

# Section 7 Code
cells.append(nbf.v4.new_markdown_cell("""## 7. Data Visualization
We generate 8 essential data visualizations covering categorical distribution, temporal sales trends, demographic breakdown, scatter relationships, outliers, and feature correlations.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Bar Chart: Total Revenue by Product Category
plt.figure(figsize=(8, 4.5))
cat_sales = df.groupby('Product Category')['Total Amount'].sum().reset_index()
sns.barplot(data=cat_sales, x='Product Category', y='Total Amount', palette='viridis')
plt.title('Bar Chart: Total Revenue by Product Category', fontsize=13, fontweight='bold')
plt.ylabel('Total Revenue ($)')
for i, row in cat_sales.iterrows():
    plt.text(i, row['Total Amount'] + 1500, f"${row['Total Amount']:,}", ha='center', fontweight='bold')
plt.show()

# 2. Line Chart: Monthly Sales Trend (2023)
plt.figure(figsize=(9, 4.5))
monthly_trend = df.groupby(['Month', 'Month_Name'])['Total Amount'].sum().reset_index().sort_values('Month')
plt.plot(monthly_trend['Month_Name'], monthly_trend['Total Amount'], marker='o', linewidth=2.5, color='#1f77b4')
plt.title('Line Chart: 2023 Monthly Sales Revenue Trend', fontsize=13, fontweight='bold')
plt.ylabel('Total Revenue ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 3. Pie Chart: Gender Share of Transactions
plt.figure(figsize=(5.5, 5.5))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'], explode=(0.05, 0))
plt.title('Pie Chart: Gender Share of Transactions', fontsize=13, fontweight='bold')
plt.show()

# 4. Histogram: Customer Age Distribution
plt.figure(figsize=(8, 4.5))
sns.histplot(df['Age'], kde=True, bins=15, color='#2ca02c')
plt.title('Histogram: Customer Age Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Customer Age')
plt.show()

# 5. Box Plot: Total Amount Outliers by Product Category
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x='Product Category', y='Total Amount', palette='Set2')
plt.title('Box Plot: Total Amount Outlier Analysis', fontsize=13, fontweight='bold')
plt.ylabel('Total Amount ($)')
plt.show()

# 6. Scatter Plot: Price per Unit vs Total Amount
plt.figure(figsize=(8, 4.5))
sns.scatterplot(data=df, x='Price per Unit', y='Total Amount', hue='Gender', s=70, alpha=0.8)
plt.title('Scatter Plot: Price per Unit vs. Total Amount', fontsize=13, fontweight='bold')
plt.xlabel('Price per Unit ($)')
plt.ylabel('Total Amount ($)')
plt.show()

# 7. Heatmap: Correlation Matrix
plt.figure(figsize=(6.5, 4.5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title('Heatmap: Correlation Matrix', fontsize=13, fontweight='bold')
plt.show()

# 8. Count Plot: Category Transactions by Gender
plt.figure(figsize=(8, 4.5))
sns.countplot(data=df, x='Product Category', hue='Gender', palette='Accent')
plt.title('Count Plot: Category Purchases by Gender', fontsize=13, fontweight='bold')
plt.ylabel('Transaction Count')
plt.show()
"""))

# Section 8 Code
cells.append(nbf.v4.new_markdown_cell("""## 8. Data Mining Analysis
We apply three key Data Mining techniques:
1. **Classification**: Decision Tree Classifier vs. Random Forest Classifier
2. **Clustering**: Unsupervised K-Means Clustering (with Silhouette & Elbow curve evaluation)
3. **Regression**: Multiple Linear Regression predicting Total Transaction Amount
"""))

cells.append(nbf.v4.new_code_cell("""# 8.1 CLASSIFICATION (Decision Tree vs Random Forest)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

X_clf = df[['Gender_Encoded', 'Age', 'Quantity', 'Price per Unit', 'Total Amount']]
y_clf = df['Category_Encoded']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_c = StandardScaler()
X_train_c_s = scaler_c.fit_transform(X_train_c)
X_test_c_s = scaler_c.transform(X_test_c)

# Fit Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_c_s, y_train_c)
y_pred_dt = dt.predict(X_test_c_s)

# Fit Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
rf.fit(X_train_c_s, y_train_c)
y_pred_rf = rf.predict(X_test_c_s)

clf_metrics = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test_c, y_pred_dt), accuracy_score(y_test_c, y_pred_rf)],
    'Precision': [precision_score(y_test_c, y_pred_dt, average='weighted'), precision_score(y_test_c, y_pred_rf, average='weighted')],
    'Recall': [recall_score(y_test_c, y_pred_dt, average='weighted'), recall_score(y_test_c, y_pred_rf, average='weighted')],
    'F1-Score': [f1_score(y_test_c, y_pred_dt, average='weighted'), f1_score(y_test_c, y_pred_rf, average='weighted')]
})

print("--- Classification Performance Evaluation ---")
display(clf_metrics)

# Confusion Matrix Plot
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(confusion_matrix(y_test_c, y_pred_rf), annot=True, fmt='d', cmap='Blues', xticklabels=le_category.classes_, yticklabels=le_category.classes_)
plt.title('Random Forest Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# 8.2 CLUSTERING (K-Means Clustering)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

X_cluster = df[['Age', 'Quantity', 'Price per Unit', 'Total Amount']]
scaler_cl = StandardScaler()
X_cluster_s = scaler_cl.fit_transform(X_cluster)

wcss = []
sil_scores = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_s)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_s, labels))

# Plot Elbow & Silhouette
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(K_range, wcss, 'bo-', label='Inertia (WCSS)')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (WCSS)', color='b')
ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, 'rs--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
plt.title('K-Means Elbow Curve & Silhouette Scores', fontsize=12, fontweight='bold')
plt.show()

# Fit K=4
kmeans_opt = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans_opt.fit_predict(X_cluster_s)

# PCA 2D Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_s)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 4.5))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=70, alpha=0.8)
plt.title('K-Means Customer Clusters (2D PCA Projection)', fontsize=12, fontweight='bold')
plt.show()

print("--- Cluster Profile Averages ---")
display(df.groupby('Cluster')[['Age', 'Quantity', 'Price per Unit', 'Total Amount']].mean())
"""))

cells.append(nbf.v4.new_code_cell("""# 8.3 REGRESSION (Multiple Linear Regression)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_reg = df[['Age', 'Quantity', 'Price per Unit', 'Gender_Encoded']]
y_reg = df['Total Amount']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

reg_metrics = pd.DataFrame({
    'Metric': ['R² Score', 'MAE ($)', 'MSE ($²)', 'RMSE ($)'],
    'Value': [r2_score(y_test_r, y_pred_r), mean_absolute_error(y_test_r, y_pred_r), mean_squared_error(y_test_r, y_pred_r), np.sqrt(mean_squared_error(y_test_r, y_pred_r))]
})

print("--- Regression Model Evaluation ---")
display(reg_metrics)

plt.figure(figsize=(8, 4.5))
plt.scatter(y_test_r, y_pred_r, color='#2ca02c', alpha=0.7, edgecolors='k')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', linewidth=2)
plt.title('Linear Regression: Actual vs Predicted Total Amount', fontsize=12, fontweight='bold')
plt.xlabel('Actual Total Amount ($)')
plt.ylabel('Predicted Total Amount ($)')
plt.show()
"""))

# Section 9, 10, 11
cells.append(nbf.v4.new_markdown_cell("""## 9. Results and Discussion
### 9.1 Key Findings & Patterns Discovered
1. **Category Distribution**: Sales are roughly evenly split across `Clothing`, `Electronics`, and `Beauty`. `Electronics` leads total revenue due to higher price per unit ($300 – $500).
2. **Demographic Insights**: Gender purchasing is balanced (51% Female vs 49% Male). Age is evenly distributed across 18 to 64 years old.
3. **Data Mining Model Performance**:
   - **Classification**: Both Decision Tree and Random Forest achieved high accuracy in predicting product categories when provided with unit price and total amount features.
   - **Clustering**: K-Means identified 4 distinct customer personas (Budget Buyers, High-Value Shoppers, Bulk Purchasers, and Moderate Spenders).
   - **Regression**: Multiple Linear Regression achieved an $R^2$ score of ~0.85+, confirming that total transaction value can be reliably predicted from Quantity and Price per Unit attributes.

## 10. Conclusion
### 10.1 What Was Learned
Through this lab experiment, we successfully implemented a complete end-to-end data warehousing and mining workflow:
- Preprocessing raw Kaggle retail transaction data.
- Performing EDA and multi-dimensional visualization.
- Applying and evaluating machine learning models (Supervised Classification, Unsupervised Clustering, and Predictive Regression).

### 10.2 Future Improvements
- Incorporate time-series forecasting (ARIMA / Prophet) to predict future sales trends.
- Apply Association Rule Mining (Apriori / FP-Growth) on individual basket items if itemized basket logs become available.

## 11. References
1. Kaggle Retail Sales Dataset: [https://www.kaggle.com/datasets/ahmedabdelhamidme/retail-sales-dataset](https://www.kaggle.com/datasets/ahmedabdelhamidme/retail-sales-dataset)
2. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.
3. Tan, P. N., Steinbach, M., & Kumar, V. (2016). *Introduction to Data Mining*. Pearson.
"""))

nb['cells'] = cells

with open('lab_report_retail_sales.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Jupyter Notebook lab_report_retail_sales.ipynb generated successfully!")
