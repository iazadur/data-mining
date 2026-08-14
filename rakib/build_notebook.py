import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# E-commerce Data Warehousing & Data Mining Lab Report
## Analysis of Kaggle E-commerce Sales & Profit Dataset

**Experiment Title:** E-commerce Sales Preprocessing, EDA, Classification, Clustering & Profit Prediction  
**Student Name:** Rakib  
**Course:** Data Warehousing and Data Mining Lab  
**Dataset:** [Kaggle E-commerce Dataset](https://www.kaggle.com/datasets/mervemenekse/ecommerce-dataset) (51,290 Records)  

---
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Experiment Title
**E-commerce Sales & Profit Data Preprocessing, Exploratory Data Analysis (EDA), and Classification, Clustering & Regression Modeling**

## 2. Objective
1. Clean e-commerce transaction logs (51,290 rows), handle missing values, encode categoricals, and scale numerical attributes.
2. Perform Exploratory Data Analysis (EDA) with 8 visualizations analyzing `Sales`, `Profit`, `Discount`, `Shipping_Cost`, and `Order_Priority`.
3. Implement Data Mining algorithms:
   - **Classification:** Decision Tree & Random Forest predicting `Order_Priority`.
   - **Clustering:** K-Means Clustering ($K=4$) segmenting customer order behaviors.
   - **Predictive Regression:** Multiple Linear Regression predicting transaction `Profit`.
4. Evaluate models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Silhouette Score, and $R^2$.
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Data Preprocessing & EDA Setup"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('E-commerce Dataset.csv')
print("Dataset Shape:", df.shape)

df['Aging'] = df['Aging'].fillna(df['Aging'].median())
df['Gender'] = df['Gender'].fillna('Unknown')
df['Customer_Login_type'] = df['Customer_Login_type'].fillna('Guest')

le_priority = LabelEncoder()
df['Priority_Encoded'] = le_priority.fit_transform(df['Order_Priority'])

print("\nMissing Values Count:")
print(df.isnull().sum().sum())
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Classification & Clustering Analysis"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score

df_sample = df.sample(10000, random_state=42)

le_cat = LabelEncoder()
le_gen = LabelEncoder()
df_sample['Category_Encoded'] = le_cat.fit_transform(df_sample['Product_Category'])
df_sample['Gender_Encoded'] = le_gen.fit_transform(df_sample['Gender'])

X_clf = df_sample[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Cost', 'Category_Encoded', 'Gender_Encoded']]
y_clf = df_sample['Priority_Encoded']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_c)
X_test_s = scaler.transform(X_test_c)

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
rf.fit(X_train_s, y_train_c)
y_pred_rf = rf.predict(X_test_s)

print("Random Forest Priority Accuracy:", accuracy_score(y_test_c, y_pred_rf))

X_reg = df_sample[['Sales', 'Quantity', 'Discount', 'Shipping_Cost']]
y_reg = df_sample['Profit']
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_tr_r, y_tr_r)
print("Profit Prediction Linear Regression R2 Score:", r2_score(y_te_r, reg.predict(X_te_r)))
"""))

nb['cells'] = cells

with open('lab_report_ecommerce.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Rakib Jupyter Notebook lab_report_ecommerce.ipynb generated successfully!")
