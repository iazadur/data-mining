# Data Warehousing & Data Mining Lab Projects

This repository contains comprehensive Data Warehousing, Exploratory Data Analysis (EDA), and Machine Learning / Data Mining lab reports, Jupyter Notebooks, datasets, and presentation assets for three student projects:

---

## 📁 Repository Structure

```
├── azad/                           # Azad's Retail Sales Lab Project
│   ├── retail_sales_dataset.csv     # Kaggle Retail Sales Dataset (1,000 records)
│   ├── LAB_REPORT.md               # 11-Section Lab Report
│   ├── LAB_REPORT.pdf               # Rendered PDF Report
│   ├── LAB_REPORT.docx              # Editable MS Word Document
│   ├── lab_report_retail_sales.ipynb # Executed Jupyter Notebook
│   ├── VIVA_QUESTIONS_ANSWERS.md    # Top 20 Viva Q&A Guide
│   ├── generate_report_assets.py    # Data Analysis & Plotting Script
│   └── images/                     # 12 High-Res Charts & ML Output Plots
│
├── aduri/                          # Aduri's Healthcare / Cancer Lab Project
│   ├── breast-cancer.csv           # Kaggle Breast Cancer Dataset (1,138 records)
│   ├── LAB_REPORT.md               # Healthcare Lab Report
│   ├── LAB_REPORT.pdf               # Rendered PDF Report
│   ├── LAB_REPORT.docx              # Editable MS Word Document
│   ├── lab_report_breast_cancer.ipynb # Executed Jupyter Notebook
│   ├── VIVA_QUESTIONS_ANSWERS.md    # Healthcare ML Viva Guide
│   ├── generate_report_assets.py    # Healthcare Analysis Script
│   └── images/                     # 12 Medical Diagnostic Charts
│
└── rakib/                          # Rakib's E-commerce Lab Project
    ├── E-commerce Dataset.csv      # Kaggle E-commerce Sales Dataset (51,290 records)
    ├── LAB_REPORT.md               # E-commerce Lab Report
    ├── LAB_REPORT.pdf               # Rendered PDF Report
    ├── LAB_REPORT.docx              # Editable MS Word Document
    ├── lab_report_ecommerce.ipynb   # Executed Jupyter Notebook
    ├── VIVA_QUESTIONS_ANSWERS.md    # E-commerce ML Viva Guide
    ├── generate_report_assets.py    # Sales & Profit Analysis Script
    └── images/                     # 12 E-commerce Sales Charts
```

---

## 📊 Summary of Projects & Datasets

| Student | Domain | Dataset Source | Record Count | Machine Learning Tasks |
| :--- | :--- | :--- | :--- | :--- |
| **Azad** | Retail Sales | [Kaggle Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset) | 1,000 | Decision Tree, Random Forest, K-Means ($K=4$), Linear Regression ($R^2 = 0.8569$) |
| **Aduri** | Healthcare | [Kaggle Dataset](https://www.kaggle.com/datasets/mehmetisik/breast-cancercsv) | 1,138 | Random Forest (**99.12% Accuracy**), K-Means ($K=2$), Multiple Linear Regression ($R^2 = 0.9962$) |
| **Rakib** | E-commerce | [Kaggle Dataset](https://www.kaggle.com/datasets/mervemenekse/ecommerce-dataset) | 51,290 | Decision Tree, Random Forest, K-Means ($K=4$), Profit Linear Regression ($R^2 = 0.7850$) |

---

## 🛠️ Execution & Build Scripts

Each project folder contains standalone utility scripts:
- `generate_report_assets.py`: Runs data preprocessing, EDA, ML models (Classification, K-Means Clustering with PCA, Linear Regression), and saves PNG plots into `images/`.
- `build_notebook.py`: Builds executable `.ipynb` Jupyter Notebook files.
- `convert_to_pdf.py`: Renders `LAB_REPORT.md` into styled PDF format.
- `convert_to_docx.py`: Converts `LAB_REPORT.md` into formatted MS Word `.docx` documents.
