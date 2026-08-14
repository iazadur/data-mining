# Data Warehousing & Data Mining Lab Viva Guide
## Comprehensive Top 20 Interview Questions & Answers (বাংলায় ব্যাখ্যা)

**Dataset:** Kaggle Retail Sales Dataset (1,000 Records)  
**Topic:** Data Preprocessing, EDA, Classification, Clustering, Regression & Evaluation  
**Target:** Data Warehousing and Data Mining Lab Defense / Viva Exam  

---

## 📌 Category 1: General & Dataset Basics

### **Q1: তোমার ল্যাব রিপোর্টের টপিক এবং ডাটাসেটটি সম্পর্কে সংক্ষেপে বলো।**
- **উত্তর:**  
  "স্যার, আমার ল্যাব রিপোর্টের টপিক হলো **Retail Sales Data Preprocessing, Exploratory Data Analysis (EDA), and Advanced Data Mining Analysis**। ডাটাসেটটি Kaggle থেকে নেওয়া ১,০০০টি ট্রানজেকশনের রেকর্ড। এতে ২০২৩ সালের কাস্টমারদের বয়সের ডেমোগ্রাফিক (`Age`, `Gender`), প্রোডাক্ট ক্যাটাগরি (`Beauty`, `Clothing`, `Electronics`), ইউনিট প্রাইস, কোয়ান্টিটি এবং মোট খরচের (`Total Amount`) তথ্য রয়েছে।"

### **Q2: Data Warehousing এবং Data Mining এর মূল পার্থক্য কী?**
- **উত্তর:**  
  - **Data Warehousing:** একাধিক হেটেরোজিনিয়াস উৎস থেকে আসা বিশাল পরিমাণ ডাটাকে একটি সেন্ট্রালাইজড রিপোজিটরিতে ক্লিন ও স্ট্রাকচারড করে জমিয়ে রাখার প্রক্রিয়া (Data Storage & Integration)।
  - **Data Mining:** সেই জমিয়ে রাখা ডাটা থেকে মেশিন লার্নিং ও স্ট্যাটিস্টিক্যাল অ্যালগরিদম ব্যবহার করে অনাবিষ্কৃত প্যাটার্ন, ট্রেন্ড বা কাস্টমার সেগমেন্টেশন খুঁজে বের করার প্রক্রিয়া (Knowledge Discovery in Databases - KDD)।

### **Q3: Data Mining Life Cycle বা KDD (Knowledge Discovery in Databases) প্রসেসের ধাপগুলো কী কী?**
- **উত্তর:**  
  KDD প্রসেসের মূল ৭টি ধাপ রয়েছে:
  1. **Data Cleaning:** নয়েজ ও ইনকন্সিস্টেন্ট ডাটা রিমুভ করা।
  2. **Data Integration:** একাধিক ডাটা সোর্সকে একসাথে যুক্ত করা।
  3. **Data Selection:** এনালাইসিসের জন্য প্রয়োজনীয় ডাটা সিলেক্ট করা।
  4. **Data Transformation:** ডাটাকে এনকোডিং বা স্কেলিং এর মাধ্যমে প্রস্তুত করা।
  5. **Data Mining:** ক্লাসিফিকেশন, ক্লাস্টারিং বা রিগ্রেশন অ্যালগরিদম প্রয়োগ করা।
  6. **Pattern Evaluation:** পাওয়া প্যাটার্ন বা ইভালুয়েশন মেট্রিক্স যাচাই করা।
  7. **Knowledge Presentation:** চার্ট ও রিপোর্টের মাধ্যমে প্রেজেন্ট করা।

---

## 📌 Category 2: Data Preprocessing & Cleaning

### **Q4: ডাটাসেটে Missing Value বা Duplicate Data ছিল কি? থাকলে কীভাবে হ্যান্ডেল করেছো?**
- **উত্তর:**  
  "স্যার, ডাটাসেট ফিল্টারিং ও কোড চেকিং করে দেখা গেছে এতে **০টি Null/Missing Value** রয়েছে এবং **০টি Duplicate রেকর্ড** পাওয়া গেছে (১০০% ক্লিন ও ইউনিক ডাটা)।"

### **Q5: Categorical Encoding (Label Encoding) কেন ব্যবহার করা হয়েছে?**
- **উত্তর:**  
  "মেশিন লার্নিং অ্যালগরিদম (যেমন: Decision Tree, Regression) সরাসরি টেক্সট/স্ট্রিং ক্যাটাগোরিক্যাল কলাম নিয়ে কাজ করতে পারে না, নিউমেরিক ভ্যালুর প্রয়োজন হয়। তাই:
  - `Gender` কলামকে `LabelEncoder` দিয়ে `Female = 0`, `Male = 1`
  - `Product Category` কলামকে `Beauty = 0`, `Clothing = 1`, `Electronics = 2` এ রূপান্তরিত করা হয়েছে।"

### **Q6: Data Standardization / Scaling (StandardScaler) কেন দরকার হলো?**
- **উত্তর:**  
  "আমাদের ডাটাসেটের ফিচারগুলোর স্কেল আলাদা ছিল—যেমন `Age` ১৮–৬৪ বছর, কিন্তু `Price per Unit` $২৫–$৫০০ এবং `Total Amount` $২৫–$২,০০০। স্কেল আলাদা থাকলে K-Means Clustering বা Regression এ বড় স্কেলের কলামটি ডমিনেট করবে (Euclidean distance পক্ষপাতদুষ্ট হবে)। তাই `StandardScaler` ব্যবহার করে প্রতিটি ফিচারের গড়াংশ $\mu = 0$ এবং স্ট্যান্ডার্ড ডেভিয়েশন $\sigma = 1$ এ নরম্যালাইজ করা হয়েছে।"

---

## 📌 Category 3: Exploratory Data Analysis (EDA)

### **Q7: Correlation Analysis থেকে কী গুরুত্বপূর্ণ স্ট্যাটিস্টিক্যাল সম্পর্ক পাওয়া গেছে?**
- **উত্তর:**  
  "Pearson Correlation Coefficient ($r$) এনালাইসিস করে দেখা গেছে:
  - `Price per Unit` এর সাথে `Total Amount` এর অতি উচ্চ পজিটিভ সম্পর্ক বিদ্যমান ($r = 0.852$)।
  - `Quantity` এর সাথে `Total Amount` এর মাঝারি পজিটিভ সম্পর্ক রয়েছে ($r = 0.374$)।
  - কাস্টমারের বয়সের (`Age`) সাথে মোট খরচের কোনো বিশেষ প্রভাব বা সম্পর্ক নেই ($r \approx -0.060$)।"

### **Q8: Box Plot এ কোনো Outlier পাওয়া গেছে কি?**
- **উত্তর:**  
  "Box Plot এর মাধ্যমে দেখা গেছে কিছু ট্রানজেকশন অ্যামাউন্ট $২,০০০ পর্যন্ত উঠেছে। তবে এগুলো ডাটা এন্ট্রি এরর নয়, এগুলো বৈধ ট্রানজেকশন (Legitimate High-Value Purchases); কারণ $৫০০ মূল্যের ১টি ইলেকট্রনিক্স আইটেম ৪টি কিনলে মোট মূল্য $২,০০০ হয়।"

---

## 📌 Category 4: Data Mining Models & Technical Deep Dive

### **Q9: Classification এ কোন কোন অ্যালগরিদম ব্যবহার করেছো এবং রেজাল্ট কেমন?**
- **উত্তর:**  
  "আমরা **Decision Tree Classifier** ($max\_depth = 5$) এবং **Random Forest Classifier** ($n\_estimators = 100$) ব্যবহার করে কাস্টমারের বয়স, জেন্ডার, কোয়ান্টিটি ও ইউনিট প্রাইস থেকে `Product Category` প্রেডিক্ট করার চেষ্টা করেছি। এতে Accuracy এসেছে প্রায় ৩০%–৩৩%। এর বাস্তবসম্মত কারণ হলো—সব ক্যাটাগরিতেই (Beauty, Clothing, Electronics) সমপরিমাণ দামের এবং পরিমাণের পণ্য বিক্রি হয়, তাই শুধু ডেমোগ্রাফিক ডাটা দিয়ে ক্যাটাগরি সম্পূর্ণ আলাদা করা সম্ভব নয়।"

### **Q10: Decision Tree বনাম Random Forest – এদের মূল পার্থক্য কী এবং কেন Random Forest সাধারণত ভালো কাজ করে?**
- **উত্তর:**  
  - **Decision Tree:** একটি একক সিদ্ধান্ত নেওয়ার গাছ যা পুরো ডাটার ওপর রুলস তৈরি করে। এটি সহজেই ট্রেনিং ডাটায় Overfit হতে পারে।
  - **Random Forest:** এটি একটি Ensemble Learning (Bagging) পদ্ধতি। এটি শত শত Decision Tree তৈরি করে এবং তাদের ভোটের গড়াংশের ভিত্তিতে ফাইনাল সিদ্ধান্ত দেয়। এতে Overfitting এর ঝুঁকি কম থাকে এবং সাধারণ ডাটায় ভালো ফলাফল দেয়।

### **Q11: Clustering এ কোন অ্যালগরিদম ব্যবহার করা হয়েছে এবং Optimal K কীভাবে নির্বাচন করেছো?**
- **উত্তর:**  
  "আমরা Unsupervised **K-Means Clustering** অ্যালগরিদম ব্যবহার করেছি। ক্লাস্টার সংখ্যা $K$ চুজ করার জন্য **Elbow Method (WCSS/Inertia)** এবং **Silhouette Score Analysis** চালিয়ে **Optimal $K = 4$** নির্বাচন করেছি।"

### **Q12: K-Means এ Elbow Method এবং Silhouette Score কীভাবে কাজ করে?**
- **উত্তর:**  
  - **Elbow Method:** এটি ক্লাস্টারের মধ্যকার দূরত্বের স্কয়ারের যোগফল (Inertia বা WCSS) প্লট করে। যেখানে গ্রাফের ঢাল হঠাৎ বাঁক নেয় (Elbow point), সেটি Optimal $K$ দেয়।
  - **Silhouette Score (-1 to +1):** এটি মাপে ক্লাস্টারের পয়েন্টগুলো নিজের ক্লাস্টারের কতটা কাছে এবং পাশের ক্লাস্টার থেকে কতটা দূরে। স্কোর ১ এর কাছাকাছি থাকা মানে ভালো ক্লাস্টারিং।"

### **Q13: K-Means এর ৪টি কাস্টমার সেগমেন্ট (Cluster Profiles) কী প্রকাশ করে?**
- **উত্তর:**  
  1. **Cluster 0 (Budget Single Buyers):** গড় বয়স ৪২.১ বছর, দামি আইটেম অল্প পরিমাণে কেনেন।
  2. **Cluster 1 (Moderate Spenders):** গড় বয়স ৪১.৮ বছর, কম দামি পণ্য বেশি পরিমাণে কেনেন (গড় ৩.৫৫ units)।
  3. **Cluster 2 (Low-Cost Shoppers):** কম দামে ১-২টি পণ্য কেনেন (গড় খরচ $৫২.৫৬)।
  4. **Cluster 3 (High-Value VIP Shoppers):** সবচেয়ে মূল্যবান কাস্টমার গ্রুপ, যারা প্রতি কেনাকাটায় গড়ে **$১,৩৬৫.৫৮** খরচ করেন।

### **Q14: PCA (Principal Component Analysis) কী এবং ক্লাস্টারিং ভিজ্যুয়ালাইজেশনে কেন PCA ব্যবহার করেছো?**
- **উত্তর:**  
  "PCA হলো একটি **Dimensionality Reduction** টেকনিক। আমাদের কাস্টমার ক্লাস্টারিং ফিচার ছিল ৪টি (`Age`, `Quantity`, `Price per Unit`, `Total Amount`)। ৪-ডিমেনশনাল ডাটা ২ডি স্ক্রিনে প্লট করা সম্ভব নয়। তাই PCA ব্যবহার করে ডাটার ইনফরমেশন/ভ্যারিয়েন্স অক্ষুণ্ণ রেখে ৪ডি ফিচারকে ২ডি কম্পোনেন্টে (`PCA1`, `PCA2`) নামিয়ে এনে scatter plot এ ক্লাস্টারগুলো সুন্দরভাবে ভিজ্যুয়ালাইজ করেছি।"

### **Q15: Predictive Regression এ কী ফলাফল পাওয়া গেছে?**
- **উত্তর:**  
  "আমরা **Multiple Linear Regression** দিয়ে কাস্টমারের বয়স, কোয়ান্টিটি, ইউনিট প্রাইস ও জেন্ডার ব্যবহার করে `Total Amount` প্রেডিক্ট করেছি। আমাদের মডেলের **$R^2$ Score এসেছে 0.8569 (বা 85.7%)**, যা প্রমাণ করে মডেলটি ৮৫.৭% সঠিকভাবে রেভিনিউ হিসাব করতে পারে।"

#### **Regression ইকুয়েশন:**
$$\text{Total Amount} = -406.37 + (179.58 \times \text{Quantity}) + (2.49 \times \text{Price per Unit}) - (0.90 \times \text{Age}) + (11.88 \times \text{Gender})$$

### **Q16: Logistic Regression বনাম Multiple Linear Regression – দুটোর পার্থক্য কী?**
- **উত্তর:**  
  - **Multiple Linear Regression:** ব্যবহার করা হয় Continuous Numeric Target (যেমন: মোট টাকার পরিমাণ) প্রেডিক্ট করতে।
  - **Logistic Regression:** ব্যবহার করা হয় Categorical/Binary Class (যেমন: Yes/No, Spam/Ham) এর সম্ভাবনা বা প্রবাবিলিটি প্রেডিক্ট করতে।

### **Q17: Overfitting ও Underfitting কী এবং কীভাবে বুঝবে তোমার মডেল ওভারফিট বা আন্ডারফিট করেছে?**
- **উত্তর:**  
  - **Overfitting:** মডেল ট্রেনিং ডাটায় ১০০% সঠিক পারফর্ম করে কিন্তু আনসিন টেস্ট ডাটায় খারাপ রেজাল্ট দেয় (মডেল ডাটা মুখস্থ করে ফেলে)।
  - **Underfitting:** মডেল ট্রেনিং ও টেস্ট উভয় ডাটায় বাজে পারফর্ম করে (মডেল প্যাটার্ন শিখতেই পারে না)।
  - আমরা ডাটাকে Train/Test স্প্লিট করে (৮০% ট্রেনিং, ২০% টেস্টিং) নিরপেক্ষ টেস্ট ডাটায় ইভালুয়েট করেছি।

### **Q18: Apriori / Association Rule Mining কেন এই ডাটাসেটে প্রয়োগ করা হয়নি?**
- **উত্তর:**  
  "Apriori অ্যালগরিদম চালাতে আইটেমভিত্তিক Market Basket Transaction Log (যেমন: ক্রেতা পাউরুটি কিনলে সাথে মাখন কেনার রুলস) প্রয়োজন হয়। আমাদের এই Kaggle ডাটাসেটের প্রতি লাইনে একক প্রোডাক্ট ক্যাটাগরি লেখা আছে, কোনো মাল্টি-আইটেম শপিং বাস্কেট লগ নেই। ভবিষ্যতে আইটেমাইজড ট্রানজেকশন লগ পাওয়া গেলে Apriori বা FP-Growth অ্যালগরিদম প্রয়োগ করা যাবে।"

---

## 📌 Category 5: Performance Evaluation & Business Insights

### **Q19: Classification evaluation metrics আর Regression metrics এর মধ্যে মূল পার্থক্য কী?**
- **উত্তর:**  
  - **Classification Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix):** ডিসক্রিট ক্যাটাগরি বা শ্রেণী বিভাগের নির্ভুলতা পরিমাপ করে।
  - **Regression Metrics ($R^2$ Score, MAE, MSE, RMSE):** কন্টিনিউয়াস সংখ্যা প্রেডিকশনে প্রেডিক্টেড মান ও প্রকৃত মানের মধ্যকার গাণিতিক পার্থক্য বা ত্রুটি (Error) পরিমাপ করে।

### **Q20: এই ডাটা এনালাইসিস থেকে রিটেইল শপের বিজনেস ম্যানেজারদের কী Actionable Business Advice দেবে?**
- **উত্তর:**  
  1. **VIP Customer Retention:** Cluster 3 (High-Value VIP Shoppers) কাস্টমারদের জন্য স্পেশাল লয়্যালটি প্রোগ্রাম ও পারসোনালাইজড অফার দেওয়া উচিত, কারণ তারা সবচেয়ে বেশি রেভিনিউ আনে।
  2. **Seasonal Inventory Planning:** মে ও অক্টোবর মাসে সেলস সবচেয়ে বেশি পিক নেয়, তাই এই সময় শপে ইনভেন্টরি স্টক বাড়াতে হবে।
  3. **Product Bundling:** Electronics ও Clothing প্রোডাক্ট ক্যাটাগরি মোট বিক্রির প্রায় ৬৮% জেনারেট করে, তাই এদের সাথে Beauty প্রোডাক্ট কসমেটিক্স বান্ডিল ডিসকাউন্টে দেওয়া যেতে পারে।

---

## 🎓 Viva Key Takeaways Sheet

| Concept | Key Value / Technique Used |
| :--- | :--- |
| **Dataset Size** | 1,000 Records, 9 Primary Columns |
| **Data Quality** | 0 Missing Values, 0 Duplicates |
| **Top Revenue Category** | Electronics ($156,905 gross revenue) |
| **Key Revenue Correlates** | Price per Unit ($r = 0.852$) & Quantity ($r = 0.374$) |
| **Dimensionality Reduction**| PCA (4D Features reduced to 2D for Visualization) |
| **Optimal Clusters ($K$)** | $K = 4$ (Validated via Elbow Method & Silhouette Analysis) |
| **Regression Fit ($R^2$)** | **0.8569** (MAE = $172.95, RMSE = $204.64) |
