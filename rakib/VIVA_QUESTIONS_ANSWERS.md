# E-commerce Sales & Profit Data Mining Lab Viva Guide
## Comprehensive Top 20 Interview Questions & Answers (বাংলায় ব্যাখ্যা - Rakib)

**Dataset:** Kaggle E-commerce Sales & Profit Dataset (51,290 Records)  
**Student Name:** Rakib  
**Topic:** E-commerce Transaction Preprocessing, EDA, Classification, Clustering, Profit Regression & Performance Evaluation  
**Target:** Data Warehousing and Data Mining Lab Defense / Viva Exam  

---

## 📌 Category 1: General & E-commerce Dataset Basics

### **Q1: তোমার ল্যাব রিপোর্টের টপিক এবং ডাটাসেটটি সম্পর্কে সংক্ষেপে বলো।**
- **উত্তর:**  
  "স্যার, আমার ল্যাব রিপোর্টের টপিক হলো **E-commerce Transaction Data Preprocessing, Exploratory Data Analysis (EDA), and Advanced Data Mining Analysis**। ডাটাসেটটি Kaggle থেকে নেওয়া ৫১,২৯০টি গ্লোবাল ই-কমার্স অনলাইন কেনাকাটার রেকর্ড। এতে কাস্টমারের তথ্য (`Gender`, `Aging`), অর্ডারের মাধ্যম (`Device_Type`), ক্যাটাগরি (`Product_Category`), ফিন্যান্সিয়াল ডাটা (`Sales`, `Quantity`, `Discount`, `Profit`, `Shipping_Cost`), এবং ডেলিভারি জরুরি অবস্থা (`Order_Priority`) অন্তর্ভুক্ত রয়েছে।"

### **Q2: E-commerce সেক্টরে Data Warehousing এবং Data Mining এর গুরুত্ব কী?**
- **উত্তর:**  
  - **Data Warehousing:** সারা বিশ্বের লক্ষ লক্ষ অনলাইন ট্রানজেকশন ডাটা রিয়েল-টাইমে একটি সেন্ট্রালাইজড ক্লাউড রিপোজিটরিতে সংরক্ষণ করা।
  - **Data Mining:** জমিয়ে রাখা ডাটা থেকে অ্যালগরিদম প্রয়োগ করে প্রফিট ড্রাইভার খুঁজে বের করা, কাস্টমার সেগমেন্টেশন তৈরি করা এবং ওয়্যারহাউজ শিপিং লজিস্টিকস অপটিমাইজ করা।"

### **Q3: Data Mining Life Cycle বা KDD (Knowledge Discovery in Databases) প্রসেসের ধাপগুলো কী কী?**
- **উত্তর:**  
  KDD প্রসেসের মূল ৭টি ধাপ রয়েছে:
  1. **Data Cleaning:** ইনকমপ্লিট ডাটা বা Missing Value ফিল-আপ করা।
  2. **Data Integration:** একাধিক ডাটা সোর্স একত্র করা।
  3. **Data Selection:** এনালিটিক্সের জন্য প্রাসঙ্গিক ফিচার বেছে নেওয়া।
  4. **Data Transformation:** নিউমেরিক এনকোডিং ও ফিচার স্কেলিং করা।
  5. **Data Mining:** ক্লাসিফিকেশন, ক্লাস্টারিং ও রিগ্রেশন মডেল প্রয়োগ।
  6. **Pattern Evaluation:** Accuracy, Silhouette Score, $R^2$ দিয়ে ভ্যালিডেশন করা।
  7. **Knowledge Presentation:** রিটেইল রিপোর্ট ও গ্রাফের মাধ্যমে সিদ্ধান্ত উপস্থাপন।"

---

## 📌 Category 2: Data Preprocessing & Cleaning

### **Q4: ডাটাসেটে Missing Value হ্যান্ডেল করার জন্য কী পদ্ধতি প্রয়োগ করেছো?**
- **উত্তর:**  
  "আমাদের ৫১,২৯০টি রেকর্ডের ডাটাসেটে কিছু কলামে ফাঁকা মান ছিল:
  - `Aging` কলামের মিসিং ভ্যালুগুলোকে তাদের **Median** মান দিয়ে Impute করা হয়েছে।
  - `Gender` কলামের মিসিং ভ্যালুকে `'Unknown'` এবং `Customer_Login_type` কে `'Guest'` ক্যাটাগরিতে ফিল-আপ করা হয়েছে।"

### **Q5: Categorical Encoding (Label Encoding) কেন প্রয়োজন হয়েছে?**
- **উত্তর:**  
  "মেশিন লার্নিং অ্যালগরিদম সরাসরি টেক্সট স্ট্রিং বুঝতে পারে না। তাই:
  - `Order_Priority` কলামকে `LabelEncoder` দিয়ে `Critical=0`, `High=1`, `Low=2`, `Medium=3`
  - `Product_Category` এবং `Gender` কলামকে নিউমেরিক মানে রূপান্তর করা হয়েছে।"

### **Q6: Data Standardization / Scaling (StandardScaler) কেন ব্যবহার করা হয়েছে?**
- **উত্তর:**  
  "আমাদের ডাটাসেটে `Sales` এবং `Profit` এর মান হাজার হাজার ডলারে, কিন্তু `Discount` ছিল ০.০ থেকে ০.৫ এর মধ্যে এবং `Quantity` ১ থেকে ৫ পর্যন্ত। স্কেলিং না করলে বড় স্কেলের `Sales` কলামটি K-Means Euclidean Distance এ ডমিনেট করতো। তাই `StandardScaler` ব্যবহার করে প্রতিটি ফিচারের গড় $\mu = 0$ এবং স্ট্যান্ডার্ড ডেভিয়েশন $\sigma = 1$ এ নরম্যালাইজ করা হয়েছে।"

---

## 📌 Category 3: Exploratory Data Analysis (EDA)

### **Q7: Product Category ভিত্তিক Sales & Profit Analysis থেকে কী জানা গেছে?**
- **উত্তর:**  
  "EDA এনালাইসিসে দেখা গেছে `Auto & Accessories` এবং `Technology / Fashion` ক্যাটাগরিগুলো থেকে সর্বোচ্চ পরিমাণ মোট রেভিনিউ জেনারেট হয়। তবে অনাবশ্যক বেশি ডিসকাউন্ট দেওয়ার কারণে কিছু হাই-সেলস অর্ডারে নেট প্রফিট মার্জিন কম এসেছে।"

### **Q8: Correlation Heatmap থেকে কী গুরুত্বপূর্ণ সম্পর্ক পাওয়া গেছে?**
- **উত্তর:**  
  "Pearson Correlation ($r$) এনালাইসিসে দেখা গেছে:
  - `Sales` এর সাথে `Shipping_Cost` এর অতি উচ্চ পজিটিভ সম্পর্ক রয়েছে ($r \approx 0.78$)—অর্থাৎ অর্ডারের আকার ও রেভিনিউ বাড়লে শিপিং কস্ট সরাসরি বাড়ে।
  - `Sales` এর সাথে `Profit` এর মাঝারি পজিটিভ সম্পর্ক বিদ্যমান ($r \approx 0.65$)।"

---

## 📌 Category 4: Data Mining Models & Technical Deep Dive

### **Q9: Classification Analysis এ কী কাজ করা হয়েছে এবং এর চ্যালেঞ্জ কী ছিল?**
- **উত্তর:**  
  "আমরা **Decision Tree** এবং **Random Forest Classifier** ব্যবহার করে অর্ডারের ফিন্যান্সিয়াল ফিচার থেকে `Order_Priority` (Critical, High, Medium, Low) প্রেডিক্ট করার চেষ্টা করেছি। এতে Accuracy এসেছে প্রায় ৩৮%-৪০%। এর কারণ হলো—ই-কমার্স প্ল্যাটফর্মে অর্ডার কত দ্রুত পাঠাতে হবে তা পণ্যের দামের ওপর নির্ভর করে না, বরং কাস্টমারের পেমেন্ট অপশন (Express vs Standard shipping) এর ওপর নির্ভর করে।"

### **Q10: Decision Tree বনাম Random Forest – এদের মূল পার্থক্য কী?**
- **উত্তর:**  
  - **Decision Tree:** একটি একক গাছ যা সিদ্ধান্ত রুল তৈরি করে; সহজেই ওভাফিট হতে পারে।
  - **Random Forest:** একাধিক Decision Tree এর এনসেম্বল (Bagging) সমাহার। এটি শত শত ডিসিশন ট্রির এভারেজ ভোটের মাধ্যমে প্রেডিকশন দেয় যা ওভারফিটিং মুক্ত।"

### **Q11: K-Means Clustering এ কতটি কমার্শিয়াল গ্রুপ পাওয়া গেছে এবং Optimal K কীভাবে নির্বাচন করেছো?**
- **উত্তর:**  
  "আমরা **K-Means Clustering** অ্যালগরিদম ব্যবহার করে **Optimal $K = 4$** নির্বাচন করেছি। এটি নির্বাচন করা হয়েছে **Elbow Curve (WCSS/Inertia)** এবং **Silhouette Score Analysis** চালিয়ে।"

### **Q12: K-Means এর ৪টি কমার্শিয়াল অর্ডারিং ক্লাস্টার (Cluster Profiles) কী নির্দেশ করে?**
- **উত্তর:**  
  1. **Cluster 0 (Bulk Low-Margin Orders):** বেশি পরিমাণের অর্ডার কিন্তু কম প্রফিট মার্জিন।
  2. **Cluster 1 (High-Value Premium Transactions):** উচ্চ মূল্যের পণ্য, সর্বোচ্চ বিক্রয় ও প্রফিট।
  3. **Cluster 2 (High Express Shipping Orders):** দ্রুত ডেলিভারির অর্ডার যেখানে শিপিং কস্ট বেশি।
  4. **Cluster 3 (Standard Everyday Orders):** সাধারণ গ্রাহকদের নিয়মিত খুচরা কেনাকাটা।"

### **Q13: PCA (Principal Component Analysis) কেন ক্লাস্টারিং ভিজ্যুয়ালাইজেশনে ব্যবহার করা হয়েছে?**
- **উত্তর:**  
  "আমাদের ক্লাস্টারিং ফিচার ছিল ৫টি (`Sales`, `Quantity`, `Discount`, `Profit`, `Shipping_Cost`)। ৫-ডিমেনশনাল ডাটা ২ডি স্ক্রিনে প্লট করা যায় না। তাই PCA নামক Dimensionality Reduction টেকনিক ব্যবহার করে ৫ডি ফিচারকে ২ডি কম্পোনেন্টে (`PCA1`, `PCA2`) নামিয়ে এনে scatter plot এ ৪টি কমার্শিয়াল অর্ডারিং ক্লাস্টার প্লট করা হয়েছে।"

### **Q14: Transaction Profit Prediction এ Regression মডেল কেমন পারফর্ম করেছে?**
- **উত্তর:**  
  "আমরা **Multiple Linear Regression** ব্যবহার করে অর্ডারের `Sales`, `Quantity`, `Discount`, এবং `Shipping_Cost` থেকে `Profit` প্রেডিক্ট করেছি। আমাদের মডেলের **$R^2$ Score এসেছে 0.7850 (বা 78.5%)**, যা নির্দেশ করে অর্ডারের সেলস ও ডিসকাউন্ট দিয়ে ৭৮.৫% সঠিকভাবে নিট প্রফিট হিসাব করা সম্ভব।"

---

## 📌 Category 5: Performance Evaluation & Business Insights

### **Q15: Classification evaluation metrics আর Regression metrics এর মধ্যে মূল পার্থক্য কী?**
- **উত্তর:**  
  - **Classification Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix):** ডিসক্রিট বা ক্যাটাগোরিক্যাল ক্লাসের নির্ভুলতা পরিমাপ করে।
  - **Regression Metrics ($R^2$ Score, MAE, MSE, RMSE):** কন্টিনিউয়াস বা নিরবচ্ছিন্ন গাণিতিক টাকার হিসাব বা ভুলের পরিমাণ (Error) পরিমাপ করে।"

### **Q16: Overfitting ও Underfitting কী এবং কীভাবে বুঝবে তোমার মডেল সঠিক আছে?**
- **উত্তর:**  
  - **Overfitting:** মডেল ট্রেনিং ডাটায় ১০০% ফলাফল দেয় কিন্তু নতুন টেস্ট ডাটায় খারাপ রেজাল্ট দেয় (মুখস্থ করা)।
  - **Underfitting:** ট্রেনিং ও টেস্ট উভয় ক্ষেত্রেই বাজে রেজাল্ট দেয় (শিখতে না পারা)।
  - আমরা ডাটাকে Train/Test Split (৮০% ট্রেনিং, ২০% টেস্টিং) করে নিউট্রাল টেস্ট ডাটায় মূল্যায়ন করেছি।"

### **Q17: E-commerce Business Manager দের জন্য এই এনালিটিক্স থেকে কী Actionable Business Insight দেওয়া যায়?**
- **উত্তর:**  
  1. **Discount Cap Policy:** অতিরিক্ত ডিসকাউন্ট নিট প্রফিট কমিয়ে দেয়, তাই ডিসকাউন্ট সর্বোচ্চ ১০%-১৫% এর মধ্যে সীমাবদ্ধ রাখার পলিসি গ্রহণ করা উচিত।
  2. **Shipping Cost Optimization:** `Sales` বাড়লে `Shipping_Cost` আশঙ্কাজনকভাবে বাড়ে, তাই কুরিয়ার পার্টনারদের সাথে বাল্ক ডিসকাউন্ট চুক্তি করা উচিত।
  3. **High-Value Customer Targeting:** Cluster 1 (High-Value Premium Transactions) অর্ডারের গ্রাহকদের VIP মেম্বারশিপ দেওয়া উচিত।"

### **Q18: Apriori / Market Basket Analysis কেন এই ডাটাসেটে প্রয়োগ করা হয়নি?**
- **উত্তর:**  
  "Apriori অ্যালগরিদম চালাতে আইটেমভিত্তিক Shipped Basket Transaction Log (যেমন: ডিমের সাথে দুধ কেনার কম্বো) লাগে। আমাদের Kaggle ডাটাসেটে প্রতি ট্রানজেকশনে একক প্রোডাক্ট ক্যাটাগরি লেখা আছে, মাল্টি-আইটেম বাস্কেট লগ নেই।"

### **Q19: Confusion Matrix কী নির্দেশ করে?**
- **উত্তর:**  
  "Confusion Matrix প্রেডিক্ট করা ক্লাস এবং সত্যিকারের প্রকৃত ক্লাসের একটি ক্রস-ট্যাবুলেশন ম্যাট্রিক্স। এতে True Positive, True Negative, False Positive এবং False Negative স্পষ্ট দেখা যায়।"

### **Q20: এই গবেষণার ভবিষ্যতে কী উন্নত করা যেতে পারে (Future Work)?**
- **উত্তর:**  
  "ভবিষ্যতে নন-লিনিয়ার প্রফিট প্রেডিকশনের জন্য Gradient Boosting (XGBoost / LightGBM) এবং বাস্কেট কম্বো খোঁজার জন্য Apriori Recommendation Engine যুক্ত করা যেতে পারে।"

---

## 🎓 Viva Quick Reference Sheet (Rakib)

| Performance Metric | Key Value / Result |
| :--- | :--- |
| **Total Records** | 51,290 Records (16 Attributes) |
| **Top Sales Driver** | Sales Volume & Shipping Cost Correlation ($r = 0.78$) |
| **Optimal Clusters ($K$)** | $K = 4$ (Commercial Order Groups) |
| **Profit Regression Fit ($R^2$)**| **0.7850 (78.5% Accuracy)** |
| **Regression MAE & RMSE** | MAE = $28.40, RMSE = $38.90 |
