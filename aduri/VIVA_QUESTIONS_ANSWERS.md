# Breast Cancer Diagnostic Data Mining Lab Viva Guide
## Top 20 Custom Interview Questions & Answers (Aduri - Healthcare Domain)

**Dataset:** Kaggle Breast Cancer Wisconsin Diagnostic Dataset (1,138 Augmented Records)  
**Student Name:** Aduri  

---

## 📌 Category 1: General & Dataset Basics
### Q1: তোমার ল্যাব রিপোর্টের টপিক এবং ডাটাসেট সম্পর্কে বলো।
- **উত্তর:** "স্যার, আমার ল্যাব রিপোর্টের টপিক হলো **Breast Cancer Diagnostic Data Preprocessing, EDA, and Data Mining Analysis**। ডাটাসেটটি Kaggle-এর Breast Cancer Wisconsin Diagnostic Dataset। এতে ১,১৩৮টি টিউমার সেলের রেকর্ড রয়েছে এবং প্রতিটি টিউমারের ৩০টি জিওমেট্রিক ও টেক্সচার ফিচার এবং ডায়াগনোসিস টার্গেট কলাম (`Malignant` vs `Benign`) রয়েছে।"

### Q2: মেডিক্যাল ডাটা মাইনিং কেন গুরুত্বপূর্ণ?
- **উত্তর:** "মেডিক্যাল ডাটাতে মেশিন লার্নিং বা ডাটা মাইনিং প্রয়োগ করে ক্যানসার টিউমার ম্যালিগন্যান্ট (ক্ষতিকর) নাকি বিনাইন (ক্ষতিহীন) তা দ্রুত এবং ৯৯%+ নির্ভুলতার সাথে শনাক্ত করা যায়, যা ডাক্তারের সিদ্ধান্ত গ্রহণে সাহায্য করে।"

---

## 📌 Category 2: Data Preprocessing & ML Models
### Q3: Classification এ কোন কোন মডেল ব্যবহার করেছো এবং রেজাল্ট কেমন?
- **উত্তর:** "আমরা Decision Tree, Random Forest এবং Logistic Regression ব্যবহার করেছি। **Random Forest Classifier সর্বোচ্চ 99.12% Accuracy** এবং 1.00 Precision অর্জন করেছে।"

### Q4: K-Means Clustering এ কয়টি কাস্টমার/পেশেন্ট গ্রুপ পাওয়া গেছে?
- **উত্তর:** "Optimal $K = 2$ নির্ধারণ করা হয়েছে, যা সরাসরি পেশেন্টদের Malignant ও Benign দুটি মেডিক্যাল গ্রুপের সাথে মিলে যায় (Silhouette Score = 0.4125)।"

### Q5: Regression এ কী ফলাফল এসেছে?
- **উত্তর:** "Multiple Linear Regression দিয়ে টিউমারের `perimeter_mean` প্রেডিক্ট করে **$R^2 = 0.9962$** পাওয়া গেছে।"
