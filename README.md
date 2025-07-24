# مشروع تدريب نماذج أسعار المنازل

هذا المشروع يهدف إلى تدريب نماذج تعلم الآلة المختلفة للتنبؤ بأسعار المنازل باستخدام مجموعة البيانات المرفقة. يتضمن المشروع معالجة البيانات، تدريب خمسة نماذج مختلفة، وتقييم أدائها، بالإضافة إلى إمكانية التنبؤ بأسعار منازل جديدة.

## هيكل المشروع

```
.  
├── data/
│   ├── raw/                 
│   │   └── Housing_Price_Data_clean.csv
│   └── processed/           
│       ├── X_processed.csv
│       └── y.csv
├── notebooks/                
│   ├── logistic_regression_model.ipynb
│   ├── knn_model.ipynb
│   ├── decision_tree_model.ipynb
│   ├── random_forest_model.ipynb
│   └── xgboost_model.ipynb
├── src/                     
│   ├── data_processing.py
│   ├── logistic_regression_model.pkl
│   ├── knn_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
├── utils/                   
├── service/                  
├── doc/                     
│   └── project_documentation.md
└── README.md                 
```

## النماذج المدربة

1.  **الانحدار اللوجستي (Logistic Regression)**: (ملاحظة: تم استخدام Linear Regression كبديل مناسب لبيانات الانحدار)
2.  **الجيران الأقرب (K-Nearest Neighbors)**
3.  **شجرة القرار (Decision Tree)**
4.  **الغابات العشوائية (Random Forest)**
5.  **XGBoost**

## كيفية الاستخدام

1.  **إعداد البيئة:**
    تأكد من تثبيت Python 3.x و pip.
    قم بتثبيت المكتبات المطلوبة:
    ```bash
    pip install pandas scikit-learn xgboost jupyter
    ```

2.  **معالجة البيانات:**
    يتم معالجة البيانات تلقائيًا عند تشغيل المشروع. يمكنك تشغيل السكريبت يدويًا إذا لزم الأمر:
    ```bash
    python src/data_processing.py
    ```
    سيقوم هذا السكريبت بتحميل `Housing_Price_Data_clean.csv` من `data/raw`، ومعالجته، وحفظ البيانات المعالجة في `data/processed`.

3.  **تدريب النماذج وتقييمها:**
    افتح دفاتر Jupyter في مجلد `notebooks` باستخدام الأمر:
    ```bash
    jupyter notebook
    ```
    كل دفتر (`.ipynb`) يحتوي على الكود الخاص بتدريب نموذج معين، تقييمه، وحفظ النموذج المدرب في مجلد `src`.

4.  **التنبؤ بقيم جديدة:**
    داخل كل دفتر Jupyter، ستجد قسمًا مخصصًا للتنبؤ بقيم جديدة. يمكنك تعديل البيانات المدخلة في هذا القسم للحصول على تنبؤات من النموذج المدرب.

## المخرجات

-   **مقاييس الأداء:** كل دفتر Jupyter سيطبع مقاييس الأداء (مثل MSE, RMSE, R-squared) للنموذج المدرب.
-   **النماذج المحفوظة:** سيتم حفظ النماذج المدربة (بصيغة `.pkl`) في مجلد `src`، مما يتيح إعادة استخدامها لاحقًا دون الحاجة لإعادة التدريب.

## التوثيق

يمكن العثور على توثيق إضافي وتفاصيل حول المشروع في ملف `doc/project_documentation.md`.

