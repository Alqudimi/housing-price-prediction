# توثيق مشروع تدريب نماذج أسعار المنازل

## مقدمة

يهدف هذا المشروع إلى بناء نماذج تعلم الآلة المختلفة للتنبؤ بأسعار المنازل باستخدام مجموعة من الخصائص المختلفة للمنازل. تم تصميم المشروع ليكون منظمًا وقابلًا للصيانة، مع فصل واضح بين مراحل معالجة البيانات، تدريب النماذج، والتوثيق.

## البيانات

### البيانات الأصلية
-   **الملف:** `data/raw/Housing_Price_Data_clean.csv`
-   **الوصف:** يحتوي على بيانات أسعار المنازل مع خصائص مختلفة مثل المساحة، عدد الغرف، الحمامات، وغيرها.

### الخصائص في البيانات:
-   `price`: سعر المنزل (المتغير التابع)
-   `area`: مساحة المنزل
-   `bedrooms`: عدد غرف النوم
-   `bathrooms`: عدد الحمامات
-   `stories`: عدد الطوابق
-   `mainroad`: هل المنزل على طريق رئيسي (1/0)
-   `guestroom`: هل يوجد غرفة ضيوف (1/0)
-   `basement`: هل يوجد قبو (1/0)
-   `hotwaterheating`: هل يوجد تدفئة مياه ساخنة (1/0)
-   `airconditioning`: هل يوجد تكييف (1/0)
-   `parking`: عدد أماكن الوقوف
-   `prefarea`: هل المنطقة مفضلة (1/0)
-   `furnishingstatus`: حالة الأثاث (0: غير مفروش، 1: مفروش جزئيًا، 2: مفروش بالكامل)

### معالجة البيانات
يتم معالجة البيانات باستخدام `src/data_processing.py` والذي يقوم بـ:
1. تحميل البيانات الأصلية
2. فصل المتغيرات المستقلة (X) عن المتغير التابع (y)
3. تحديد المتغيرات الفئوية والرقمية
4. تطبيق التطبيع (StandardScaler) على المتغيرات الرقمية
5. تطبيق One-Hot Encoding على المتغيرات الفئوية
6. حفظ البيانات المعالجة في `data/processed/`

## النماذج المستخدمة

### 1. الانحدار الخطي (Linear Regression)
-   **الملف:** `notebooks/logistic_regression_model.ipynb`
-   **الوصف:** تم استخدام الانحدار الخطي كبديل للانحدار اللوجستي لأن البيانات تحتوي على متغير تابع مستمر (السعر)
-   **المعاملات:** الافتراضية
-   **الاستخدام:** مناسب للبيانات الخطية البسيطة

### 2. الجيران الأقرب (K-Nearest Neighbors)
-   **الملف:** `notebooks/knn_model.ipynb`
-   **المعاملات:** `n_neighbors=5`
-   **الوصف:** يعتمد على المسافة بين النقاط للتنبؤ
-   **الاستخدام:** جيد للبيانات غير الخطية

### 3. شجرة القرار (Decision Tree)
-   **الملف:** `notebooks/decision_tree_model.ipynb`
-   **المعاملات:** `random_state=42`
-   **الوصف:** ينشئ قواعد قرار بناءً على الخصائص
-   **الاستخدام:** سهل التفسير والفهم

### 4. الغابات العشوائية (Random Forest)
-   **الملف:** `notebooks/random_forest_model.ipynb`
-   **المعاملات:** `n_estimators=100, random_state=42`
-   **الوصف:** يجمع عدة أشجار قرار لتحسين الأداء
-   **الاستخدام:** أداء عالي وأقل عرضة للإفراط في التدريب

### 5. XGBoost
-   **الملف:** `notebooks/xgboost_model.ipynb`
-   **المعاملات:** `objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42`
-   **الوصف:** خوارزمية تعزيز متقدمة
-   **الاستخدام:** أداء عالي جداً في معظم المسابقات

## مقاييس التقييم

يتم تقييم كل نموذج باستخدام المقاييس التالية:
-   **MSE (Mean Squared Error):** متوسط مربع الخطأ
-   **RMSE (Root Mean Squared Error):** جذر متوسط مربع الخطأ
-   **R-squared:** معامل التحديد (نسبة التباين المفسر)

## كيفية التنبؤ بقيم جديدة

1. تأكد من أن البيانات الجديدة تحتوي على نفس الخصائص الموجودة في البيانات الأصلية
2. قم بمعالجة البيانات الجديدة بنفس الطريقة المستخدمة في التدريب
3. استخدم النموذج المحفوظ للتنبؤ

### مثال على التنبؤ:
```python
import pandas as pd
import joblib

# تحميل النموذج المحفوظ
model = joblib.load('src/random_forest_model.pkl')

# إعداد البيانات الجديدة (يجب أن تتطابق مع X_processed.csv)
new_data = pd.DataFrame([[...]], columns=X.columns)

# التنبؤ
prediction = model.predict(new_data)
print(f"السعر المتوقع: {prediction[0]:.2f}")
```

## التحسينات المستقبلية

1. **تحسين المعاملات:** استخدام Grid Search أو Random Search لإيجاد أفضل معاملات
2. **إضافة نماذج جديدة:** مثل Support Vector Regression أو Neural Networks
3. **تحليل الخصائص:** دراسة أهمية الخصائص المختلفة
4. **التحقق المتقاطع:** استخدام K-Fold Cross Validation لتقييم أفضل
5. **واجهة مستخدم:** إنشاء واجهة ويب للتنبؤ السهل

## المتطلبات التقنية

### المكتبات المطلوبة:
```
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
numpy>=1.21.0
joblib>=1.1.0
jupyter>=1.0.0
```

### إصدار Python:
Python 3.7 أو أحدث

## الاستنتاجات

هذا المشروع يوفر إطار عمل شامل لتدريب وتقييم نماذج تعلم الآلة المختلفة للتنبؤ بأسعار المنازل. الهيكل المنظم يسهل الصيانة والتطوير المستقبلي، بينما دفاتر Jupyter المنفصلة تتيح مقارنة سهلة بين النماذج المختلفة.

