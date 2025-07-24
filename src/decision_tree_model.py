#!/usr/bin/env python
# coding: utf-8

# # تدريب نموذج شجرة القرار (Decision Tree)

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# تحميل البيانات المعالجة
X = pd.read_csv("/home/ubuntu/data/processed/X_processed.csv")
y = pd.read_csv("/home/ubuntu/data/processed/y.csv")

# تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تهيئة وتدريب نموذج شجرة القرار
# يمكن تعديل المعاملات مثل max_depth لتحسين الأداء
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# تقييم أداء النموذج
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"متوسط الخطأ التربيعي (MSE): {mse:.2f}")
print(f"جذر متوسط الخطأ التربيعي (RMSE): {rmse:.2f}")
print(f"معامل التحديد (R-squared): {r2:.2f}")

# حفظ النموذج المدرب
joblib.dump(model, "/home/ubuntu/src/decision_tree_model.pkl")
print("تم حفظ نموذج شجرة القرار في: /home/ubuntu/src/decision_tree_model.pkl")


# In[ ]:


# التنبؤ بقيم جديدة
# قم بتعديل القيم أدناه للتنبؤ بسعر منزل جديد
# يجب أن تتطابق ترتيب الأعمدة مع X_processed.csv
# يمكنك طباعة X.columns لمعرفة الترتيب الصحيح

# مثال على إدخال قيم جديدة (يجب استبدالها بالقيم الفعلية للمستخدم)
# new_data = pd.DataFrame([[...]], columns=X.columns)

# لتبسيط الأمر، سنقوم بإنشاء دالة للتنبؤ
def predict_new_house_price(model, new_input_data):
    # يجب أن تكون new_input_data عبارة عن DataFrame بنفس الأعمدة والترتيب مثل X_processed
    prediction = model.predict(new_input_data)
    return prediction[0]

# مثال على كيفية استخدام الدالة (يجب استبدال القيم بقيم حقيقية من المستخدم)
# يمكنك الحصول على X.columns من الخلية السابقة لمعرفة أسماء الأعمدة
# new_house_features = pd.DataFrame(np.array([[...]]), columns=X.columns)
# predicted_price = predict_new_house_price(model, new_house_features)
# print(f"السعر المتوقع للمنزل الجديد: {predicted_price:.2f}")

