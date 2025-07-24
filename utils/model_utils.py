"""
ملف مساعد يحتوي على دوال مفيدة للعمل مع النماذج
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_model(model_path):
    """
    تحميل نموذج محفوظ
    
    Args:
        model_path (str): مسار ملف النموذج
    
    Returns:
        model: النموذج المحمل
    """
    return joblib.load(model_path)

def load_processed_data():
    """
    تحميل البيانات المعالجة
    
    Returns:
        tuple: (X, y) البيانات المعالجة
    """
    X = pd.read_csv("/home/ubuntu/data/processed/X_processed.csv")
    y = pd.read_csv("/home/ubuntu/data/processed/y.csv")
    return X, y

def evaluate_model(model, X_test, y_test):
    """
    تقييم أداء النموذج
    
    Args:
        model: النموذج المدرب
        X_test: بيانات الاختبار (المتغيرات المستقلة)
        y_test: بيانات الاختبار (المتغير التابع)
    
    Returns:
        dict: قاموس يحتوي على مقاييس الأداء
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def predict_house_price(model, house_features):
    """
    التنبؤ بسعر منزل جديد
    
    Args:
        model: النموذج المدرب
        house_features (pd.DataFrame): خصائص المنزل الجديد
    
    Returns:
        float: السعر المتوقع
    """
    prediction = model.predict(house_features)
    return prediction[0] if len(prediction) == 1 else prediction

def compare_models(model_paths, X_test, y_test):
    """
    مقارنة أداء عدة نماذج
    
    Args:
        model_paths (dict): قاموس يحتوي على أسماء النماذج ومساراتها
        X_test: بيانات الاختبار (المتغيرات المستقلة)
        y_test: بيانات الاختبار (المتغير التابع)
    
    Returns:
        pd.DataFrame: جدول مقارنة أداء النماذج
    """
    results = []
    
    for model_name, model_path in model_paths.items():
        try:
            model = load_model(model_path)
            metrics = evaluate_model(model, X_test, y_test)
            metrics['Model'] = model_name
            results.append(metrics)
        except Exception as e:
            print(f"خطأ في تحميل أو تقييم النموذج {model_name}: {e}")
    
    return pd.DataFrame(results)[['Model', 'MSE', 'RMSE', 'R2']]

def create_sample_input():
    """
    إنشاء مثال على البيانات المدخلة للتنبؤ
    
    Returns:
        pd.DataFrame: مثال على البيانات المدخلة
    """
    # تحميل البيانات المعالجة للحصول على أسماء الأعمدة
    X, _ = load_processed_data()
    
    # إنشاء مثال بقيم افتراضية (يجب تعديلها حسب الحاجة)
    sample_data = np.zeros((1, len(X.columns)))
    sample_df = pd.DataFrame(sample_data, columns=X.columns)
    
    return sample_df

