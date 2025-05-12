import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Загрузка данных
df = pd.read_csv('birds_data.csv')

# Отдельно сохраняем оригинальный список признаков
feature_cols = [c for c in df.columns if c != 'species']

# Кодирование категориальных признаков
encoders = {}
for col in feature_cols + ['species']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Сохраняем энкодеры
joblib.dump(encoders, 'encoders.pkl')

# Подготовка X и y
X = df[feature_cols]
y = df['species']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(model, 'model.pkl')

# Оценка точности на тесте
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.2f}")
