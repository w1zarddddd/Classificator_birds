import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Параметры генерации
num_samples = 1000  # Общее количество записей
np.random.seed(42)

# Список видов птиц
species = [
    'Изумрудный колибри',
    'Красный кардинал',
    'Снегирь',
    'Синяя сойка',
    'Золотой фазан',
    'Белый лебедь',
    'Черный дрозд',
    'Пестрый дятел'
]

# Словарь характеристик для каждого вида
species_profiles = {
    'Изумрудный колибри': {
        'beak_type': 'Длинный и тонкий',
        'plumage_color': 'Зеленый',
        'eye_color': 'Черный',
        'habitat': 'Тропические леса',
        'feeding_type': 'Нектароядный',
        'body_size_range': (80, 100)
    },
    'Красный кардинал': {
        'beak_type': 'Короткий и крепкий',
        'plumage_color': 'Красный',
        'eye_color': 'Коричневый',
        'habitat': 'Леса',
        'feeding_type': 'Всеядный',
        'body_size_range': (200, 220)
    },
    'Снегирь': {
        'beak_type': 'Короткий и крепкий',
        'plumage_color': 'Красный',
        'eye_color': 'Черный',
        'habitat': 'Леса',
        'feeding_type': 'Травоядный',
        'body_size_range': (150, 170)
    },
    'Синяя сойка': {
        'beak_type': 'Крюкообразный',
        'plumage_color': 'Синий',
        'eye_color': 'Серый',
        'habitat': 'Леса',
        'feeding_type': 'Всеядный',
        'body_size_range': (250, 270)
    },
    'Золотой фазан': {
        'beak_type': 'Короткий и крепкий',
        'plumage_color': 'Золотой',
        'eye_color': 'Желтый',
        'habitat': 'Горные районы',
        'feeding_type': 'Травоядный',
        'body_size_range': (600, 700)
    },
    'Белый лебедь': {
        'beak_type': 'Плоский',
        'plumage_color': 'Белый',
        'eye_color': 'Черный',
        'habitat': 'Водоёмы',
        'feeding_type': 'Травоядный',
        'body_size_range': (1200, 1500)
    },
    'Черный дрозд': {
        'beak_type': 'Короткий и крепкий',
        'plumage_color': 'Чёрный',
        'eye_color': 'Коричневый',
        'habitat': 'Леса',
        'feeding_type': 'Всеядный',
        'body_size_range': (230, 250)
    },
    'Пестрый дятел': {
        'beak_type': 'Долотообразный',
        'plumage_color': 'Чёрно-белый',
        'eye_color': 'Красный',
        'habitat': 'Леса',
        'feeding_type': 'Насекомоядный',
        'body_size_range': (180, 200)
    }
}

# Генерация данных
data = []
for _ in range(num_samples):
    # Выбор случайного вида
    bird_species = np.random.choice(species)
    profile = species_profiles[bird_species]

    # Генерация признаков с небольшим шумом
    data.append({
        'species': bird_species,
        'beak_type': profile['beak_type'],
        'plumage_color': profile['plumage_color'],
        'eye_color': profile['eye_color'],
        'habitat': profile['habitat'],
        'feeding_type': profile['feeding_type'],
        'body_size': np.random.randint(*profile['body_size_range']) + np.random.normal(0, 2)
    })

# Создание DataFrame
df = pd.DataFrame(data)

# Нормализация и округление числовых признаков
df['body_size'] = df['body_size'].round().astype(int)

# Сохранение в CSV
df.to_csv('birds_data.csv', index=False)

print(f"Датасет сгенерирован. Размер: {df.shape}")
print("Пример данных:")
print(df.head())