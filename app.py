import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Функция для загрузки модели (кэшируется для скорости)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('chest_xray_efficientnet_model.keras')

# Загружаем модель
model = load_model()

# Загружаем оптимальный порог классификации
try:
    with open('optimal_threshold.txt', 'r') as f:
        optimal_threshold = float(f.read())
except FileNotFoundError:
    optimal_threshold = 0.66  # Значение по умолчанию, если файл не найден

# Функция предобработки изображения
def preprocess_image(img):
    img = img.convert('RGB')  # Конвертируем в RGB
    img = img.resize((224, 224))  # Изменяем размер под вход модели
    arr = np.array(img)
    arr = preprocess_input(arr)  # Нормализация для EfficientNet
    arr = np.expand_dims(arr, axis=0)  # Добавляем размер батча
    return arr

# Функция предсказания
def predict(arr):
    prob = model.predict(arr)[0, 0]
    label = "Pneumonia" if prob >= optimal_threshold else "Normal"
    return label, prob

# Заголовок и описание приложения
st.title("Диагностика пневмонии по рентгеновскому снимку")
st.write("Загрузите рентген грудной клетки, чтобы получить диагноз (PNG, JPG, JPEG).")
st.warning("Внимание: Эта программа является экспериментальной и может ошибаться. Результаты не заменяют профессиональную медицинскую диагностику. Обратитесь к врачу для точного диагноза.")

# Поле для загрузки изображения
uploaded_file = st.file_uploader("Выберите снимок", type=["png", "jpg", "jpeg"])

# Обработка загруженного изображения
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженный снимок", use_container_width=True)
    arr = preprocess_image(image)
    label, prob = predict(arr)
    if label == "Pneumonia":
        st.error(f"Результат: {label} (вероятность: {prob:.2f})")
    else:
        st.success(f"Результат: {label} (вероятность: {prob:.2f})")
    st.write(f"Примечание: Вероятность выше {optimal_threshold:.2f} считается 'Pneumonia'. Это значение было оптимизировано на основе валидационного набора.")

