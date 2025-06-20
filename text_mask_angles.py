import cv2
import numpy as np
import pytesseract
import os
from matplotlib import pyplot as plt
import time  # Импортируем модуль time

# Путь к изображению
image_folder = r"path-to-your-images"
names = ['image-names']

PATCH_SIZE = 512  # Устанавливаем размер патча

# Функция для загрузки изображения


def load_image(path):
    return cv2.imread(path)

# Уменьшение зернистости и улучшение текста


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Модифицируем параметры порогирования
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 5)  # Увеличен blockSize и уменьшено C
    return binary

# Глобальные OCR-боксы


def get_global_ocr_boxes(image):
    data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data['text'])):
        try:
            conf = float(data['conf'][i])
        except ValueError:
            continue
        if conf > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append((x, y, x + w, y + h))
    return boxes

# Классификация контура


def classify_contour(cnt, global_boxes, offset_x, offset_y):
    x, y, w, h = cv2.boundingRect(cnt)
    abs_x1, abs_y1 = x + offset_x, y + offset_y
    abs_x2, abs_y2 = abs_x1 + w, abs_y1 + h

    is_text = any(abs_x1 < ox2 and abs_y1 < oy2 and abs_x2 > ox1 and abs_y2 > oy1
                  for (ox1, oy1, ox2, oy2) in global_boxes)

    area = cv2.contourArea(cnt)
    aspect = w / h if h > 0 else 0

    if is_text:
        return (0, 255, 0)  # текст — зелёный
    elif area > 500 and 0.5 < aspect < 2.0:  # График — красный
        return (0, 0, 255)  # график — красный
    else:
        return (255, 255, 0)  # дефект — жёлтый

# Разделение на патчи с обработкой


def split_and_classify(image, binary, global_boxes):
    h, w = binary.shape
    pad_h = (h + PATCH_SIZE - 1) // PATCH_SIZE * PATCH_SIZE
    pad_w = (w + PATCH_SIZE - 1) // PATCH_SIZE * PATCH_SIZE

    padded_img = np.zeros((pad_h, pad_w), dtype=binary.dtype)
    padded_img[:h, :w] = binary

    color_mask = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)

    for y in range(0, pad_h, PATCH_SIZE):
        for x in range(0, pad_w, PATCH_SIZE):
            patch = padded_img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            contours, _ = cv2.findContours(
                patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                color = classify_contour(cnt, global_boxes, x, y)
                cv2.drawContours(
                    color_mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE], [cnt], -1, color, 1)

    return color_mask[:h, :w]

# Основной процесс


def get_contours(image_path):
    start_time = time.time()  # Начало отсчета времени

    # Шаг 1: Загрузка и предобработка изображения
    original_img = load_image(image_path)
    preprocessed = preprocess_image(original_img)
    ocr_boxes = get_global_ocr_boxes(original_img)

    # Шаг 2: Разбиение на патчи и классификация контуров
    result_mask = split_and_classify(original_img, preprocessed, ocr_boxes)

    # Создание маски OCR
    ocr_mask = np.zeros_like(original_img)
    for (x1, y1, x2, y2) in ocr_boxes:
        cv2.rectangle(ocr_mask, (x1, y1), (x2, y2),
                      (255, 255, 255), 1)  # Отображаем боксы OCR

    end_time = time.time()
    # Визуализация
    plt.figure(figsize=(14, 10))

    # Покажем 3 изображения:
    plt.subplot(1, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Маска OCR (боксы)")
    plt.imshow(ocr_mask)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Маска с классификацией (уменьшен шум)")
    plt.imshow(result_mask)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    elapsed_time = end_time - start_time  # Вычисляем время выполнения
    # Выводим время в консоль
    print(
        f"Время обработки изображения '{image_path}': {elapsed_time:.2f} секунд")


# Обрабатываем каждое изображение из списка
for image_name in names:
    image_path = os.path.join(image_folder, image_name)
    get_contours(image_path)
