import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

show_filters=False
show_lab1=False
show_lab2=False
show_lab3=True
thresh_num = 127

# Загрузка изображения
image = cv2.imread('images/objects1.jpg')  # Замените 'images/cat.jpg' на путь к вашему изображению
if image is None:
    print("Ошибка: изображение не найдено.")
    exit()

# Уменьшение размера изображения для более компактного отображения
scale_percent = 50  # Уменьшаем изображение до 50% от оригинального размера
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Изменение размера изображения
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Преобразование изображения в градации серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки для создания бинарного изображения
_, thresh = cv2.threshold(gray_image, thresh_num, 255, cv2.THRESH_BINARY)

# Применение медианного фильтра
median_filtered = cv2.medianBlur(image, 5)  # Размер окна 5x5

# Применение свертки с ядром [1, -1] для выделения горизонтальных краев
kernel = np.array([[1, -1]])  # Ядро для выделения горизонтальных краев
conv_filtered = cv2.filter2D(thresh, -1, kernel)  # Применение свертки к бинарному изображению

# Применение гауссова размытия
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)  # Ядро 5x5, стандартное отклонение 0

# Черно-белое изображение (градации серого)
black_white_image = thresh

# Медианный фильтр + преобразование в градации серого + свертка
# Сначала применяем медианный фильтр к изображению в градациях серого
median_gray = cv2.medianBlur(image, 5)
gray_median = cv2.cvtColor(median_gray, cv2.COLOR_BGR2GRAY)
_, conv_thresh = cv2.threshold(median_gray, thresh_num, 255, cv2.THRESH_BINARY)
# Затем применяем свертку с ядром [1, -1]
median_gray_conv = cv2.filter2D(conv_thresh, -1, kernel)
median_gray_conv = cv2.cvtColor(median_gray_conv, cv2.COLOR_BGR2GRAY)
median_gray_conv = cv2.cvtColor(median_gray_conv, cv2.COLOR_GRAY2BGR)

# Создание сетки для отображения результатов
# Мы создадим одно большое изображение, в котором будут размещены все результаты
# Размер сетки: 3 строки, 2 столбца (всего 6 изображений)

# Размеры каждого изображения в сетке
height, width = image.shape[:2]

if(show_filters==True):
    # Создаем пустое изображение для сетки
    grid_image = np.zeros((3 * height, 2 * width, 3), dtype=np.uint8)

    # Размещаем оригинальное изображение в левом верхнем углу
    grid_image[0:height, 0:width] = image
    cv2.putText(grid_image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 155, 0), 2)

    # Размещаем медианный фильтр в правом верхнем углу
    grid_image[0:height, width:2*width] = median_filtered
    cv2.putText(grid_image, 'Median Filter', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Размещаем результат свертки в левом среднем углу
    # Преобразуем бинарное изображение в 3 канала для отображения в сетке
    conv_filtered_color = cv2.cvtColor(conv_filtered, cv2.COLOR_GRAY2BGR)
    grid_image[height:2*height, 0:width] = conv_filtered_color
    cv2.putText(grid_image, 'Convolution [1, -1]', (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Размещаем гауссово размытие в правом среднем углу
    grid_image[height:2*height, width:2*width] = gaussian_filtered
    cv2.putText(grid_image, 'Gaussian Blur', (width + 10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Размещаем черно-белое изображение в левом нижнем углу
    black_white_image_color = cv2.cvtColor(black_white_image, cv2.COLOR_GRAY2BGR)
    grid_image[2*height:3*height, 0:width] = black_white_image_color
    cv2.putText(grid_image, 'Black & White', (10, 2*height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Размещаем медианный фильтр + градации серого + свертку в правом нижнем углу
    grid_image[2*height:3*height, width:2*width] = median_gray_conv
    cv2.putText(grid_image, 'Median + Gray + Conv', (width + 10, 2*height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Отображение сетки с результатами
    cv2.imshow('Filtering Results', grid_image)



if(show_lab1==True):
    print("Lab1")
    # 1. Размытие по Гауссу
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Повышение резкости (метод 1: свертка с ядром)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # 3. Повышение резкости (метод 2: маска нерезкости)
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    sharpened_unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # 4. Выделение границ с помощью оператора Собеля
    edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5, borderType=2)
    edges = cv2.convertScaleAbs(edges)

    # 5. Комбинирование результатов
    combined = cv2.addWeighted(blurred_image, 0.4, edges, 0.6, 0)
    combined = cv2.addWeighted(combined, 0.5, sharpened_unsharp, 0.5, 0)


    # 6. Отображение результатов
    def show_images(original, blurred, edges, sharpened, combined):
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 3, 1)
        plt.title('Оригинальное изображение')
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Размытие по Гауссу')
        plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title('Выделение границ')
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('Повышение резкости 1')
        plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Повышение резкости 2')
        plt.imshow(cv2.cvtColor(sharpened_unsharp, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Комбинация изображений')
        plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    show_images(image, blurred_image, edges, sharpened, combined)


if(show_lab2==True):

    mode = input("Выберите режим (1 - видеофайл, 2 - камера): ")

    if mode == "1":
        # Обработка видеофайла "catball.mp4"1
        video_source='images/catball.mp4'
        lower_color = np.array([30, 100, 100])  # Нижняя граница HSV для желтого цвета
        upper_color = np.array([50, 255, 255])  # Верхняя граница HSV для желтого цвета
    elif mode == "2":
        # Обработка видеопотока с камеры
        video_source = 0  # Индекс камеры (0 для встроенной камеры)
        lower_color = np.array([94, 80, 2])  # Нижняя граница HSV для синего цвета
        upper_color = np.array([126, 255, 255])  # Верхняя граница HSV для синего цвета
    else:
        print("Неверный режим. Завершение программы.")
        exit()


    # Захват видеопотока
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Создание маски для выделения объекта по цвету
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        # Поиск контуров на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если контуры найдены
        if contours:
            # Находим наибольший контур (предполагаем, что это наш объект)
            largest_contour = max(contours, key=cv2.contourArea)

            # Вычисляем центр масс контура
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Рисуем контур и центр масс на кадре
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Выводим координаты центра масс в верхнем левом углу
            cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Если объект не найден, выводим сообщение
            cv2.putText(frame, "Object not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Отображение результата
        cv2.imshow('Frame', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break





if(show_lab3==True):
    print("Lab3")

    # Загрузка предобученных каскадов Хаара
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Инициализация видеопотока с камеры
    cap = cv2.VideoCapture(0)

    # Переменные для расчета FPS
    prev_time = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Детекция лиц
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Переменные для проверки улыбки и глаз
        smile_detected = False
        eyes_detected = False

        for (x, y, w, h) in faces:
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Область интереса (ROI) для глаз и улыбки
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Детекция глаз
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            if len(eyes) >= 2:  # Если найдено хотя бы два глаза
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            else:
                eyes_detected = False

            # Детекция улыбки
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=20, minSize=(20, 20))
            if len(smiles) > 0:  # Если найдена улыбка
                smile_detected = True
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            else:
                smile_detected = False

        # Вывод сообщений
        if not smile_detected:
            cv2.putText(frame, "Smile", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not eyes_detected:
            cv2.putText(frame, "Openn your eyes", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Расчет FPS
        current_time = time.time()
        time_diff = current_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = current_time

        # Отображение FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Отображение кадра
        cv2.imshow('Face Detection', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()



# Ожидание нажатия клавиши для закрытия окна
cv2.waitKey(0)
cv2.destroyAllWindows()