import cv2
from cvzone.HandTrackingModule import HandDetector
import os
import time

labels = ['Alacrity', 'Chaos_Meteor', 'Cold_Snap', 'Deafening_Blast', 'EMP',
          'Forge_Spirit', 'Ghost_Walk', 'Hold_pos', 'Ice_Wall',
          'Sun_Strike', 'Tornado']

print("=== СБОР ДАТАСЕТА ДЛЯ YOLO POSE ===")
for i, name in enumerate(labels):
    print(f"[{i}] - {name}")

class_idx = int(input("\nВведите ЦИФРУ класса (например, 2 для Cold_Snap): "))
mode = input("Введите 'train' или 'val': ").strip().lower()
if mode not in ['train', 'val']: mode = 'train'

base_path = f"yolo_pose_dataset/{mode}"
img_folder = f"{base_path}/images"
label_folder = f"{base_path}/labels"

os.makedirs(img_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)

class_name = labels[class_idx]
print(f"\n Запись жеста [{class_name}] в папку [{mode}]. Жмите 'S' для сохранения.")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
counter = 0

while True:
    success, image = cap.read()
    if not success: break

    img_h, img_w, _ = image.shape
    pure_image = image.copy()  # Сохраняем чистый кадр для датасета

    # Ищем руку, но отключаем внутреннюю отрисовку cvzone, чтобы она нам не мешала
    hands, img = detector.findHands(image, draw=False)

    yolo_line = ""

    if hands:
        hand = hands[0]

        # 1. РАМКА РУКИ
        x, y, w, h = hand["bbox"]

        # Рисуем рамку вручную синим цветом, чтобы ты видел, что рука детектируется
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        x_center_norm = (x + (w / 2)) / img_w
        y_center_norm = (y + (h / 2)) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        yolo_line = f"{class_idx} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

        # 2. РУЧНАЯ ОТРИСОВКА ВСЕХ 21 ТОЧЕК
        # Извлекаем список точек. Каждая точка — это строго [x, y] или [x, y, z]
        lmList = hand["lmList"]

        for lm in lmList:
            # Безопасное извлечение координат x и y, независимо от версии cvzone
            cx, cy = int(lm[0]), int(lm[1])

            # РИСУЕМ ЗЕЛЕНУЮ ТОЧКУ ВРУЧНУЮ на экране! Теперь она обязана появиться
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # Нормализуем для YOLO текстового файла
            cx_norm = cx / img_w
            cy_norm = cy / img_h
            yolo_line += f" {cx_norm:.6f} {cy_norm:.6f} 2"

    # Выводим инфо на экран
    cv2.putText(image, f"Class: {class_name} | Saved: {counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Data Collector", image)

    key = cv2.waitKey(1)
    if key == ord('z') or key == ord('я'):
        if yolo_line != "":
            counter += 1
            timestamp = int(time.time() * 1000)
            file_name = f"{class_name}_{timestamp}"

            # Сохраняем ЧИСТУЮ картинку и ТЕКСТОВУЮ разметку
            cv2.imwrite(f"{img_folder}/{file_name}.jpg", pure_image)
            with open(f"{label_folder}/{file_name}.txt", "w") as f:
                f.write(yolo_line)

            print(f"Сохранено: {file_name} (Всего: {counter})")
        else:
            print("Рука не в кадре!")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()