import pydirectinput
import cv2
from collections import deque, Counter
import time
import sys
from pathlib import Path
from ultralytics import YOLO  # Импортируем нашу YOLO

# 1. Настройка путей и камеры
cap = cv2.VideoCapture(0)
def resource_path(relative_path):
    """Находит файл и в обычном запуске, и внутри exe."""
    if getattr(sys, "frozen", False):
        # --onefile: файлы лежат во временной папке _MEIPASS
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        base = Path(__file__).resolve().parent
    return base / relative_path

model_path = str(resource_path("best.pt"))
classifier = YOLO(model_path)

# Итоговый список классов (БЕЗ Invoke, строго соответствующий yaml-файлу!)
labels = ['Alacrity', 'Chaos_Meteor', 'Cold_Snap', 'Deafening_Blast', 'EMP',
          'Forge_Spirit', 'Ghost_Walk', 'Hold_pos', 'Ice_Wall',
          'Sun_Strike', 'Tornado']

spell_keys = {
    "Alacrity":        ["w", "w", "e", "r", "d"],
    "Chaos_Meteor":    ["e", "e", "w", "r", "d"],
    "Cold_Snap":       ["q", "q", "q", "r", "d"],
    "Deafening_Blast": ["q", "w", "e", "r", "d"],
    "EMP":             ["w", "w", "w", "r", "d"],
    "Forge_Spirit":    ["e", "e", "q", "r", "d"],
    "Ghost_Walk":      ["q", "q", "w", "r", "d"],
    "Ice_Wall":        ["q", "q", "e", "r", "d"],
    "Sun_Strike":      ["e", "e", "e", "r", "d"],
    "Tornado":         ["w", "w", "q", "r", "d"],
}

# Настройки таймингов и буфера
CONFIRM_TIME = 0.25  # Для YOLO позу распознавать легче, можно снизить задержку каста!
VOTE_WINDOW = 8
MIN_VOTES = 6
KEY_DELAY = 0.01
D_EXTRA_DELAY = 0.05
pydirectinput.PAUSE = 0
pydirectinput.FAILSAFE = False

prediction_buffer = deque(maxlen=VOTE_WINDOW)
candidate_label = None
candidate_since = None
triggered = False

def press_sequence(keys):
    """Быстро нажимает клавиши в Доте."""
    for k in keys:
        if k == "d":
            time.sleep(D_EXTRA_DELAY)
        pydirectinput.keyDown(k)
        time.sleep(KEY_DELAY)
        pydirectinput.keyUp(k)
        time.sleep(KEY_DELAY)

def reset_state():
    global candidate_label, candidate_since, triggered
    prediction_buffer.clear()
    candidate_label = None
    candidate_since = None
    triggered = False

while True:
    success, frame = cap.read()
    if not success:
        break

    # conf=0.6 означает, что мы игнорируем случайные шумы на фоне, если уверенность ниже 60%
    results = classifier(frame, conf=0.5, verbose=False)[0]

    # Проверяем, нашла ли YOLO руку со скелетом в кадре
    if results is not None and getattr(results, 'boxes', None) is not None and len(results.boxes) > 0:
        # 1. Получаем индекс класса для самой первой найденной руки
        index = int(results.boxes.cls[0].item())
        prediction = labels[index]
        prediction_buffer.append(prediction)

        # Достаем координаты рамки, чтобы нарисовать её для красоты на экране
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Отрисовка скелета руки прямо из предсказаний YOLO (зеленые точки суставов)
        if results.keypoints is not None:
            for kpt in results.keypoints.xy[0]:
                cx, cy = int(kpt[0]), int(kpt[1])
                if cx > 0 and cy > 0: # Если точка определена
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


        stable = None
        if len(prediction_buffer) == VOTE_WINDOW:
            label, votes = Counter(prediction_buffer).most_common(1)[0]
            if votes >= MIN_VOTES:
                stable = label

        now = time.time()

        if stable != candidate_label:
            candidate_label = stable
            candidate_since = now if stable else None
            triggered = False

        elif stable is not None and not triggered:
            held = now - candidate_since
            if held >= CONFIRM_TIME:
                keys = spell_keys.get(stable)
                if keys:
                    print(f"🧙‍♂️ Прокаст Инвокера: {stable} -> {keys}")
                    press_sequence(keys)
                triggered = True

        # Вывод текста на экран
        if candidate_label:
            cv2.putText(frame, candidate_label + (" OK" if triggered else ""),
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:

        reset_state()

    cv2.imshow("Dota 2 Invoker Pose Control", frame)
    if cv2.waitKey(1) == 27: # Выход на ESC
        break

cap.release()
cv2.destroyAllWindows()
