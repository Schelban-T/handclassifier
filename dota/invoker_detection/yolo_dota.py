from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Указываем путь к созданному YAML-файлу конфигурации
    dataset_yaml = "pose_dataset.yaml"

    # 2. Загружаем модель YOLO Pose (заменили -cls на -pose)
    model = YOLO("yolo26n-pose.pt")

    print("Запуск быстрого обучения детектора позы (скелета рук)...")
    results = model.train(
        data=dataset_yaml,  # Передаем YAML-файл, а не папку!
        epochs=60,
        imgsz=640,  # Ставим 640 для качественного поиска мелких суставов пальцев

        # 🔥 ТУРБО-НАСТРОЙКИ ПОД ТВОЮ RTX 5060 Ti:
        batch=64,  # 256 для imgsz=640 идеально займет память твоей 16GB карты
        workers=0,  # Полная защита от зависания потоков в Windows
        val=False,  # Не тратим время на проверки во время учебы

        # Оптимизатор и настройки против переобучения
        optimizer="AdamW",
        lr0=0.001,
        patience=20,

        # Аугментация
        scale=0.3,
        perspective=0.0005,
        hsv_h=0.02,
        hsv_s=0.4,
        hsv_v=0.4,
        fliplr=0.0,  # Строго 0, чтобы левая/правая рука не путались

        project="dota2_pose_project_zaebalo",
        name="yolo26_pose_turbo_zaebalo"
    )

    print("Обучение завершено успешно!")