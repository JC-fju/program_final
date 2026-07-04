import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# CNN 手勢辨識訓練程式 ROI 版
# 適用資料夾：dataset_sorted/0 ~ dataset_sorted/5
# 輸出模型：gesture_cnn_model.h5
# =====================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_DIR = Path("dataset_sorted")
MODEL_PATH = Path("gesture_cnn_model.h5")
LABEL_PATH = Path("gesture_labels.json")

LABELS = ["0", "1", "2", "3", "4", "5"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 60
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================
# 檢查與收集資料
# =====================================================
def collect_image_paths():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"找不到資料集資料夾：{DATASET_DIR.resolve()}\n"
            "請確認資料夾名稱是 dataset_sorted，底下有 0,1,2,3,4,5 六個資料夾。"
        )

    print("📁 資料集位置：", DATASET_DIR.resolve())

    per_class_files = {}
    total = 0

    for label in LABELS:
        class_dir = DATASET_DIR / label

        if not class_dir.exists():
            raise FileNotFoundError(f"找不到類別資料夾：{class_dir}")

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        files = sorted(files)
        per_class_files[label] = files
        total += len(files)

        print(f"類別 {label}：{len(files)} 張")

        if len(files) < 30:
            print(f"⚠️ 類別 {label} 圖片偏少，建議至少 30 張以上。")

    if total == 0:
        raise RuntimeError("dataset_sorted 裡沒有任何圖片，請先確認圖片是否放對資料夾。")

    print(f"✅ 圖片總數：{total} 張")
    print()

    return per_class_files


# =====================================================
# 分層切分，每類都切出 validation
# =====================================================
def make_stratified_split(per_class_files):
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []

    for label_index, label in enumerate(LABELS):
        files = per_class_files[label][:]
        random.shuffle(files)

        val_count = max(1, int(len(files) * VAL_RATIO))
        val_files = files[:val_count]
        train_files = files[val_count:]

        for p in train_files:
            train_paths.append(str(p))
            train_labels.append(label_index)

        for p in val_files:
            val_paths.append(str(p))
            val_labels.append(label_index)

        print(f"類別 {label}：訓練 {len(train_files)} 張，驗證 {len(val_files)} 張")

    print()
    print(f"訓練集：{len(train_paths)} 張")
    print(f"驗證集：{len(val_paths)} 張")
    print()

    return train_paths, train_labels, val_paths, val_labels


# =====================================================
# 中心裁切成正方形，再 resize 64x64
# 注意：保持 0~255，模型內部會 Rescaling(1/255)
# =====================================================
def center_crop_square(image):
    image.set_shape([None, None, 1])

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    side = tf.minimum(height, width)

    offset_y = (height - side) // 2
    offset_x = (width - side) // 2

    image = tf.image.crop_to_bounding_box(
        image,
        offset_y,
        offset_x,
        side,
        side
    )

    return image


def load_and_preprocess(path, label, training=False):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image = center_crop_square(image)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    if training:
        image = tf.image.random_brightness(image, max_delta=40)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        # 注意：手勢辨識不建議水平翻轉，1 和鏡像 1 意義不同
    
    image = tf.clip_by_value(image, 0, 255)
    label = tf.one_hot(label, depth=len(LABELS))
    return image, label


# =====================================================
# Dataset
# =====================================================
def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, l: load_and_preprocess(p, l, training=training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =====================================================
# CNN 模型
# =====================================================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),

        # app.py 預測時不要再 /255，因為模型已在這裡正規化
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(96, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        layers.Dense(len(LABELS), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =====================================================
# 測試 dataset_sorted，每類最多 50 張
# =====================================================
def evaluate_dataset_samples(model, per_class_files, max_per_class=50):
    print()
    print("🔎 測試模型對 dataset_sorted 的辨識結果")

    total = 0
    correct = 0

    for true_index, true_label in enumerate(LABELS):
        files = per_class_files[true_label][:max_per_class]

        class_total = 0
        class_correct = 0

        for p in files:
            img_bytes = tf.io.read_file(str(p))
            img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
            img = center_crop_square(img)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.cast(img, tf.float32)
            img = tf.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)[0]
            pred_index = int(np.argmax(pred))

            class_total += 1
            total += 1

            if pred_index == true_index:
                class_correct += 1
                correct += 1

        acc = class_correct / class_total if class_total else 0
        print(f"類別 {true_label}：{class_correct}/{class_total}，準確率 {acc:.2%}")

    overall = correct / total if total else 0
    print()
    print(f"整體測試：{correct}/{total}")
    print(f"整體準確率：{overall:.2%}")


# =====================================================
# 主程式
# =====================================================
def main():
    per_class_files = collect_image_paths()
    train_paths, train_labels, val_paths, val_labels = make_stratified_split(per_class_files)

    train_ds = build_dataset(train_paths, train_labels, training=True)
    val_ds = build_dataset(val_paths, val_labels, training=False)

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    print()
    print("🚀 開始訓練 CNN 模型...")

    class_weight = {
        0: 1.0,
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 1.0,
        5: 1.0,
    }
    print("⚖️  Class weights:", class_weight)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    print()
    print("📊 最終評估：")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss：{loss:.4f}")
    print(f"Validation Accuracy：{acc:.4f}")

    model.save(MODEL_PATH)

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print()
    print(f"✅ 模型已儲存：{MODEL_PATH.resolve()}")
    print(f"✅ 類別順序已儲存：{LABEL_PATH.resolve()}")

    evaluate_dataset_samples(model, per_class_files, max_per_class=50)

    print()
    print("下一步：")
    print("1. 重新啟動 Flask：python app.py")
    print("2. 開啟 http://127.0.0.1:5000/achievements")
    print("3. 測試時請將手放入綠色 ROI 框框內。")


if __name__ == "__main__":
    main()
