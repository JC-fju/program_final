import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# =====================================================
# CNN 手勢辨識訓練程式（MobileNetV2 遷移學習版）
# 適用資料夾：dataset_sorted/0 ~ dataset_sorted/5
# 輸出模型：gesture_cnn_model.h5
# =====================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_DIR = Path("dataset_sorted")
MODEL_PATH  = Path("gesture_cnn_model.h5")
LABEL_PATH  = Path("gesture_labels.json")

LABELS = ["0", "1", "2", "3", "4", "5"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMG_SIZE   = (96, 96)   # MobileNetV2 建議最小 96x96
BATCH_SIZE = 16
EPOCHS     = 60
VAL_RATIO  = 0.2
SEED       = 42

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
    """中心裁切成正方形"""
    image.set_shape([None, None, 3])   # RGB
    shape  = tf.shape(image)
    height, width = shape[0], shape[1]
    side   = tf.minimum(height, width)
    offset_y = (height - side) // 2
    offset_x = (width  - side) // 2
    return tf.image.crop_to_bounding_box(image, offset_y, offset_x, side, side)


def load_and_preprocess(path, label, training=False):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)  # RGB
    image = center_crop_square(image)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    if training:
        image = tf.image.random_brightness(image, max_delta=40)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        # 手勢辨識不做水平翻轉（1 和鏡像 1 語意不同）
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0]+10, IMG_SIZE[1]+10)
        image = tf.image.random_crop(image, size=[IMG_SIZE[0], IMG_SIZE[1], 3])

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
# MobileNetV2 遷移學習模型
# =====================================================
def build_model():
    # 載入預訓練 MobileNetV2，不含分類層
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # 第一階段：凍結，只訓練分類頭

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # MobileNetV2 預期輸入在 [-1, 1]，這裡做正規化
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


# =====================================================
# 測試 dataset_sorted，每類最多 50 張
# =====================================================
def evaluate_dataset_samples(model, per_class_files, max_per_class=50):
    print()
    print("🔎 測試模型對 dataset_sorted 的辨識結果")

    total, correct = 0, 0

    for true_index, true_label in enumerate(LABELS):
        files = per_class_files[true_label][:max_per_class]
        class_total, class_correct = 0, 0

        for p in files:
            img_bytes = tf.io.read_file(str(p))
            img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)  # RGB
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
    print(f"\n整體測試：{correct}/{total}")
    print(f"整體準確率：{overall:.2%}")


# =====================================================
# 主程式
# =====================================================
def main():
    per_class_files = collect_image_paths()
    train_paths, train_labels, val_paths, val_labels = make_stratified_split(per_class_files)

    train_ds = build_dataset(train_paths, train_labels, training=True)
    val_ds   = build_dataset(val_paths,   val_labels,   training=False)

    model, base_model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="gesture_cnn_best.h5", monitor="val_accuracy",
            save_best_only=True, verbose=1),
    ]

    # ── 第一階段：凍結 base，只訓練分類頭 ──
    print("\n🚀 第一階段：訓練分類頭（base_model 凍結）...")
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    # ── 第二階段：解凍後半段 fine-tune ──
    print("\n🔓 第二階段：Fine-tune（解凍 MobileNetV2 後 50 層）...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("\n📊 最終評估：")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss：{loss:.4f}")
    print(f"Validation Accuracy：{acc:.4f}")

    model.save(MODEL_PATH)

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 模型已儲存：{MODEL_PATH.resolve()}")
    print(f"✅ 類別順序已儲存：{LABEL_PATH.resolve()}")

    evaluate_dataset_samples(model, per_class_files, max_per_class=50)

    print("\n下一步：")
    print("1. 重新啟動 Flask：python app.py")
    print("2. 開啟 http://127.0.0.1:5000/achievements")
    print("3. 注意：前處理已改為 RGB，app.py 也要對應更新。")


if __name__ == "__main__":
    main()
