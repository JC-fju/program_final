import json
import os
import random
import cv2
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# 時空雙維度 (CNN + LSTM) 手語辨識訓練程式
# 適用資料夾：dataset_seq (包含 20 幀的連續動作序列)
# 輸出模型：gesture_lstm_model.h5
# =====================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

DATASET_DIR = Path("dataset_seq")
MODEL_PATH = Path("gesture_lstm_model.h5")
LABEL_PATH = Path("gesture_labels.json")

# ⚠️ 請確保這裡的 LABELS 和你錄影程式 (capture_burst_camera.py) 裡的一模一樣
LABELS = ["0", "1", "2", "3", "4", "5", "Thanks", "Empty_Slot"]

SEQ_LENGTH = 20
IMG_SIZE = (64, 64)
BATCH_SIZE = 8  # 因為 LSTM 吃矩陣較大，Batch 設小一點避免記憶體爆炸
EPOCHS = 40
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# 1. 檢查與收集「動作序列」資料
# =====================================================
def collect_sequence_paths():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"找不到資料集資料夾：{DATASET_DIR.resolve()}")

    print("📁 資料集位置：", DATASET_DIR.resolve())
    
    per_class_seqs = {}
    total_seqs = 0

    for label in LABELS:
        class_dir = DATASET_DIR / label
        if not class_dir.exists():
            continue
            
        seq_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        seq_paths = []
        
        for seq_dir in seq_dirs:
            # 找到該資料夾下所有 jpg，照檔名 (frame_00 ~ 19) 排序
            frames = sorted(list(seq_dir.glob("*.jpg")))
            if len(frames) == SEQ_LENGTH:
                seq_paths.append([str(f) for f in frames])
            else:
                print(f"⚠️ {seq_dir.name} 張數不對 ({len(frames)}/{SEQ_LENGTH})，已略過。")
        
        per_class_seqs[label] = seq_paths
        total_seqs += len(seq_paths)
        print(f"類別 {label}：{len(seq_paths)} 個動作序列")

    if total_seqs == 0:
        raise RuntimeError("沒有找到任何完整的 20 幀動作序列，請先去錄製資料！")

    print(f"\n✅ 總計動作序列數：{total_seqs} 個 (共 {total_seqs * SEQ_LENGTH} 張圖片)\n")
    return per_class_seqs

# =====================================================
# 2. 分層切分 (Train / Validation)
# =====================================================
def make_stratified_split(per_class_seqs):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for label_index, label in enumerate(LABELS):
        if label not in per_class_seqs or len(per_class_seqs[label]) == 0:
            continue
            
        seqs = per_class_seqs[label][:]
        random.shuffle(seqs)

        val_count = max(1, int(len(seqs) * VAL_RATIO))
        # 確保資料量太少時不會出錯
        if len(seqs) == 1: 
            val_count = 0 
            
        val_seqs = seqs[:val_count]
        train_seqs = seqs[val_count:]

        for s in train_seqs:
            train_paths.append(s)
            train_labels.append(label_index)

        for s in val_seqs:
            val_paths.append(s)
            val_labels.append(label_index)

    print(f"訓練集：{len(train_paths)} 個序列")
    print(f"驗證集：{len(val_paths)} 個序列\n")
    return train_paths, train_labels, val_paths, val_labels

# =====================================================
# 3. 建立 tf.data.Dataset (使用 Generator)
# =====================================================
def make_dataset(seq_paths, labels, training=False):
    def gen():
        # 把資料綁在一起，如果是訓練就打亂順序
        data = list(zip(seq_paths, labels))
        if training:
            random.shuffle(data)
            
        for paths, label_idx in data:
            frames = []
            for p in paths:
                # 讀取單張並轉為灰階、縮放
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                img = np.expand_dims(img, axis=-1) # (64, 64, 1)
                frames.append(img)
            
            # 轉換為 NumPy 矩陣，形狀會是 (20, 64, 64, 1)
            frames = np.array(frames, dtype=np.float32)
            yield frames, label_idx

    # 使用 Generator 創建 Dataset
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# =====================================================
# 4. 建立 CNN + LSTM 模型
# =====================================================
def build_lstm_model():
    model = models.Sequential([
        # 輸入張量形狀：(時間長度, 高, 寬, 通道)
        layers.Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 1)),

        layers.TimeDistributed(layers.Rescaling(1.0 / 255)),

        # 讓這 20 幀圖片分別通過一樣的 CNN 萃取空間特徵
        layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding="same", activation="relu")),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        # 攤平每張圖片的特徵圖
        layers.TimeDistributed(layers.Flatten()),

        # 🌟 進入 LSTM 網路，分析這 20 組特徵的時間軌跡！
        # return_sequences=False 代表 LSTM 看完 20 步後，只輸出一個總結結論
        layers.LSTM(128, return_sequences=False),
        
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # 最終分類 (因為資料給整數標籤，使用 sparse_categorical_crossentropy)
        layers.Dense(len(LABELS), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =====================================================
# 主程式
# =====================================================
def main():
    per_class_seqs = collect_sequence_paths()
    train_paths, train_labels, val_paths, val_labels = make_stratified_split(per_class_seqs)

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False)

    model = build_lstm_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    print("🚀 開始訓練 CNN+LSTM 模型...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("\n📊 最終評估：")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation Loss：{loss:.4f}")
    print(f"Validation Accuracy：{acc:.4f}")

    # 儲存模型與標籤
    model.save(MODEL_PATH)
    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 模型已儲存：{MODEL_PATH.resolve()}")
    print("下一步：我們將去改寫 app.py 裡的 /predict_gesture API，實作「20 幀滑動窗口」！")

if __name__ == "__main__":
    main()