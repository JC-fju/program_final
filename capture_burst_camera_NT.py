import cv2
import time
from pathlib import Path
from datetime import datetime
import os


DATASET_DIR = Path("dataset_seq")
LABELS = ["0", "1", "2", "3", "4", "5", "Thanks"]

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_SIZE = 360

SEQUENCE_LENGTH = 20
FPS_DELAY = 0.08

ui_state = {
    "label": "0",
    "expanded": False,
    "recording": False
}

# 稍微加寬按鈕，避免 Empty_Slot 字被切到
BTN_X1, BTN_X2 = 15, 145 
BTN_Y_START = 100
BTN_H = 40

def mouse_callback(event, x, y, flags, param):
    """處理滑鼠點擊事件的邏輯"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. 檢查是否點擊了「主按鈕」
        if BTN_X1 <= x <= BTN_X2 and BTN_Y_START <= y <= BTN_Y_START + BTN_H:
            ui_state["expanded"] = not ui_state["expanded"]
            return
        
        # 2. 如果選單是展開的，檢查是否點擊了某個「選項」
        if ui_state["expanded"]:
            for i, label in enumerate(LABELS):
                opt_y = BTN_Y_START + (i + 1) * BTN_H
                if BTN_X1 <= x <= BTN_X2 and opt_y <= y <= opt_y + BTN_H:
                    ui_state["label"] = label
                    ui_state["expanded"] = False
                    print(f"✅ 已切換類別：{label}")
                    return
            
            # 3. 如果點在選單外，就自動收合
            ui_state["expanded"] = False

def ensure_folders():
    for label in LABELS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)

def get_roi_rect(frame_width, frame_height):
    x = (frame_width - ROI_SIZE) // 2
    y = (frame_height - ROI_SIZE) // 2
    return x, y, ROI_SIZE, ROI_SIZE

def crop_roi(frame):
    h, w = frame.shape[:2]
    x, y, roi_w, roi_h = get_roi_rect(w, h)
    roi = frame[y:y + roi_h, x:x + roi_w]
    return roi

def save_sequence(frames, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    seq_dir = DATASET_DIR / label / f"seq_{timestamp}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        filename = seq_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(filename), frame)
    return seq_dir

def draw_ui(frame, frame_count, last_saved_path):
    h, w = frame.shape[:2]
    x, y, roi_w, roi_h = get_roi_rect(w, h)

    # 半透明遮罩
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, y), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, y + roi_h), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, y), (x, y + roi_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x + roi_w, y), (w, y + roi_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # ROI 框
    roi_color = (0, 80, 255) if ui_state["recording"] else (0, 255, 80)
    cv2.rectangle(frame, (x, y), (x + roi_w, y + roi_h), roi_color, 3)

    # 頂部狀態列
    top_bar_color = (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 86), top_bar_color, -1)

    if ui_state["recording"]:
        status = f"RECORDING... {frame_count}/{SEQUENCE_LENGTH}"
        status_color = (0, 80, 255)
    else:
        status = "READY (Press Space)"
        status_color = (0, 255, 80)

    cv2.putText(frame, f"Label: {ui_state['label']} | {status}", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Mouse: Change Label | SPACE: Record | Q: Quit", (18, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

    # ==========================================
    # 🌟 繪製下拉式選單 (換成高對比顏色)
    # ==========================================
    # 畫主按鈕：亮藍色 (OpenCV 是 BGR 格式，所以 (255, 200, 100) 偏淺藍)
    cv2.rectangle(frame, (BTN_X1, BTN_Y_START), (BTN_X2, BTN_Y_START + BTN_H), (255, 200, 100), -1)
    # 畫白色粗邊框
    cv2.rectangle(frame, (BTN_X1, BTN_Y_START), (BTN_X2, BTN_Y_START + BTN_H), (255, 255, 255), 2)
    
    arrow = "^" if ui_state["expanded"] else "v"
    # 文字改成黑色，比較容易看
    cv2.putText(frame, f"{ui_state['label']}  {arrow}", (BTN_X1 + 15, BTN_Y_START + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 如果展開，畫出所有選項
    if ui_state["expanded"]:
        for i, label in enumerate(LABELS):
            opt_y = BTN_Y_START + (i + 1) * BTN_H
            # 選項底色：淺灰色
            cv2.rectangle(frame, (BTN_X1, opt_y), (BTN_X2, opt_y + BTN_H), (200, 200, 200), -1)
            cv2.rectangle(frame, (BTN_X1, opt_y), (BTN_X2, opt_y + BTN_H), (255, 255, 255), 1)
            
            # 被選中的項目會是紅色文字，其他的則是黑色文字
            text_color = (0, 0, 255) if label == ui_state["label"] else (0, 0, 0)
            font_thickness = 2 if label == ui_state["label"] else 1
            cv2.putText(frame, label, (BTN_X1 + 10, opt_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, font_thickness)

    return frame

def main():
    ensure_folders()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("❌ 無法開啟攝影機。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 必須先建立視窗，才能綁定滑鼠事件
    cv2.namedWindow("Sequence Capture")
    cv2.setMouseCallback("Sequence Capture", mouse_callback)

    sequence_frames = [] 
    last_saved_path = None
    last_frame_time = 0

    print("✅ 序列攝影機 (高對比UI版) 已啟動")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        now = time.time()

        if ui_state["recording"] and (now - last_frame_time) >= FPS_DELAY:
            roi = crop_roi(frame)
            sequence_frames.append(roi)
            last_frame_time = now

            if len(sequence_frames) == SEQUENCE_LENGTH:
                last_saved_path = save_sequence(sequence_frames, ui_state["label"])
                print(f"🎬 已儲存動作序列：{last_saved_path}")
                ui_state["recording"] = False
                sequence_frames = []

        display = frame.copy()
        display = draw_ui(display, len(sequence_frames), last_saved_path)
        
        cv2.imshow("Sequence Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord(" ") and not ui_state["recording"]:
            ui_state["recording"] = True
            sequence_frames = []
            last_frame_time = time.time()
            ui_state["expanded"] = False # 錄影時自動收起選單
            print("🔴 開始錄製動作...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()