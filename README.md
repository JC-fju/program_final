陳宥翔 高心宇
# EDEN 社群平台

輔仁大學數學系程設期末專題，結合 Flask 網頁、資料爬蟲、CNN 手勢辨識與論壇互動功能。

## 功能介紹

- **首頁**：自動爬取數學系最新消息（系所公告、課程公告、演講、獎助學金等），支援 tag 篩選
- **討論區**：會員發文、回覆、編輯、刪除，支援圖片/影片附件與個人頁面
- **成果展示**：CNN 手勢辨識（0～5），手勢 1/2 可控制走馬燈切換
- **會員系統**：註冊、登入、頭像上傳

## 技術架構

| 項目 | 技術 |
|------|------|
| 後端 | Python / Flask |
| 資料庫 | PostgreSQL（Render）/ SQLite（本地） |
| 爬蟲 | Selenium（動態網頁） |
| AI 模型 | TensorFlow / Keras CNN |
| 部署 | Render + Docker |

## 本地執行方式

### 方法一：Docker（推薦）

確保已安裝並啟動 Docker Desktop，在專案根目錄執行：

```bash
docker build -t eden_project .
docker run -d -p 5000:5000 --name web_server eden_project
```

開啟瀏覽器前往 `http://localhost:5000`

### 方法二：直接執行

```bash
pip install -r requirements.txt
python app.py
```

## 環境變數設定

在專案根目錄建立 `.env` 檔案：

```
SECRET_KEY=your_secret_key
DATABASE_URL=your_postgresql_url
```

## 影片上傳支援

本專案支援影片上傳，需安裝 FFmpeg：

**Windows：**
```bash
winget install -e --id Gyan.FFmpeg
```

安裝後確認：
```bash
ffmpeg -version
```

## CNN 手勢辨識訓練

訓練資料放置於 `dataset_sorted/0` ～ `dataset_sorted/5`，執行：

```bash
python train_gesture_cnn.py
```

訓練完成後會產生 `gesture_cnn_model.h5`，重啟 Flask 即可使用。

## 專案結構

```
├── app.py                  # Flask 主程式
├── fju_scraper.py          # 數學系公告爬蟲
├── train_gesture_cnn.py    # CNN 手勢辨識訓練
├── capture_burst_camera.py # 訓練資料拍攝工具
├── gesture_cnn_model.h5    # 訓練好的模型
├── gesture_labels.json     # 類別標籤
├── Dockerfile
├── requirements.txt
├── static/
│   ├── css/
│   ├── image/
│   └── uploads/            # 使用者上傳檔案（不含於 git）
└── templates/              # HTML 頁面
```
