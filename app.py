import io
import os
import time
import base64
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from pathlib import Path
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename # 新增：用來過濾危險的檔案名稱
from dotenv import load_dotenv



load_dotenv() 

app = Flask(__name__)

# =========================
# CNN 手勢辨識模型設定
# =========================

BASE_DIR = Path(__file__).resolve().parent
GESTURE_MODEL_PATH = BASE_DIR / "gesture_cnn_model.h5"

GESTURE_LABELS = ["0", "1", "2", "3", "4", "5"]

gesture_model = None

try:
    if GESTURE_MODEL_PATH.exists():
        gesture_model = load_model(GESTURE_MODEL_PATH)
        print(f"✅ 手勢 CNN 模型載入成功：{GESTURE_MODEL_PATH}")
    else:
        print(f"⚠️ 找不到手勢 CNN 模型：{GESTURE_MODEL_PATH}")
except Exception as e:
    print("❌ 手勢 CNN 模型載入失敗：", e)

# --- 建立給 HTML 用的時間轉換過濾器 ---
@app.template_filter('tw_time')
def tw_time_filter(dt):
    from datetime import timedelta
    return (dt + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')

# --- 設定上傳檔案的資料夾與允許的格式 ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'} # 允許圖片與影片格式
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 如果 uploads 資料夾不存在，程式啟動時自動建立
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 檢查副檔名的輔助函式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 資料庫設定 ---
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key")
db_url = os.environ.get("DATABASE_URL", "sqlite:///eden.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 1. 既有的最新消息模型
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20), nullable=False)
    tag = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    link = db.Column(db.String(500), nullable=False, unique=True)

# 2. 會員資料模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    avatar_url = db.Column(db.String(200), nullable=True)
    bio = db.Column(db.Text, nullable=True)  # 新增：自我介紹
    posts = db.relationship('Post', backref='author', lazy=True)
    replies = db.relationship('Reply', backref='author', lazy=True)
# 3. 論壇貼文模型
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    media_url = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    replies = db.relationship('Reply', backref='post', lazy=True, cascade='all, delete-orphan')

# 4. 回覆模型
class Reply(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)



# =========================
# CNN 手勢辨識 API：圖片前處理
# =========================
def preprocess_gesture_image(image):
    """
    CNN 攝影機圖片前處理。

    必須和 train_gesture_cnn.py 的 load_and_preprocess() 保持一致：
    1. 灰階
    2. resize 成 64x64
    3. 保持 0~255
    4. 不要 /255.0，因為模型內部已經有 Rescaling(1.0 / 255)
    """

    image = ImageOps.grayscale(image)
    image = image.resize((64, 64))

    img_array = np.array(image).astype("float32")

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/predict_gesture", methods=["POST"])
def predict_gesture():
    """
    接收 achievements.html 傳來的 base64 攝影機截圖，
    使用 gesture_cnn_model.h5 預測手勢數字 0~5。
    """
    if gesture_model is None:
        return jsonify({
            "success": False,
            "error": "CNN 模型尚未載入，請確認 gesture_cnn_model.h5 是否放在 app.py 同一層"
        }), 500

    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "error": "沒有收到 image 資料"
            }), 400

        image_data = data["image"]

        # 前端傳來通常會是 data:image/png;base64,xxxxx
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processed_image = preprocess_gesture_image(image)
        predictions = gesture_model.predict(processed_image, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_label = GESTURE_LABELS[predicted_index]
        confidence = float(predictions[predicted_index])

        probabilities = {
            label: float(predictions[index])
            for index, label in enumerate(GESTURE_LABELS)
        }

        return jsonify({
            "success": True,
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        print("❌ 手勢預測失敗：", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# 初始化資料表
with app.app_context():
    db.create_all()

# --- 網頁路由處理 ---

last_scrape_time = 0
scrape_status = "idle"   # idle | running | done


def run_scraper_background():
    global last_scrape_time, scrape_status
    scrape_status = "running"
    try:
        from fju_scraper import run_scraper
        run_scraper(db, News, app.app_context())
        last_scrape_time = time.time()
        print("🕒 爬蟲更新完成")
    except Exception as e:
        print("❌ 爬蟲失敗：", e)
    finally:
        scrape_status = "done"


@app.route("/scrape_status")
def scrape_status_api():
    return jsonify({"status": scrape_status})


@app.route("/")
def home():
    global last_scrape_time, scrape_status
    current_time = time.time()

    if current_time - last_scrape_time > 3600 and scrape_status != "running":
        scrape_status = "running"
        t = threading.Thread(target=run_scraper_background, daemon=True)
        t.start()
        print("🚀 背景爬蟲啟動")
    else:
        print("⚡ 讀取資料庫快取，略過爬蟲")

    # 所有新聞依日期新到舊排序（date 格式 yyyy-mm-dd 可直接字串排序）
    all_news = News.query.order_by(News.date.desc(), News.id.desc()).all()

    # 所有不重複的 tag，供前端篩選
    tags = sorted(set(n.tag for n in all_news))

    # 論壇最新 3 篇（未登入也能看預覽）
    latest_posts = Post.query.order_by(Post.created_at.desc()).limit(3).all()

    return render_template("index.html",
                           news_list=all_news,
                           tags=tags,
                           latest_posts=latest_posts,
                           scraping=(scrape_status == "running"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template("register.html", error="帳號已存在，請換一個名字！")

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.flush()   # 先取得 user.id 再處理頭像

        file = request.files.get("avatar")
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"avatar_{new_user.id}_{int(time.time())}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            new_user.avatar_url = f"uploads/{filename}"

        db.session.commit()
        return redirect(url_for("login"))

    return render_template("register.html", error=None)

@app.route("/login", methods=["GET", "POST"])
def login():
    error_msg = None  # 準備一個變數裝錯誤訊息
    
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("forum"))
        else:
            # 發生錯誤時，不要 return 字串，而是設定錯誤訊息
            error_msg = "帳號或密碼錯誤，請檢查後再試一次！"
            
    # 將錯誤訊息傳給登入頁面
    return render_template("login.html", error=error_msg)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/forum", methods=["GET", "POST"])
def forum():
    if "user_id" not in session:
        return redirect(url_for("login"))
        
    if request.method == "POST":
        content = request.form.get("content")
        file = request.files.get("media_file") # 接收上傳的檔案
        media_url = None
        
        # 處理檔案上傳
        if file and allowed_file(file.filename):
            # 把檔名過濾掉危險字元，並加上時間戳記防止檔名重複覆蓋
            original_filename = secure_filename(file.filename)
            filename = f"{int(time.time())}_{original_filename}"
            
            # 儲存檔案到 static/uploads/ 裡面
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 存入資料庫的路徑字串 (給 url_for 用的相對路徑)
            media_url = f"uploads/{filename}"

        if content and content.strip():
            # 新增貼文時，把 media_url 也存進去
            new_post = Post(content=content.strip(), user_id=session["user_id"], media_url=media_url)
            db.session.add(new_post)
            db.session.commit()
            return redirect(url_for("forum"))
            
    all_posts = Post.query.order_by(Post.created_at.desc()).all()
    current_user_obj = User.query.get(session["user_id"])
    return render_template("forum.html", current_user=session["username"],
                           current_user_obj=current_user_obj, posts=all_posts)

@app.route("/delete_post/<int:post_id>", methods=["POST"])
def delete_post(post_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    post = Post.query.get_or_404(post_id)
    if post.user_id != session["user_id"]:
        return "權限不足", 403
    db.session.delete(post)
    db.session.commit()
    return redirect(url_for("forum"))


@app.route("/delete_reply/<int:reply_id>", methods=["POST"])
def delete_reply(reply_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    reply = Reply.query.get_or_404(reply_id)
    if reply.user_id != session["user_id"]:
        return "權限不足", 403
    post_id = reply.post_id
    db.session.delete(reply)
    db.session.commit()
    return redirect(url_for("forum") + f"#post-{post_id}")


@app.route("/edit_post/<int:post_id>", methods=["POST"])
def edit_post(post_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    post = Post.query.get_or_404(post_id)
    if post.user_id != session["user_id"]:
        return "權限不足", 403
    new_content = request.form.get("content", "").strip()
    if new_content:
        post.content = new_content
        db.session.commit()
    return redirect(url_for("forum") + f"#post-{post_id}")


@app.route("/reply/<int:post_id>", methods=["POST"])
def reply(post_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    content = request.form.get("content", "").strip()
    if content:
        new_reply = Reply(content=content, user_id=session["user_id"], post_id=post_id)
        db.session.add(new_reply)
        db.session.commit()
    return redirect(url_for("forum") + f"#post-{post_id}")


@app.route("/profile/<username>", methods=["GET", "POST"])
def profile(username):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.filter_by(username=username).first_or_404()

    # 只有本人才能編輯自己的自我介紹
    if request.method == "POST" and session["user_id"] == user.id:
        new_bio = request.form.get("bio", "").strip()
        user.bio = new_bio
        db.session.commit()
        return redirect(url_for("profile", username=username))

    posts = Post.query.filter_by(user_id=user.id).order_by(Post.created_at.desc()).all()
    return render_template("profile.html", profile_user=user, posts=posts,
                           current_user=session.get("username"))


@app.route("/upload_avatar", methods=["POST"])
def upload_avatar():
    if "user_id" not in session:
        return redirect(url_for("login"))
    file = request.files.get("avatar")
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"avatar_{session['user_id']}_{int(time.time())}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        user = User.query.get(session["user_id"])
        user.avatar_url = f"uploads/{filename}"
        db.session.commit()
    return redirect(url_for("profile", username=session["username"]))


@app.route("/about")
def about(): return render_template("about.html")

@app.route("/achievements")
def achievements(): return render_template("achievements.html")

@app.route("/contact")
def contact(): return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)

