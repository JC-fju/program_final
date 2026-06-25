import os
import time
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv  # 新增這行：載入 dotenv 套件

# 讀取 .env 檔案中的變數
load_dotenv() 

app = Flask(__name__)

# --- 改從環境變數讀取機密資訊 ---

# 1. 讀取 Secret Key (如果找不到，給一個預設值防呆)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key")

# 2. 讀取資料庫網址
db_url = os.environ.get("DATABASE_URL", "sqlite:///eden.db")

# Render PostgreSQL 網址修正防呆機制
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
    # 與貼文建立一對多關聯
    posts = db.relationship('Post', backref='author', lazy=True)

# 3. 論壇貼文模型
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# 初始化資料表
with app.app_context():
    db.create_all()

# --- 網頁路由處理 ---

last_scrape_time = 0 

@app.route("/")
def home():
    global last_scrape_time
    current_time = time.time()
    
    # 設定：距離上次爬取超過 1 小時 (3600秒)，或伺服器剛啟動時，才觸發爬蟲
    if current_time - last_scrape_time > 3600:
        from fju_scraper import run_scraper
        run_scraper(db, News, app.app_context())
        last_scrape_time = current_time
        print("🕒 觸發爬蟲更新資料")
    else:
        print("⚡ 讀取資料庫快取，略過爬蟲")
        
    all_news = News.query.order_by(News.id.desc()).limit(6).all()
    return render_template("index.html", news_list=all_news)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "帳號已存在，請換一個名字！"
            
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
        
    return render_template("register.html")

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
    # 權限驗證：未登入者直接導向登入頁面
    if "user_id" not in session:
        return redirect(url_for("login"))
        
    # 處理發布新貼文 (POST)
    if request.method == "POST":
        content = request.form.get("content")
        if content and content.strip():
            new_post = Post(content=content.strip(), user_id=session["user_id"])
            db.session.add(new_post)
            db.session.commit()
            return redirect(url_for("forum"))
            
    # 讀取所有貼文 (按時間由新到舊排序)
    all_posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template("forum.html", current_user=session["username"], posts=all_posts)

@app.route("/about")
def about(): return render_template("about.html")

@app.route("/achievements")
def achievements(): return render_template("achievements.html")

@app.route("/contact")
def contact(): return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)