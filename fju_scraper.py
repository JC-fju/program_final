# fju_scraper.py
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def run_scraper(db, News, app_context):
    """負責爬取系所公告，並將結果直接存入資料庫"""
    
    print("啟動瀏覽器，準備爬取資料...")
    options = Options()
    # 保持非無頭模式，讓視窗正常彈出
    # options.add_argument("--headless") 
    # options.add_argument("--disable-gpu")
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # 減少系統無關警告
    
    driver = webdriver.Chrome(options=options)
    target_url = "https://www.math.fju.edu.tw/zh-hant/news/%E7%B3%BB%E6%89%80%E5%85%AC%E5%91%8A"
    
    try:
        driver.get(target_url)
        time.sleep(2) # 等待網頁載入
        
        rows = driver.find_elements(By.CSS_SELECTOR, "table.category tbody tr")
        
        # 使用 Flask 的推入上下文，確保在獨立的爬蟲函式中也能安全操作資料庫
        with app_context:
            for row in rows[:6]:
                try:
                    # 抓取日期
                    date_text = row.find_element(By.CSS_SELECTOR, "td.list-date").text.strip()
                    date_parts = date_text.split('/')
                    if len(date_parts) == 3:
                        formatted_date = f"{date_parts[1]}-{date_parts[2]}"
                    else:
                        formatted_date = date_text
                    
                    # 抓取標題與連結
                    title_element = row.find_element(By.CSS_SELECTOR, "td.list-title a")
                    title_text = title_element.text.strip()
                    link_href = title_element.get_attribute("href")
                    
                    # 【核心邏輯】檢查此公告是否已存在於資料庫中
                    existing_news = News.query.filter_by(link=link_href).first()
                    
                    if not existing_news:
                        # 資料庫中沒有這條連結，視為新消息，進行新增
                        new_item = News(
                            date=formatted_date,
                            tag="系所公告",
                            title=title_text,
                            link=link_href
                        )
                        db.session.add(new_item)
                        print(f"成功抓取並新增至資料庫: {title_text}")
                    else:
                        # 已經有了，直接略過
                        print(f"提示：公告已存在，略過寫入: {title_text}")
                        
                except Exception as e:
                    print(f"抓取單筆資料時發生錯誤略過: {e}")
                    continue

            # 走訪完畢後，統一提交變更
            db.session.commit()
            print("✅ 資料庫更新程序執行完畢！")
            
    except Exception as e:
        print(f"爬蟲整體執行失敗: {e}")
    finally:
        driver.quit() # 確保瀏覽器關閉