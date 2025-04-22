import numpy as np
import requests
import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json

# Khởi tạo requests session và headers
session = requests.Session()
edge_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
session.headers.update({
    "User-Agent": edge_user_agent
})

# Cấu hình Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# Regex
SKILL_RE = re.compile(r'("skills"\s*:\s*\[(.*?)\])|("What you\'ll learn"\s*:\s*\[(.*?)\])', re.S)

# Đọc link
file_path = "coursera_course_urls.txt"
with open(file_path, "r", encoding="utf-8") as f:
    course_links = [line.strip() for line in f if line.strip()]

data = []
for idx, link in enumerate(course_links):
    try:
        driver.get(link)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        html = driver.page_source
    except:
        print(f"⚠️ Không load được trang: {link}")
        continue

    time.sleep(1)

    def safe_find_all(div_list, index, class_name):
        try:
            return div_list[index].find("div", class_=class_name)
        except:
            return None

    def get_text_by_xpath(xpath):
        try:
            text = driver.find_element(By.XPATH, xpath).text
            return text if text.strip() else np.nan
        except:
            return np.nan

    def get_content_from_meta(xpath):
        try:
            content = driver.find_element(By.XPATH, xpath).get_attribute("content")
            return content if content.strip() else np.nan
        except:
            return np.nan

    # Lấy HTML bằng requests để tìm skills
    try:
        driver.get(link)
        html = driver.page_source
        skills_blk = SKILL_RE.search(html)
        if skills_blk:
            skills = ", ".join(json.loads("[" + skills_blk.group(2) + "]")) if skills_blk.group(2) else \
                     ", ".join(json.loads("[" + skills_blk.group(4) + "]")) if skills_blk.group(4) else np.nan
        else:
            skills = np.nan
    except Exception as e:
        print(f"⚠️ Lỗi khi lấy skills cho {link}: {e}")
        skills = np.nan

    # Nếu không lấy được skills bằng requests, thử lại bằng BeautifulSoup
    if skills is np.nan:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        spans = soup.select("ul.css-yk0mzy span.css-o5tswl")
        if spans:
            skills = [s.get_text(strip=True) for s in spans if s.get_text(strip=True)]
        else:
            divs = soup.find_all("div", attrs={"data-testid": "visually-hidden"})
            for div in divs:
                if "Category:" in div.text:
                    skill = div.text.split("Category:")[-1].strip()
                    if skill:
                        skills.append(skill)
        skills = skills if skills else np.nan

    # === Thu thập các thông tin khác ===
    try:
        title = driver.find_element(By.TAG_NAME, "h1").text.strip()
        title = title if title else np.nan
    except:
        title = np.nan

    soup = BeautifulSoup(driver.page_source, "html.parser")
    provider_div = soup.find("div", class_="css-15g7tpu")
    organization = np.nan
    if provider_div:
        span = provider_div.find("span", class_="css-6ecy9b")
        if span:
            organization = span.text.strip() if span.text.strip() else np.nan

    instructor = get_text_by_xpath('/html/body/div[2]/div/main/section[2]/div/div/div[1]/div[1]/div/div/div[2]/div[2]/div/div[2]/div[1]/p/span/a/span')
    description = get_content_from_meta('/html/head/meta[22]')
    learners = get_text_by_xpath('/html/body/div[2]/div/main/section[2]/div/div/div[1]/div[1]/div/div/div[2]/div[4]/p/span/strong/span')

    inner_blocks = soup.find_all("div", class_="css-dwgey1")
    level = np.nan
    for i, block in enumerate(inner_blocks):
        level_div = safe_find_all(inner_blocks, i, "css-fk6qfz")
        text = level_div.text.strip().lower() if level_div else ""
        if "level" in text:
            level = level_div.text.strip() if level_div else np.nan
            break

    duration_div = safe_find_all(inner_blocks, 3, "css-fk6qfz")
    duration_info = duration_div.text.strip() if duration_div and duration_div.text.strip() else np.nan

    rating_div = soup.find("div", attrs={"aria-label": lambda v: v and "stars" in v})
    rating = rating_div.text.strip() if rating_div and rating_div.text.strip() else np.nan

    try:
        elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="cml-viewer"]')
        lessons = [el.text.strip() for el in elements if el.text.strip()]
        lessons = lessons if lessons else np.nan
    except:
        lessons = np.nan

    # Ghi vào dataset
    data.append({
        'title': title,
        'organization': organization,
        'instructor': instructor,
        'duration_info': duration_info,
        'level': level,
        'description': description,
        'lessons': lessons,
        'skills': skills,
        'rating': rating,
        'learners': learners,
        'link': link
    })

    print(f"✅ Done {idx+1}/{len(course_links)}: {title}")

# Đóng driver sau khi hoàn thành
driver.quit()

df = pd.DataFrame(data)
df.to_csv("coursera_course_data.csv", index=False, encoding="utf-8-sig")