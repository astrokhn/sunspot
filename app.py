# Streamlit 앱: YOLOv5로 태양 흑점 탐지 + 날씨/온도/습도/관측 장소 기록 + + Imgur 이미지 업로드 + Notion 업로드

import streamlit as st

st.set_page_config(page_title="태양 흑점 관측 일지", layout="centered")

import os
from PIL import Image
import tempfile
from notion_client import Client
from datetime import datetime
import requests

# Notion 연동 설정
# NOTION_API_KEY = st.secrets["notion_api_key"]
# NOTION_DB_ID = st.secrets["notion_db_id"]
# notion = Client(auth=NOTION_API_KEY)

# Notion 연동 설정
notion = Client(auth=st.secrets["notion_api_key"])
NOTION_DB_ID = st.secrets["notion_db_id"]
IMGUR_CLIENT_ID = st.secrets["imgur_client_id"]
WEATHER_API_KEY = st.secrets["weather_api_key"]


# 이미지 → Imgur 업로드
def upload_image_to_imgur(image_path, client_id):
    headers = {"Authorization": f"Client-ID {client_id}"}
    with open(image_path, "rb") as f:
        data = {"image": f.read()}
    response = requests.post(
        "https://api.imgur.com/3/image", headers=headers, files=data
    )
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        raise Exception(f"Imgur upload failed: {response.status_code}, {response.text}")


# 위치 자동 감지
@st.cache_data
def get_ip_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("city", "Seoul")
    except:
        return "Seoul"


# 날씨 정보 가져오기 (OpenWeatherMap API 필요)
def get_weather_info(city):
    api_key = st.secrets["weather_api_key"]
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&lang=kr&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        weather_description = data["weather"][0]["description"]
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return weather_description, temperature, humidity
    except:
        return "알 수 없음", 0, 0


# 날씨 설명에 따른 이모지 제공
def get_weather_emoji(description):
    if "맑" in description:
        return "☀️"
    elif "구름" in description:
        return "☁️"
    elif "비" in description:
        return "🌧️"
    elif "눈" in description:
        return "❄️"
    elif "안개" in description:
        return "🌫️"
    else:
        return "🌈"


# 이미지 파일을 Imgur에 업로드 후 URL 반환
# def upload_image_to_notion(image: Image.Image, name: str) -> str:
#    buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
#    image.save(buffer.name, format="JPEG")
#    client_id = st.secrets["imgur_client_id"]
#    imgur_url = upload_image_to_imgur(buffer.name, client_id)
#    return imgur_url


def create_notion_page(
    user_name, location, weather, temperature, humidity, memo, orig_img_url
):
    today = datetime.today().strftime("%Y-%m-%d")
    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "이름": {"title": [{"text": {"content": user_name}}]},
            "관측 날짜": {"date": {"start": today}},
            "관측 장소": {"rich_text": [{"text": {"content": location}}]},
            "날씨": {"rich_text": [{"text": {"content": weather}}]},
            "온도(℃)": {"number": temperature},
            "습도(%)": {"number": humidity},
            "관측 메모": {"rich_text": [{"text": {"content": memo}}]},
            "사진 URL": {"url": orig_img_url},  # 데이터베이스 칼럼에 이미지 URL 추가
        },
    )
    page_id = page["id"]

    # 템플릿 제목2 블록에 이미지 삽입
    def find_heading_block_id(keyword):
        blocks = notion.blocks.children.list(page_id)
        for block in blocks["results"]:
            if block["type"] == "heading_2":
                texts = block["heading_2"].get("rich_text", [])
                if texts and keyword in texts[0]["text"]["content"]:
                    return block["id"]
        return None

    orig_block = find_heading_block_id("🌞 내가 찍은 태양 사진")
    if orig_block:
        notion.blocks.children.append(
            block_id=orig_block,
            children=[
                {
                    "object": "block",
                    "type": "image",
                    "image": {"type": "external", "external": {"url": orig_img_url}},
                }
            ],
        )

    return page_id


# UI 시작
st.title("🌞 태양 관측 기록기")
st.write("태양 사진과 관측 정보를 기록하고 Notion에 자동 저장할 수 있어요.")

uploaded_file = st.file_uploader(
    "태양 이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
)
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    image = Image.open(tmp_path).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    user_name = st.text_input("이름을 입력하세요")
    auto_city = get_ip_location()
    location = st.text_input(
        "관측 장소를 영어로 입력하세요 (자동 감지됨, 수정 가능, 예시: Seoul, Suwon)",
        value=auto_city,
    )

    weather_description, temperature, humidity = get_weather_info(location)
    emoji = get_weather_emoji(weather_description)

    st.info(f"📍 현재 위치: {location}")
    st.info(f"🌤️ 날씨: {weather_description} {emoji}")
    st.info(f"🌡️ 온도: {temperature}°C | 💧 습도: {humidity}%")

    memo = st.text_area("오늘의 태양 관측 메모를 남겨보세요:")

    if st.button("Notion에 기록 저장"):
        if user_name.strip() == "":
            st.warning("이름을 입력해주세요.")
        else:
            imgur_url = upload_image_to_imgur(tmp_path, IMGUR_CLIENT_ID)
            page_id = create_notion_page(
                user_name,
                location,
                f"{weather_description} {emoji}",
                temperature,
                humidity,
                memo,
                imgur_url,
            )
            notion_url = f"https://www.notion.so/{page_id.replace('-', '')}"
            st.success("✅ Notion에 기록이 저장되었습니다.")
            st.markdown(f"🔗 [기록 보기]({notion_url})")
