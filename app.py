# Streamlit 앱: YOLOv5로 태양 흑점 탐지 + 날씨/온도/습도/관측 장소 기록 + + Imgur 이미지 업로드 + Notion 업로드

import streamlit as st

st.set_page_config(page_title="태양 흑점 AI 탐지기", layout="centered")

import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
from notion_client import Client
from datetime import datetime
import base64
import requests

# Notion 연동 설정
NOTION_API_KEY = st.secrets["notion_api_key"]
NOTION_DB_ID = st.secrets["notion_db_id"]
notion = Client(auth=NOTION_API_KEY)


# Imgur 업로드 함수
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


# 사용자 도시 위치 가져오기 (IP 기반)
def get_ip_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("city", "서울")
    except:
        return "서울"


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
def upload_image_to_notion(image: Image.Image, name: str) -> str:
    buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(buffer.name, format="JPEG")
    client_id = st.secrets["imgur_client_id"]
    imgur_url = upload_image_to_imgur(buffer.name, client_id)
    return imgur_url


# Notion에 페이지 생성
def create_notion_page(
    user_name,
    sunspot_count,
    location,
    weather,
    temperature,
    humidity,
    memo,
    orig_img_url,
    result_img_url,
):
    today = datetime.today().strftime("%Y-%m-%d")
    page = notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "이름": {"title": [{"text": {"content": user_name}}]},
            "관측 날짜": {"date": {"start": today}},
            "관측 장소": {"rich_text": [{"text": {"content": location}}]},
            "흑점 개수": {"number": sunspot_count},
            "날씨": {"rich_text": [{"text": {"content": weather}}]},
            "온도(℃)": {"number": temperature},
            "습도(%)": {"number": humidity},
            "관측 메모": {"rich_text": [{"text": {"content": memo}}]},
        },
    )
    page_id = page["id"]

    # ✅ 템플릿 내 heading_2 블록 중 키워드 포함된 블록 ID 찾기
    def find_heading_block_id(keyword):
        blocks = notion.blocks.children.list(page_id)
        for block in blocks["results"]:
            if block["type"] == "heading_2":
                texts = block["heading_2"].get("rich_text", [])
                if texts and keyword in texts[0]["text"]["content"]:
                    return block["id"]
        return None

    # 🔍 각 템플릿 제목2 위치 찾기
    orig_block = find_heading_block_id("🌞 내가 찍은 태양 사진")
    ai_block = find_heading_block_id("🤖 AI가 분석한 태양 사진")

    # 🖼 원본 이미지 삽입
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

    # 🤖 AI 분석 이미지 삽입
    if ai_block:
        notion.blocks.children.append(
            block_id=ai_block,
            children=[
                {
                    "object": "block",
                    "type": "image",
                    "image": {"type": "external", "external": {"url": result_img_url}},
                }
            ],
        )

    return page_id


@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")


model = load_model()

st.title("🌞 AI가 본 태양")
st.write(
    "휴대폰으로 촬영한 태양 사진을 업로드하면, AI가 흑점을 찾아주고 개수를 알려줍니다."
)

uploaded_file = st.file_uploader(
    "태양 이미지를 업로드하세요 (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    orig_image = Image.open(tmp_path).convert("RGB")
    st.image(orig_image, caption="업로드한 이미지", use_column_width=True)

    results = model(orig_image)
    results.render()

    result_np = results.ims[0]
    result_rgb = cv2.cvtColor(result_np, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)
    st.image(result_img, caption="AI 탐지 결과", use_column_width=True)

    df = results.pandas().xyxy[0]
    count = len(df)
    st.success(f"✅ 탐지된 흑점 개수: {count}개")

    result_img.save("sunspot_result.jpg")
    with open("sunspot_result.jpg", "rb") as f:
        st.download_button(
            "📥 결과 이미지 다운로드",
            f,
            file_name="sunspot_result.jpg",
            mime="image/jpeg",
        )

    st.subheader("📓 관측 일기 기록")
    user_name = st.text_input("이름을 입력하세요")

    auto_city = get_ip_location()
    location = st.text_input(
        "관측 장소를 영어로 입력하세요 (자동 감지됨, 수정 가능, 예시: Seoul, Suwon)",
        value=auto_city,
    )

    weather_description, temperature, humidity = get_weather_info(location)
    emoji = get_weather_emoji(weather_description)

    st.info(f"📍 현재 위치: {location}")
    st.info(f"🌤️ 현재 날씨: {weather_description} {emoji}")
    st.info(f"🌡️ 온도: {temperature}°C | 💧 습도: {humidity}%")

    memo = st.text_area("오늘의 태양 관측 메모를 남겨보세요:")

    if st.button("Notion에 기록 저장"):
        if user_name.strip() == "":
            st.warning("이름을 입력해주세요.")
        else:
            orig_url = upload_image_to_notion(orig_image, "원본 태양 이미지")
            result_url = upload_image_to_notion(result_img, "AI 탐지 결과")
            page_id = create_notion_page(
                user_name,
                count,
                location,
                f"{weather_description} {emoji}",
                temperature,
                humidity,
                memo,
                orig_url,
                result_url,
            )
            st.success("✅ Notion에 기록이 저장되었습니다.")
            notion_url = f"https://www.notion.so/{page_id.replace('-', '')}"
            st.markdown(f"🔗 [Notion에서 기록 보기]({notion_url})")
else:
    st.info("이미지를 업로드하면 흑점 탐지가 시작됩니다.")
