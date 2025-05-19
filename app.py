# Gradio + YOLOv5 + Imgur + Notion 통합 앱 (Hugging Face Spaces용)

import os
import gradio as gr
import torch
from PIL import Image
import numpy as np
import tempfile
import requests
from datetime import datetime
from notion_client import Client
import json
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Notion client
try:
    notion = Client(auth=os.getenv("NOTION_TOKEN"))
    DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    if not DATABASE_ID:
        raise ValueError("NOTION_DATABASE_ID not found in environment variables")
except Exception as e:
    logger.error(f"Error initializing Notion client: {str(e)}")
    gr.Warning(
        "Failed to initialize Notion client. Please check your environment variables."
    )

# --- Imgur 설정 ---
IMGUR_CLIENT_ID = os.environ.get("imgur_client_id")

# --- 날씨 설정 ---
WEATHER_API_KEY = os.environ.get("weather_api_key")


# Load YOLOv5 model
@gr.cache_resource
def load_model():
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        gr.Warning(
            "Failed to load the YOLOv5 model. Please check if the model file exists."
        )
        return None


model = load_model()


def get_ip_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        return res.get("city", "Seoul")
    except:
        return "Seoul"


def get_weather_info(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&lang=kr&units=metric"
        res = requests.get(url).json()
        weather = res["weather"][0]["description"]
        temp = res["main"]["temp"]
        hum = res["main"]["humidity"]
        return weather, temp, hum
    except:
        return "알 수 없음", 0, 0


def get_weather_emoji(desc):
    if "맑" in desc:
        return "☀️"
    if "구름" in desc:
        return "☁️"
    if "비" in desc:
        return "🌧️"
    if "눈" in desc:
        return "❄️"
    return "🌈"


def upload_to_imgur(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, format="JPEG")
        with open(tmp.name, "rb") as f:
            headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
            data = {"image": f.read()}
            res = requests.post(
                "https://api.imgur.com/3/image", headers=headers, files=data
            )
            return res.json()["data"]["link"]


def create_notion_page(
    user, count, location, weather, temp, hum, memo, orig_url, result_url
):
    today = datetime.today().strftime("%Y-%m-%d")
    page = notion.pages.create(
        parent={"database_id": DATABASE_ID},
        properties={
            "Name": {"title": [{"text": {"content": f"Sunspot Observation - {user}"}}]},
            "Date": {"date": {"start": today}},
            "Location": {"rich_text": [{"text": {"content": location}}]},
            "Weather": {
                "rich_text": [
                    {
                        "text": {
                            "content": json.dumps(
                                {
                                    "description": weather,
                                    "temperature": temp,
                                    "humidity": hum,
                                }
                            )
                        }
                    }
                ]
            },
        },
    )
    page_id = page["id"]

    def find_block(keyword):
        blocks = notion.blocks.children.list(page_id)
        for block in blocks["results"]:
            if block["type"] == "heading_2":
                txt = block["heading_2"].get("rich_text", [])
                if txt and keyword in txt[0]["text"]["content"]:
                    return block["id"]
        return None

    orig_block = find_block("🌞 내가 찍은 태양 사진")
    ai_block = find_block("🤖 AI가 분석한 태양 사진")

    if orig_block:
        notion.blocks.children.append(
            block_id=orig_block,
            children=[
                {
                    "object": "block",
                    "type": "image",
                    "image": {"type": "external", "external": {"url": orig_url}},
                }
            ],
        )
    if ai_block:
        notion.blocks.children.append(
            block_id=ai_block,
            children=[
                {
                    "object": "block",
                    "type": "image",
                    "image": {"type": "external", "external": {"url": result_url}},
                }
            ],
        )

    return f"https://www.notion.so/{page_id.replace('-', '')}"


def app_fn(image, name, memo, location):
    results = model(image)
    results.render()
    bgr = results.ims[0]
    rgb = bgr[..., ::-1]
    result_img = Image.fromarray(rgb)
    count = len(results.pandas().xyxy[0])

    city = location or get_ip_location()
    weather_desc, temp, hum = get_weather_info(city)
    emoji = get_weather_emoji(weather_desc)
    weather = f"{weather_desc} {emoji}"

    orig_url = upload_to_imgur(image)
    result_url = upload_to_imgur(result_img)

    notion_link = create_notion_page(
        name, count, city, weather, temp, hum, memo, orig_url, result_url
    )

    label = f"✅ 흑점 {count}개 탐지됨 | 기록 저장 완료"
    return result_img, label, notion_link


app = gr.Interface(
    fn=app_fn,
    inputs=[
        gr.Image(type="pil", label="태양 이미지 업로드"),
        gr.Text(label="이름"),
        gr.Textbox(label="관측 메모"),
        gr.Text(label="관측 도시 (선택)", placeholder="기본은 자동 감지"),
    ],
    outputs=[
        gr.Image(type="pil", label="AI 분석 결과"),
        gr.Label(label="탐지 결과"),
        gr.Textbox(label="🔗 Notion 링크"),
    ],
    title="🌞 태양 흑점 탐지 + Notion 기록기",
    description="YOLOv5로 태양 흑점을 분석하고 결과를 Notion 템플릿에 자동 기록합니다.",
)

app.launch()
