# Streamlit 앱: YOLOv5로 태양 흑점 탐지 + 사용자 이름 기반 Notion 기록

import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
from notion_client import Client
from datetime import datetime

# Notion 연동 설정
NOTION_API_KEY = "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"
NOTION_DB_ID = "1f0ed1871cbb800f86eac3271c4c1f22"
notion = Client(auth=NOTION_API_KEY)


def save_to_notion(user_name, sunspot_count, memo):
    today = datetime.today().strftime("%Y-%m-%d")
    notion.pages.create(
        parent={"database_id": NOTION_DB_ID},
        properties={
            "이름": {"title": [{"text": {"content": user_name}}]},
            "관측 날짜": {"rich_text": [{"text": {"content": today}}]},
            "흑점 개수": {"number": sunspot_count},
            "관측 메모": {"rich_text": [{"text": {"content": memo}}]},
        },
    )


# YOLO 모델 로드
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
    return model


model = load_model()

# Streamlit UI
st.set_page_config(page_title="태양 흑점 AI 탐지기", layout="centered")
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

    image = Image.open(tmp_path).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    results = model(image)
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

    # 사용자 입력 및 Notion 저장
    st.subheader("📓 관측 일기 기록")
    user_name = st.text_input("이름을 입력하세요")
    memo = st.text_area("오늘의 태양 관측 메모를 남겨보세요:")
    if st.button("Notion에 기록 저장"):
        if user_name.strip() == "":
            st.warning("이름을 입력해주세요.")
        else:
            save_to_notion(user_name, count, memo)
            st.success("✅ Notion에 기록이 저장되었습니다.")
else:
    st.info("이미지를 업로드하면 흑점 탐지가 시작됩니다.")
