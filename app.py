import streamlit as st
import torch
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(page_title="ARGUS • Паводки", layout="wide")

# Многоязычность
translations = {
    "ru": {"title": "ARGUS • Прогноз паводков", "subtitle": "Акмолинская область", "select_city": "Выберите город", "temperature": "Температура воздуха, °C", "rain": "Осадки (дождь), мм", "snow": "Снежный покров, см", "soil": "Влажность почвы (0–1)", "river": "Уровень реки, см", "calculate": "Рассчитать риск", "risk": "Риск затопления", "low": "Низкий риск", "medium": "Средний риск", "high": "Высокий риск", "timeline": "Прогноз на 7 дней", "authors": "Разработано: Шамшидовым Мирасом и Бексултаном Абдыхалыком", "supervisor": "Научный руководитель: Ольга Шорникова Николаевна", "year": "2026"},
    "kk": {"title": "ARGUS • Су тасқыны болжамы", "subtitle": "Ақмола облысы", "select_city": "Қала немесе ауданды таңдаңыз", "temperature": "Ауа температурасы, °C", "rain": "Жаңбыр, мм", "snow": "Қар қабаты, см", "soil": "Топырақ ылғалдылығы (0–1)", "river": "Өзен деңгейі, см", "calculate": "Су басу қаупін есептеу", "risk": "Су басу қаупі", "low": "Төмен қауіп", "medium": "Орташа қауіп", "high": "Жоғары қауіп", "timeline": "7 күнге болжам", "authors": "Әзірлеген: Мирас Шамшидов және Бексұлтан Абдыхалық", "supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна", "year": "2026"},
    "en": {"title": "ARGUS • Flood Prediction", "subtitle": "Akmola Region", "select_city": "Select city", "temperature": "Air temperature, °C", "rain": "Rainfall, mm", "snow": "Snow cover, cm", "soil": "Soil moisture (0–1)", "river": "River level, cm", "calculate": "Calculate risk", "risk": "Flood risk", "low": "Low risk", "medium": "Medium risk", "high": "High risk", "timeline": "7-day forecast", "authors": "Developed by: Miras Shamshidov and Beksultan Abdykhalyk", "supervisor": "Scientific supervisor: Olga Shornikova Nikolaevna", "year": "2026"}
}

lang = st.sidebar.selectbox("Язык / Тіл / Language", ["Русский", "Қазақша", "English"])
lang_code = {"Русский": "ru", "Қазақша": "kk", "English": "en"}[lang]
t = translations[lang_code]

# Стили
st.markdown("""
<style>
    .stApp {background: linear-gradient(180deg, #0f172a, #1e2937);}
    .glass {background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); border-radius: 20px; border: 1px solid rgba(255,255,255,0.1);}
</style>
""", unsafe_allow_html=True)

# Загрузка модели и scaler
@st.cache_resource
def load_model_and_scaler():
    model = torch.load("models/flood_model.pt", map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar
with st.sidebar:
    st.title("ARGUS")
    city = st.selectbox(t["select_city"], ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"])

# Главная часть
st.title(t["title"])
st.subheader(f"{t['subtitle']} — {city}")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider(t["temperature"], -35.0, 40.0, 5.0)
with col2:
    rain = st.slider(t["rain"], 0.0, 60.0, 5.0)
with col3:
    snow = st.slider(t["snow"], 0.0, 45.0, 0.0)

col4, col5 = st.columns(2)
with col4:
    soil = st.slider(t["soil"], 0.0, 1.0, 0.3)
with col5:
    river = st.slider(t["river"], 35.0, 80.0, 50.0)

if st.button(t["calculate"], type="primary", use_container_width=True):
    input_dict = {
        "temperature": temp, "rain": rain, "snow": snow, "soil_moisture": soil, "river_level": river,
        "snow_melt": max(0, temp) * snow * 0.12,
        "precip_3d": rain * 3, "precip_7d": rain * 7,
        "temp_rain_inter": temp * rain, "soil_river": soil * river
    }
    for c in ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]:
        input_dict[f"city_{c}"] = 1 if c == city else 0

    df_input = pd.DataFrame([input_dict])
    X_scaled = scaler.transform(df_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(X_tensor)
        prob = torch.sigmoid(logit).item()

    risk = prob * 100
    if risk < 30:
        status, color = t["low"], "#22c55e"
    elif risk < 65:
        status, color = t["medium"], "#eab308"
    else:
        status, color = t["high"], "#ef4444"

    st.markdown(f"""
    <div class="glass" style="padding:30px; text-align:center; border-left:8px solid {color};">
        <h2 style="color:{color};">{status}</h2>
        <h1 style="font-size:4.5rem; margin:10px 0;">{risk:.1f}%</h1>
        <p>{t["risk"]}</p>
    </div>
    """, unsafe_allow_html=True)

# Футер
st.caption(t["authors"])
st.caption(t["supervisor"])
st.caption(f"Год: {t['year']}")
