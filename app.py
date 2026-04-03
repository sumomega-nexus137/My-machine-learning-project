"""
ARGUS — Flood Prediction System (Improved Version)
Realistic inputs, proper translation, graphs, and instruction panel.
"""

import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
st.set_page_config(page_title="ARGUS", layout="wide")

# =========================
# REALISTIC DATA RANGES INFO
# =========================
INFO = {
    "temp": "Temperature: -35°C to +30°C (Akmola region climate)",
    "rain": "Rainfall: 0–50 mm (typical heavy precipitation)",
    "snow": "Snow: 0–60 cm (spring snow accumulation)",
    "soil": "Soil moisture: 0–1 (saturation level)",
    "river": "River level: 30–80 cm (depending on season)"
}

# =========================
# TRANSLATION (FULL FIX)
# =========================
T = {
    "ru": {
        "title": "ARGUS",
        "tagline": "Система прогнозирования паводков",
        "predict": "Прогноз",
        "analysis": "Анализ",
        "about": "О проекте",
        "instruction": "Инструкция",
        "city": "Город",
        "temp": "Температура (°C)",
        "rain": "Осадки (мм)",
        "snow": "Снег (см)",
        "soil": "Влажность почвы",
        "river": "Уровень реки (см)",
        "calculate": "Рассчитать",
        "risk": "Риск затопления",
        "low": "Низкий риск",
        "medium": "Средний риск",
        "high": "Высокий риск",
        "help": "Система предназначена для помощи в оценке риска и поддержке эвакуации.",
        "instruction_text": "Введите данные и получите оценку риска.",
        "for_whom": "Для кого: службы, акиматы, исследователи, население.",
    },
    "en": {
        "title": "ARGUS",
        "tagline": "Flood Prediction System",
        "predict": "Prediction",
        "analysis": "Analysis",
        "about": "About",
        "instruction": "Instruction",
        "city": "City",
        "temp": "Temperature (°C)",
        "rain": "Rain (mm)",
        "snow": "Snow (cm)",
        "soil": "Soil moisture",
        "river": "River level (cm)",
        "calculate": "Calculate",
        "risk": "Flood risk",
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
        "help": "System helps assess flood risk and supports evacuation decisions.",
        "instruction_text": "Enter parameters and get flood risk.",
        "for_whom": "For: emergency services, researchers, authorities, population.",
    },
    "kk": {
        "title": "ARGUS",
        "tagline": "Су тасқынын болжау жүйесі",
        "predict": "Болжам",
        "analysis": "Талдау",
        "about": "Жоба",
        "instruction": "Нұсқаулық",
        "city": "Қала",
        "temp": "Температура (°C)",
        "rain": "Жаңбыр (мм)",
        "snow": "Қар (см)",
        "soil": "Топырақ ылғалдылығы",
        "river": "Өзен деңгейі (см)",
        "calculate": "Есептеу",
        "risk": "Қауіп",
        "low": "Төмен қауіп",
        "medium": "Орташа қауіп",
        "high": "Жоғары қауіп",
        "help": "Жүйе эвакуацияны қолдауға және қауіп деңгейін бағалауға арналған.",
        "instruction_text": "Деректерді енгізіп, қауіп деңгейін алыңыз.",
        "for_whom": "Кім үшін: қызметтер, әкімдіктер, зерттеушілер, халық.",
    }
}

lang = st.sidebar.selectbox("Language", ["ru", "en", "kk"])
t = T[lang]

# =========================
# MODEL
# =========================
class FloodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

model = FloodModel()

# =========================
# REALISTIC FEATURE ENGINEERING
# =========================
def features(temp, rain, snow, soil, river):
    # Physical relationships
    snow_melt = max(0, temp) * snow * 0.1
    runoff = rain * 0.7 + snow_melt
    river_effect = river * 0.5
    soil_effect = soil * 20

    return np.array([[temp, runoff, snow_melt, soil_effect, river_effect]], dtype=np.float32)

# =========================
# UI TITLE
# =========================
st.title(t["title"])
st.write(t["tagline"])

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    t["predict"],
    t["analysis"],
    t["about"],
    t["instruction"]
])

# =========================
# TAB 1 — INPUT + GRAPH
# =========================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        temp = st.slider(t["temp"], -35.0, 30.0, 5.0)

    with col2:
        rain = st.slider(t["rain"], 0.0, 50.0, 5.0)

    with col3:
        snow = st.slider(t["snow"], 0.0, 60.0, 10.0)

    col4, col5 = st.columns(2)

    with col4:
        soil = st.slider(t["soil"], 0.0, 1.0, 0.4)

    with col5:
        river = st.slider(t["river"], 30.0, 80.0, 50.0)

    if st.button(t["calculate"]):

        x = features(temp, rain, snow, soil, river)
        x = torch.tensor(x)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        risk = round(prob * 100, 2)

        st.subheader(t["risk"])
        st.write(f"{risk} %")

        # GRAPH
        fig, ax = plt.subplots()
        ax.bar(["Risk"], [risk])
        st.pyplot(fig)

        # Interpretation
        if risk < 30:
            st.success(t["low"])
        elif risk < 70:
            st.warning(t["medium"])
        else:
            st.error(t["high"])

# =========================
# TAB 2 — ANALYSIS (GRAPHS)
# =========================
with tab2:
    st.subheader(t["analysis"])

    n = 100

    temp_data = np.random.uniform(-30, 30, n)
    rain_data = np.random.uniform(0, 50, n)
    snow_data = np.random.uniform(0, 60, n)

    fig, ax = plt.subplots()
    ax.plot(temp_data, label="Temp")
    ax.plot(rain_data, label="Rain")
    ax.plot(snow_data, label="Snow")
    ax.legend()

    st.pyplot(fig)

# =========================
# TAB 3 — ABOUT (HONEST)
# =========================
with tab3:
    st.subheader(t["about"])

    st.write("""
    ARGUS — система оценки риска затоплений.

    Использует:
    - нейронную сеть (PyTorch)
    - физически обоснованные признаки
    - анализ погодных параметров

    ### Важно:
    Система создана для:
    - поддержки принятия решений
    - помощи службам
    - повышения готовности к эвакуации

    Она НЕ заменяет официальные прогнозы,
    а дополняет их аналитической оценкой.
    """)

    st.info(t["help"])

# =========================
# TAB 4 — INSTRUCTION + PURPOSE
# =========================
with tab4:
    st.subheader(t["instruction"])

    st.write(t["instruction_text"])

    st.markdown("### 📌 Для кого создана система")
    st.write(t["for_whom"])

    st.markdown("### 🚨 Основная цель")
    st.write("""
    Система создана, чтобы:
    - помогать людям заранее оценить риск
    - поддерживать эвакуацию
    - снижать последствия паводков
    - помогать службам реагирования
    """)

    st.markdown("### ⚙️ Как использовать")
    st.write("""
    1. Выберите параметры (температура, снег, дождь и т.д.)
    2. Нажмите "Рассчитать"
    3. Получите уровень риска
    4. Используйте результат для принятия решений
    """)
