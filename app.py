import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
st.set_page_config(page_title="ARGUS", layout="wide")

# =========================
# TRANSLATION FIX (FULL)
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
        "soil": "Влажность почвы (%)",
        "river": "Уровень реки (см)",
        "calculate": "Рассчитать",
        "risk": "Риск затопления",
        "low": "Низкий риск",
        "medium": "Средний риск",
        "high": "Высокий риск",
        "instruction_text": "Введите данные и получите прогноз риска.",
        "for_whom": "Для служб, акиматов, населения.",
    },
    "en": {
        "title": "ARGUS",
        "tagline": "Flood Prediction System",
        "predict": "Prediction",
        "analysis": "Analysis",
        "about": "About",
        "instruction": "Instruction",
        "temp": "Temperature (°C)",
        "rain": "Rain (mm)",
        "snow": "Snow (cm)",
        "soil": "Soil moisture (%)",
        "river": "River level (cm)",
        "calculate": "Calculate",
        "risk": "Flood risk",
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
        "instruction_text": "Enter data and get prediction.",
        "for_whom": "For services, authorities, population.",
    }
}

lang = st.sidebar.selectbox("Language", ["ru", "en"])
t = T[lang]

# =========================
# MODEL
# =========================
class FloodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = FloodModel()

# =========================
# FEATURES (NOW WITH %)
# =========================
def features(temp, rain, snow, soil_percent, river):
    soil = soil_percent / 100.0

    snow_melt = max(0, temp) * (snow / 100) * 0.2
    runoff = rain * 0.8 + snow_melt
    soil_effect = soil * 30
    river_effect = river * 0.6

    return np.array([[temp, runoff, snow_melt, soil_effect, river_effect]], dtype=np.float32)

# =========================
# UI
# =========================
st.title(t["title"])
st.write(t["tagline"])

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([t["predict"], t["analysis"], t["about"]])

# =========================
# TAB 1
# =========================
with tab1:
    temp = st.slider(t["temp"], -50.0, 45.0, 10.0)
    rain = st.slider(t["rain"], 0.0, 3000.0, 50.0)
    snow = st.slider(t["snow"], 0.0, 300.0, 20.0)
    soil_percent = st.slider(t["soil"], 0.0, 100.0, 30.0)
    river = st.slider(t["river"], 10.0, 300.0, 80.0)

    if st.button(t["calculate"]):
        x = features(temp, rain, snow, soil_percent, river)
        x = torch.tensor(x)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        risk = round(prob * 100, 2)

        st.subheader(t["risk"])
        st.write(f"{risk} %")

        fig, ax = plt.subplots()
        ax.bar(["Risk"], [risk])
        st.pyplot(fig)

        if risk < 30:
            st.success(t["low"])
        elif risk < 70:
            st.warning(t["medium"])
        else:
            st.error(t["high"])

# =========================
# TAB 2 (IMAGES + ANALYSIS)
# =========================
with tab2:
    st.subheader(t["analysis"])

    st.markdown("### 📸 Примеры паводков и разрушений")

    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3c/Flood_damage.jpg", caption="Flood damage")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Flood_house.jpg", caption="Flood impact")
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Flood_river_overflow.jpg", caption="River overflow")

    st.markdown("### 📊 График")
    data = np.random.uniform(0, 3000, 100)

    fig, ax = plt.subplots()
    ax.plot(data)
    st.pyplot(fig)

# =========================
# TAB 3 (FULL INFO)
# =========================
with tab3:
    st.subheader(t["about"])

    st.markdown("""
    ### 👨‍💻 Разработчики проекта:
    - Шамшидов Мирас
    - Бексултан Абдыхалык

    ### 👨‍🏫 Научный руководитель:
    - Ольга Шорникова Николаевна

    ### 📅 Год разработки:
    - 2026

    ### 🎯 Описание:
    ARGUS — система прогнозирования паводков,
    разработанная для оценки риска и поддержки принятия решений.

    Использует нейронную сеть на PyTorch и физические зависимости.
    """)
