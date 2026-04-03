import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# ====================== КОНСТАНТЫ ======================
CITIES = ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь",
          "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]

BASE_FEATURES = ["temperature", "rain", "snow", "soil_moisture", "river_level",
                 "snow_melt", "precip_3d", "precip_7d", "temp_rain_inter", "soil_river"]

CITY_FEATURES = [f"city_{c}" for c in CITIES]
FEATURE_COLUMNS = BASE_FEATURES + CITY_FEATURES
INPUT_DIM = len(FEATURE_COLUMNS)  # 20

MODEL_PATH = "models/flood_model.pt"
SCALER_PATH = "models/scaler.pkl"

# ====================== ПЕРЕВОДЫ ======================
T = {
    "ru": {
        "title": "ARGUS • Прогноз паводков", "subtitle": "Акмолинская область",
        "select_city": "Выберите город или район", "temperature": "Температура воздуха, °C",
        "rain": "Осадки (дождь), мм", "snow": "Снежный покров, см",
        "soil": "Влажность почвы (0–1)", "river": "Уровень реки, см",
        "calculate": "Рассчитать риск затопления", "risk": "Риск затопления",
        "low": "Низкий риск", "medium": "Средний риск", "high": "Высокий риск",
        "authors": "Разработано: Шамшидовым Мирасом и Бексултаном Абдыхалыком",
        "supervisor": "Научный руководитель: Ольга Шорникова Николаевна", "year": "2026",
        "creating": "Первый запуск: создаём модель...", "created": "Модель создана!"
    },
    "kk": {
        "title": "ARGUS • Су тасқыны болжамы", "subtitle": "Ақмола облысы",
        "select_city": "Қала немесе ауданды таңдаңыз", "temperature": "Ауа температурасы, °C",
        "rain": "Жаңбыр, мм", "snow": "Қар қабаты, см",
        "soil": "Топырақ ылғалдылығы (0–1)", "river": "Өзен деңгейі, см",
        "calculate": "Су басу қаупін есептеу", "risk": "Су басу қаупі",
        "low": "Төмен қауіп", "medium": "Орташа қауіп", "high": "Жоғары қауіп",
        "authors": "Әзірлеген: Мирас Шамшидов және Бексұлтан Абдыхалық",
        "supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна", "year": "2026",
        "creating": "Бірінші іске қосу: модель жасалуда...", "created": "Модель жасалды!"
    },
    "en": {
        "title": "ARGUS • Flood Prediction", "subtitle": "Akmola Region",
        "select_city": "Select city or district", "temperature": "Air temperature, °C",
        "rain": "Rainfall, mm", "snow": "Snow cover, cm",
        "soil": "Soil moisture (0–1)", "river": "River level, cm",
        "calculate": "Calculate flood risk", "risk": "Flood risk",
        "low": "Low risk", "medium": "Medium risk", "high": "High risk",
        "authors": "Developed by: Miras Shamshidov and Beksultan Abdykhalyk",
        "supervisor": "Scientific supervisor: Olga Shornikova Nikolaevna", "year": "2026",
        "creating": "First run: creating model...", "created": "Model created!"
    }
}

# ====================== МОДЕЛЬ ======================
class FloodModel(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ====================== ИНЖЕНЕРИЯ ПРИЗНАКОВ ======================
def engineer_features(temp: float, rain: float, snow: float,
                       soil: float, river: float, city: str) -> np.ndarray:
    """Собирает вектор признаков в правильном порядке — всегда."""
    row = np.zeros(INPUT_DIM, dtype=np.float32)
    vals = {
        "temperature": temp,
        "rain": rain,
        "snow": snow,
        "soil_moisture": soil,
        "river_level": river,
        "snow_melt": max(0.0, temp) * snow * 0.12,
        "precip_3d": rain * 3,
        "precip_7d": rain * 7,
        "temp_rain_inter": temp * rain,
        "soil_river": soil * river,
        f"city_{city}": 1.0,
    }
    for i, col in enumerate(FEATURE_COLUMNS):
        row[i] = vals.get(col, 0.0)
    return row.reshape(1, -1)

# ====================== ЗАГРУЗКА / СОЗДАНИЕ ======================
@st.cache_resource
def load_or_create_model():
    from sklearn.preprocessing import StandardScaler

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.info(T["ru"]["creating"])
        np.random.seed(42)
        n = 5000

        temp  = np.random.normal(3.5, 12, n)
        rain  = np.random.exponential(8, n).clip(0, 55)
        snow  = np.random.exponential(15, n).clip(0, 45)
        soil  = np.random.beta(3, 2, n)
        river = np.random.normal(52, 9, n).clip(35, 75)
        city_idx = np.random.randint(0, len(CITIES), n)

        X = np.zeros((n, INPUT_DIM), dtype=np.float32)
        for i, (t_, r_, s_, so_, rv_, ci) in enumerate(
                zip(temp, rain, snow, soil, river, city_idx)):
            X[i] = engineer_features(t_, r_, s_, so_, rv_, CITIES[ci])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)

        # Простая синтетическая метка: чем выше river+snow_melt+soil → тем выше риск
        y = torch.sigmoid(torch.tensor(
            (X_scaled[:, 4] + X_scaled[:, 5] + X_scaled[:, 3]) / 3,
            dtype=torch.float32
        )).unsqueeze(1)

        model = FloodModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = criterion(model(X_t), y)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PATH)
        st.success(T["ru"]["created"])

    # Загрузка
    scaler = joblib.load(SCALER_PATH)
    model = FloodModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, scaler

# ====================== СТИЛИ ======================
st.set_page_config(page_title="ARGUS", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    .main-header { font-size: 2.8rem; font-weight: 700; color: #0f172a; }
    .glass {
        background: white; border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08); padding: 28px;
    }
    .stButton>button { background:#0284c8; color:white; height:52px; font-size:1.1rem; }
    .stButton>button:hover { background:#0369a1; }
</style>
""", unsafe_allow_html=True)

# ====================== ОСНОВНОЙ КОД ======================
model, scaler = load_or_create_model()

# Сайдбар
lang_map = {"Русский": "ru", "Қазақша": "kk", "English": "en"}
lang = st.sidebar.selectbox("Язык / Тіл / Language", list(lang_map))
t = T[lang_map[lang]]

with st.sidebar:
    st.title("ARGUS")
    st.caption("Система прогнозирования паводков")
    city = st.selectbox(t["select_city"], CITIES)

# Заголовок
st.markdown(f"<h1 class='main-header'>{t['title']}</h1>", unsafe_allow_html=True)
st.subheader(f"{t['subtitle']} — {city}")

# Слайдеры
c1, c2, c3 = st.columns(3)
with c1: temp  = st.slider(t["temperature"], -35.0, 40.0, 5.0,  step=0.5)
with c2: rain  = st.slider(t["rain"],          0.0, 60.0, 5.0,  step=0.1)
with c3: snow  = st.slider(t["snow"],          0.0, 45.0, 0.0,  step=0.1)

c4, c5 = st.columns(2)
with c4: soil  = st.slider(t["soil"],  0.0, 1.0,  0.3, step=0.01)
with c5: river = st.slider(t["river"], 35.0, 80.0, 50.0, step=0.1)

# Предсказание
if st.button(t["calculate"], type="primary", use_container_width=True):
    x_raw    = engineer_features(temp, rain, snow, soil, river, city)
    x_scaled = scaler.transform(x_raw)          # numpy → numpy, нет проблем с именами
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(x_tensor)).item()

    risk = round(prob * 100, 1)
    if risk < 30:
        status, color = t["low"],    "#22c55e"
    elif risk < 65:
        status, color = t["medium"], "#eab308"
    else:
        status, color = t["high"],   "#ef4444"

    st.markdown(f"""
    <div class="glass" style="border-left: 8px solid {color}; margin-top: 24px;">
        <h2 style="color:{color}; margin:0;">{status}</h2>
        <h1 style="font-size:4.8rem; margin:15px 0; color:#0f172a;">{risk}%</h1>
        <p style="margin:0; font-size:1.1rem; color:#64748b;">{t['risk']}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption(t["authors"])
st.caption(t["supervisor"])
st.caption(f"© {t['year']}")
