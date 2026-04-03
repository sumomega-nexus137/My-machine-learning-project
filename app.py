import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# ====================== МОДЕЛЬ ======================
class FloodModel(nn.Module):
    def __init__(self, input_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ====================== МНОГОЯЗЫЧНОСТЬ ======================
translations = {
    "ru": {"title": "ARGUS • Прогноз паводков", "subtitle": "Акмолинская область", "select_city": "Выберите город или район", "temperature": "Температура воздуха, °C", "rain": "Осадки (дождь), мм", "snow": "Снежный покров, см", "soil": "Влажность почвы (0–1)", "river": "Уровень реки, см", "calculate": "Рассчитать риск затопления", "risk": "Риск затопления", "low": "Низкий риск", "medium": "Средний риск", "high": "Высокий риск", "authors": "Разработано: Шамшидовым Мирасом и Бексултаном Абдыхалыком", "supervisor": "Научный руководитель: Ольга Шорникова Николаевна", "year": "2026"},
    "kk": {"title": "ARGUS • Су тасқыны болжамы", "subtitle": "Ақмола облысы", "select_city": "Қала немесе ауданды таңдаңыз", "temperature": "Ауа температурасы, °C", "rain": "Жаңбыр, мм", "snow": "Қар қабаты, см", "soil": "Топырақ ылғалдылығы (0–1)", "river": "Өзен деңгейі, см", "calculate": "Су басу қаупін есептеу", "risk": "Су басу қаупі", "low": "Төмен қауіп", "medium": "Орташа қауіп", "high": "Жоғары қауіп", "authors": "Әзірлеген: Мирас Шамшидов және Бексұлтан Абдыхалық", "supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна", "year": "2026"},
    "en": {"title": "ARGUS • Flood Prediction", "subtitle": "Akmola Region", "select_city": "Select city or district", "temperature": "Air temperature, °C", "rain": "Rainfall, mm", "snow": "Snow cover, cm", "soil": "Soil moisture (0–1)", "river": "River level, cm", "calculate": "Calculate flood risk", "risk": "Flood risk", "low": "Low risk", "medium": "Medium risk", "high": "High risk", "authors": "Developed by: Miras Shamshidov and Beksultan Abdykhalyk", "supervisor": "Scientific supervisor: Olga Shornikova Nikolaevna", "year": "2026"}
}

# ====================== СТИЛИ (светлый чистый дизайн) ======================
st.set_page_config(page_title="ARGUS", layout="wide")
st.markdown("""
<style>
    .stApp {background: #f8fafc;}
    .main-header {font-size: 2.8rem; font-weight: 700; color: #0f172a;}
    .glass {background: white; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); padding: 24px;}
    .stButton>button {background: #0284c8; color: white; height: 52px; font-size: 1.1rem;}
    .stButton>button:hover {background: #0369a1;}
</style>
""", unsafe_allow_html=True)

# ====================== СОЗДАНИЕ / ЗАГРУЗКА МОДЕЛИ ======================
@st.cache_resource
def load_or_create_model():
    os.makedirs("models", exist_ok=True)
    model_path = "models/flood_model.pt"
    scaler_path = "models/scaler.pkl"

    # Если модель ещё не создана — создаём
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.info("Первый запуск: создаём модель и scaler...")
        
        np.random.seed(42)
        CITIES = ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]
        n = 5000
        
        df = pd.DataFrame({
            "temperature": np.random.normal(3.5, 12, n),
            "rain": np.random.exponential(8, n).clip(0, 55),
            "snow": np.random.exponential(15, n).clip(0, 45),
            "soil_moisture": np.random.beta(3, 2, n),
            "river_level": np.random.normal(52, 9, n).clip(35, 75),
            "city": np.random.choice(CITIES, n)
        })
        
        df["snow_melt"] = np.maximum(0, df["temperature"]) * df["snow"] * 0.12
        df["precip_3d"] = df["rain"] * 3
        df["precip_7d"] = df["rain"] * 7
        df["temp_rain_inter"] = df["temperature"] * df["rain"]
        df["soil_river"] = df["soil_moisture"] * df["river_level"]
        
        city_dummies = pd.get_dummies(df["city"], prefix="city")
        df = pd.concat([df.drop("city", axis=1), city_dummies], axis=1)
        
        # Жёстко заданный порядок колонок
        feature_columns = [
            "temperature", "rain", "snow", "soil_moisture", "river_level",
            "snow_melt", "precip_3d", "precip_7d", "temp_rain_inter", "soil_river"
        ] + [f"city_{c}" for c in CITIES]
        
        X = df[feature_columns]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        joblib.dump(scaler, scaler_path)
        
        model = FloodModel(input_dim=len(feature_columns))
        torch.save(model, model_path)
        
        st.success("Модель успешно создана!")
    
    # Загружаем готовые файлы
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    scaler = joblib.load(scaler_path)
    
    # Жёстко заданный список колонок (чтобы не было ошибки)
    feature_columns = [
        "temperature", "rain", "snow", "soil_moisture", "river_level",
        "snow_melt", "precip_3d", "precip_7d", "temp_rain_inter", "soil_river"
    ] + [f"city_{c}" for c in ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]]
    
    return model, scaler, feature_columns

model, scaler, feature_columns = load_or_create_model()

# ====================== ИНТЕРФЕЙС ======================
lang = st.sidebar.selectbox("Язык / Тіл / Language", ["Русский", "Қазақша", "English"])
lang_code = {"Русский": "ru", "Қазақша": "kk", "English": "en"}[lang]
t = translations[lang_code]

with st.sidebar:
    st.title("ARGUS")
    st.caption("Система прогнозирования паводков")
    city = st.selectbox(t["select_city"], ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"])

st.markdown(f"<h1 class='main-header'>{t['title']}</h1>", unsafe_allow_html=True)
st.subheader(f"{t['subtitle']} — {city}")

c1, c2, c3 = st.columns(3)
with c1: temp = st.slider(t["temperature"], -35.0, 40.0, 5.0, step=0.5)
with c2: rain = st.slider(t["rain"], 0.0, 60.0, 5.0, step=0.1)
with c3: snow = st.slider(t["snow"], 0.0, 45.0, 0.0, step=0.1)

c4, c5 = st.columns(2)
with c4: soil = st.slider(t["soil"], 0.0, 1.0, 0.3, step=0.01)
with c5: river = st.slider(t["river"], 35.0, 80.0, 50.0, step=0.1)

if st.button(t["calculate"], type="primary", use_container_width=True):
    input_data = {
        "temperature": temp, "rain": rain, "snow": snow,
        "soil_moisture": soil, "river_level": river,
        "snow_melt": max(0, temp) * snow * 0.12,
        "precip_3d": rain * 3, "precip_7d": rain * 7,
        "temp_rain_inter": temp * rain, "soil_river": soil * river
    }
    for c in ["Кокшетау", "Степногорск", "Щучинск", "Атбасар", "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы"]:
        input_data[f"city_{c}"] = 1 if c == city else 0
    
    df_input = pd.DataFrame([input_data])[feature_columns]
    
    X_scaled = scaler.transform(df_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logit = model(X_tensor)
        prob = torch.sigmoid(logit).item()

    risk = round(prob * 100, 1)
    if risk < 30: status, color = t["low"], "#22c55e"
    elif risk < 65: status, color = t["medium"], "#eab308"
    else: status, color = t["high"], "#ef4444"

    st.markdown(f"""
    <div class="glass" style="border-left: 8px solid {color};">
        <h2 style="color:{color}; margin:0;">{status}</h2>
        <h1 style="font-size:4.8rem; margin:15px 0; color:#0f172a;">{risk}%</h1>
        <p style="margin:0; font-size:1.1rem; color:#64748b;">{t["risk"]}</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(t["authors"])
st.caption(t["supervisor"])
st.caption(f"Год выпуска: {t['year']}")
