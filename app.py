"""
ARGUS — Flood Prediction System
Single-file version for reliable Streamlit Cloud deployment.
"""

import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# ══════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════
CITIES = [
    "Кокшетау", "Степногорск", "Щучинск", "Атбасар",
    "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы",
]

BASE_FEATURES = [
    "temperature", "rain", "snow", "soil_moisture", "river_level",
    "snow_melt", "precip_3d", "precip_7d", "temp_rain_inter", "soil_river",
]
CITY_FEATURES   = [f"city_{c}" for c in CITIES]
FEATURE_COLUMNS = BASE_FEATURES + CITY_FEATURES
INPUT_DIM       = len(FEATURE_COLUMNS)   # 20

MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "flood_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

T = {
    "ru": {
        "title": "ARGUS • Прогноз паводков",
        "subtitle": "Акмолинская область",
        "select_city": "Город / Район",
        "temperature": "Температура воздуха, °C",
        "rain": "Осадки (дождь), мм",
        "snow": "Снежный покров, см",
        "soil": "Влажность почвы (0–1)",
        "river": "Уровень реки, см",
        "calculate": "Рассчитать риск затопления →",
        "risk": "Риск затопления",
        "low": "Низкий риск",
        "medium": "Средний риск",
        "high": "Высокий риск",
        "authors": "Разработано: Мирас Шамшидов & Бексултан Абдыхалык",
        "supervisor": "Научный руководитель: Ольга Шорникова Николаевна",
    },
    "kk": {
        "title": "ARGUS • Су тасқыны болжамы",
        "subtitle": "Ақмола облысы",
        "select_city": "Қала / Аудан",
        "temperature": "Ауа температурасы, °C",
        "rain": "Жаңбыр, мм",
        "snow": "Қар қабаты, см",
        "soil": "Топырақ ылғалдылығы (0–1)",
        "river": "Өзен деңгейі, см",
        "calculate": "Су басу қаупін есептеу →",
        "risk": "Су басу қаупі",
        "low": "Төмен қауіп",
        "medium": "Орташа қауіп",
        "high": "Жоғары қауіп",
        "authors": "Әзірлеген: Мирас Шамшидов & Бексұлтан Абдыхалық",
        "supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна",
    },
    "en": {
        "title": "ARGUS • Flood Prediction",
        "subtitle": "Akmola Region, Kazakhstan",
        "select_city": "City / District",
        "temperature": "Air temperature, °C",
        "rain": "Rainfall, mm",
        "snow": "Snow cover, cm",
        "soil": "Soil moisture (0–1)",
        "river": "River level, cm",
        "calculate": "Calculate Flood Risk →",
        "risk": "Flood risk",
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
        "authors": "Developed by: Miras Shamshidov & Beksultan Abdykhalyk",
        "supervisor": "Supervisor: Olga Shornikova Nikolaevna",
    },
}

# ══════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ══════════════════════════════════════════════
class FloodModel(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ══════════════════════════════════════════════
#  FEATURE ENGINEERING  (pure numpy — no DataFrame)
# ══════════════════════════════════════════════
def build_features(temp, rain, snow, soil, river, city) -> np.ndarray:
    """Returns shape (1, INPUT_DIM) float32 array. No pandas → no column-name errors."""
    vals = {
        "temperature":     float(temp),
        "rain":            float(rain),
        "snow":            float(snow),
        "soil_moisture":   float(soil),
        "river_level":     float(river),
        "snow_melt":       max(0.0, float(temp)) * float(snow) * 0.12,
        "precip_3d":       float(rain) * 3.0,
        "precip_7d":       float(rain) * 7.0,
        "temp_rain_inter": float(temp) * float(rain),
        "soil_river":      float(soil) * float(river),
        f"city_{city}":    1.0,
    }
    row = np.array([vals.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32)
    return row.reshape(1, -1)

# ══════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════
def _train_and_save():
    from sklearn.preprocessing import StandardScaler

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    n = 8000
    temp  = np.random.normal(3.5, 12, n)
    rain  = np.random.exponential(8, n).clip(0, 55)
    snow  = np.random.exponential(15, n).clip(0, 45)
    soil  = np.random.beta(3, 2, n)
    river = np.random.normal(52, 9, n).clip(35, 75)
    cities = np.random.choice(CITIES, n)

    X = np.vstack([
        build_features(t, r, s, so, rv, c)
        for t, r, s, so, rv, c in zip(temp, rain, snow, soil, river, cities)
    ])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    # Synthetic risk signal
    risk_signal = (X_sc[:, 4] + X_sc[:, 5] + X_sc[:, 3]) / 3.0
    y = torch.sigmoid(torch.tensor(risk_signal, dtype=torch.float32)).unsqueeze(1)
    X_t = torch.tensor(X_sc, dtype=torch.float32)

    model     = FloodModel(INPUT_DIM)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(60):
        optimizer.zero_grad()
        criterion(model(X_t), y).backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    # Save ONLY state_dict (safe across PyTorch versions)
    torch.save(model.state_dict(), MODEL_PATH)
    return scaler, model

# ══════════════════════════════════════════════
#  LOAD OR CREATE
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Try loading existing files
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            model  = FloodModel(INPUT_DIM)
            # weights_only=True: reads only tensors, never executes arbitrary code
            state  = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return scaler, model
        except Exception as e:
            # Incompatible / corrupt files → wipe and retrain
            st.warning(f"Old model files detected, retraining… ({type(e).__name__})")
            for p in (MODEL_PATH, SCALER_PATH):
                if os.path.exists(p):
                    os.remove(p)

    # Train from scratch
    with st.spinner("⚙️ Training model (first run only, ~10s)…"):
        scaler, model = _train_and_save()
    st.toast("✅ Model ready!", icon="🌊")
    return scaler, model

# ══════════════════════════════════════════════
#  PAGE CONFIG & STYLES
# ══════════════════════════════════════════════
st.set_page_config(page_title="ARGUS", page_icon="🌊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0f1e; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #0d1424;
    border-right: 1px solid #1e2d4a;
}

.main-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.sub-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}
.result-card {
    background: #111827;
    border-radius: 20px;
    padding: 36px 40px;
    margin-top: 24px;
    border: 1px solid #1e2d4a;
}
.risk-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.risk-status { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.5rem; }
.risk-percent {
    font-family: 'Space Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
}
.footer-note {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #334155;
    margin-top: 2px;
}
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
div[data-testid="stSlider"] label {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    height: 56px;
}
.stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════
scaler, model = get_model()

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">ARGUS</div>', unsafe_allow_html=True)
    st.caption("FLOOD PREDICTION SYSTEM")
    st.divider()

    lang = st.selectbox("🌐 Language / Тіл / Язык", ["Русский", "Қазақша", "English"])
    t = T[{"Русский": "ru", "Қазақша": "kk", "English": "en"}[lang]]

    st.divider()
    city = st.selectbox(t["select_city"], CITIES)
    st.divider()

    st.markdown('<p class="footer-note">v2.1 · Akmola Region · 2026</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════
st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t["subtitle"]} — {city}</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: temp  = st.slider(t["temperature"], -35.0, 40.0, 5.0,  step=0.5)
with c2: rain  = st.slider(t["rain"],          0.0, 60.0, 5.0,  step=0.1)
with c3: snow  = st.slider(t["snow"],          0.0, 45.0, 0.0,  step=0.1)

c4, c5 = st.columns(2)
with c4: soil  = st.slider(t["soil"],   0.0, 1.0,  0.3, step=0.01)
with c5: river = st.slider(t["river"], 35.0, 80.0, 50.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button(t["calculate"], use_container_width=True):
    x_raw    = build_features(temp, rain, snow, soil, river, city)
    x_scaled = scaler.transform(x_raw)                          # numpy in → no name errors
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(x_tensor)).item()

    risk = round(prob * 100, 1)
    if risk < 30:   status, color, icon = t["low"],    "#22c55e", "🟢"
    elif risk < 65: status, color, icon = t["medium"], "#f59e0b", "🟡"
    else:           status, color, icon = t["high"],   "#ef4444", "🔴"

    st.markdown(f"""
    <div class="result-card" style="border-left: 6px solid {color};">
        <div class="risk-label">{icon} {t["risk"]}</div>
        <div class="risk-status" style="color:{color};">{status}</div>
        <div class="risk-percent" style="color:{color};">{risk}<span style="font-size:1.8rem;opacity:0.5;">%</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f'<p class="footer-note">{t["authors"]}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="footer-note">{t["supervisor"]}</p>', unsafe_allow_html=True)
