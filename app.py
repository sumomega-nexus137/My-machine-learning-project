"""
ARGUS — Flood Prediction System
Single-file version for Streamlit Cloud deployment.
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
INPUT_DIM       = len(FEATURE_COLUMNS)

MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "flood_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

T = {
    "ru": {
        "title": "ARGUS",
        "tagline": "Система прогнозирования паводков — Акмолинская область",
        "tab_predict": "Прогноз",
        "tab_about": "О проекте",
        "tab_info": "Паводки в регионе",
        "select_city": "Город / Район",
        "temperature": "Температура воздуха, °C",
        "rain": "Осадки (дождь), мм",
        "snow": "Снежный покров, см",
        "soil": "Влажность почвы (0–1)",
        "river": "Уровень реки, см",
        "calculate": "Рассчитать риск затопления",
        "risk": "Риск затопления",
        "low": "Низкий риск",
        "medium": "Средний риск",
        "high": "Высокий риск",
        "authors": "Разработано: Мирас Шамшидов и Бексултан Абдыхалык",
        "supervisor": "Научный руководитель: Ольга Шорникова Николаевна",
    },
    "kk": {
        "title": "ARGUS",
        "tagline": "Су тасқынын болжау жуйесi — Акмола облысы",
        "tab_predict": "Болжам",
        "tab_about": "Жоба туралы",
        "tab_info": "Аймактагы су тасқыны",
        "select_city": "Қала / Аудан",
        "temperature": "Ауа температурасы, °C",
        "rain": "Жаңбыр, мм",
        "snow": "Қар қабаты, см",
        "soil": "Топырақ ылғалдылығы (0–1)",
        "river": "Өзен деңгейі, см",
        "calculate": "Су басу қаупін есептеу",
        "risk": "Су басу қаупі",
        "low": "Төмен қауіп",
        "medium": "Орташа қауіп",
        "high": "Жоғары қауіп",
        "authors": "Əзiрлеген: Мирас Шамшидов жəне Бексұлтан Абдыхалық",
        "supervisor": "Ғылыми жетекшi: Ольга Шорникова Николаевна",
    },
    "en": {
        "title": "ARGUS",
        "tagline": "Flood Prediction System — Akmola Region, Kazakhstan",
        "tab_predict": "Prediction",
        "tab_about": "About",
        "tab_info": "Floods in the Region",
        "select_city": "City / District",
        "temperature": "Air temperature, °C",
        "rain": "Rainfall, mm",
        "snow": "Snow cover, cm",
        "soil": "Soil moisture (0–1)",
        "river": "River level, cm",
        "calculate": "Calculate Flood Risk",
        "risk": "Flood risk",
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
        "authors": "Developed by: Miras Shamshidov and Beksultan Abdykhalyk",
        "supervisor": "Supervisor: Olga Shornikova Nikolaevna",
    },
}

# ══════════════════════════════════════════════
#  MODEL
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
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════
def build_features(temp, rain, snow, soil, river, city) -> np.ndarray:
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

    n      = 8000
    temp   = np.random.normal(3.5, 12, n)
    rain   = np.random.exponential(8, n).clip(0, 55)
    snow   = np.random.exponential(15, n).clip(0, 45)
    soil   = np.random.beta(3, 2, n)
    river  = np.random.normal(52, 9, n).clip(35, 75)
    cities = np.random.choice(CITIES, n)

    X = np.vstack([
        build_features(t, r, s, so, rv, c)
        for t, r, s, so, rv, c in zip(temp, rain, snow, soil, river, cities)
    ])

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    risk_signal = (X_sc[:, 4] + X_sc[:, 5] + X_sc[:, 3]) / 3.0
    y   = torch.sigmoid(torch.tensor(risk_signal, dtype=torch.float32)).unsqueeze(1)
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
    torch.save(model.state_dict(), MODEL_PATH)
    return scaler, model

# ══════════════════════════════════════════════
#  LOAD MODEL — zero st.* calls inside cache
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            model  = FloodModel(INPUT_DIM)
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            model.eval()
            return scaler, model, "ok"
        except Exception:
            for p in (MODEL_PATH, SCALER_PATH):
                if os.path.exists(p):
                    os.remove(p)

    scaler, model = _train_and_save()
    return scaler, model, "trained"

# ══════════════════════════════════════════════
#  PAGE CONFIG & STYLES
# ══════════════════════════════════════════════
st.set_page_config(page_title="ARGUS", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #ffffff;
    color: #111111;
}

.stApp { background: #ffffff; }

section[data-testid="stSidebar"] {
    background: #f5f5f2;
    border-right: 1px solid #e2e2de;
}

.argus-logo {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 2.6rem;
    color: #111111;
    letter-spacing: -0.02em;
    line-height: 1;
}

.argus-tagline {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #999;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-top: 6px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #111;
    border-bottom: 2px solid #111;
    padding-bottom: 10px;
    margin-bottom: 24px;
    margin-top: 8px;
}

.card {
    background: #f5f5f2;
    border-radius: 3px;
    padding: 26px 30px;
    margin-bottom: 14px;
    border-left: 3px solid #111;
}

.card-accent {
    background: #f5f5f2;
    border-radius: 3px;
    padding: 26px 30px;
    margin-bottom: 14px;
    border-left: 3px solid #aaa;
}

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #111;
    margin-bottom: 10px;
}

.card-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 300;
    color: #333;
    line-height: 1.75;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}

.stat-box {
    background: #111;
    color: #fff;
    border-radius: 3px;
    padding: 24px 20px;
    text-align: center;
}

.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 8px;
}

.stat-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 400;
    color: #bbb;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.result-block {
    padding: 32px 36px;
    border-radius: 3px;
    margin-top: 24px;
    background: #f5f5f2;
    border: 1px solid #e2e2de;
}

.result-risk-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 8px;
}

.result-risk-status {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 4px;
}

.result-risk-number {
    font-family: 'Playfair Display', serif;
    font-size: 5.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.04em;
}

.footer-line {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 300;
    color: #bbb;
    margin-top: 3px;
}

.tech-pill {
    display: inline-block;
    background: #111;
    color: #fff;
    font-family: 'IBM Plex Sans', monospace;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 2px;
    margin: 3px 4px 3px 0;
    letter-spacing: 0.05em;
}

.divider-line {
    border: none;
    border-top: 1px solid #e2e2de;
    margin: 20px 0;
}

div[data-testid="stSlider"] label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #444;
}

.stButton > button {
    width: 100%;
    background: #111111;
    color: #ffffff;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 2px;
    height: 52px;
}

.stButton > button:hover { background: #333; }

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #111;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    padding: 12px 28px;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  BOOT
# ══════════════════════════════════════════════
with st.spinner("Загрузка модели..."):
    scaler, model, boot_status = _load_model()

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="argus-logo">ARGUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="argus-tagline">Flood Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    lang = st.selectbox("Язык / Тiл / Language", ["Русский", "Қазақша", "English"])
    t = T[{"Русский": "ru", "Қазақша": "kk", "English": "en"}[lang]]

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    city = st.selectbox(t["select_city"], CITIES)
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    st.markdown('<p class="footer-line">v2.3 · Akmola Region · 2026</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="footer-line">{t["authors"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="footer-line">{t["supervisor"]}</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
st.markdown(f'<div class="argus-logo" style="font-size:3.5rem;">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="argus-tagline" style="margin-bottom:32px;">{t["tagline"]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([t["tab_predict"], t["tab_about"], t["tab_info"]])

# ─────────────────────────────────────────────
#  TAB 1 — PREDICTION
# ─────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Прогноз риска затопления</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: temp  = st.slider(t["temperature"], -35.0, 40.0,  5.0, step=0.5)
    with c2: rain  = st.slider(t["rain"],          0.0, 60.0,  5.0, step=0.1)
    with c3: snow  = st.slider(t["snow"],          0.0, 45.0,  0.0, step=0.1)

    c4, c5 = st.columns(2)
    with c4: soil  = st.slider(t["soil"],   0.0,  1.0,  0.3, step=0.01)
    with c5: river = st.slider(t["river"], 35.0, 80.0, 50.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(t["calculate"], use_container_width=True):
        x_raw    = build_features(temp, rain, snow, soil, river, city)
        x_scaled = scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()

        risk = round(prob * 100, 1)
        if risk < 30:   status, color = t["low"],    "#16a34a"
        elif risk < 65: status, color = t["medium"], "#d97706"
        else:           status, color = t["high"],   "#dc2626"

        st.markdown(f"""
        <div class="result-block" style="border-left: 5px solid {color};">
            <div class="result-risk-label">{city} — {t["risk"]}</div>
            <div class="result-risk-status" style="color:{color};">{status}</div>
            <div class="result-risk-number" style="color:{color};">{risk}<span style="font-size:2rem;color:#ccc;">%</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Снеготаяние", f"{round(max(0.0, temp) * snow * 0.12, 1)} мм/сут")
        with col_b:
            st.metric("Осадки за 7 дней", f"{round(rain * 7, 1)} мм")
        with col_c:
            st.metric("Индекс насыщения", f"{round(soil * river, 1)}")

# ─────────────────────────────────────────────
#  TAB 2 — ABOUT PROJECT
# ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">О проекте ARGUS</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Проблема</div>
        <div class="card-body">
            Весенние паводки — ежегодное стихийное бедствие для Акмолинской области. Традиционные методы
            мониторинга опираются на ручной сбор данных и экспертный опыт, что замедляет реакцию
            служб реагирования и не позволяет заблаговременно информировать население.
            Каждый дополнительный день предупреждения снижает экономический ущерб и спасает жизни.
            ARGUS создан для того, чтобы предоставить этот критически важный временной буфер.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Что делает система</div>
        <div class="card-body">
            ARGUS анализирует совокупность гидрометеорологических параметров — температуру воздуха,
            уровень снежного покрова, интенсивность осадков, влажность почвы и уровень воды в реке —
            и на основе нейронной сети рассчитывает вероятность затопления для конкретного
            населённого пункта. Результат выдаётся мгновенно и не требует специальных технических знаний.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Для кого разработан ARGUS</div>
        <div class="card-body">
            Система ориентирована на три категории пользователей: сотрудников районных акиматов
            и служб гражданской защиты, которым необходим оперативный инструмент оценки обстановки;
            работников метеорологических станций, дополняющих традиционные наблюдения цифровым анализом;
            а также учащихся и исследователей, изучающих методы машинного обучения в приложении
            к задачам гидрологии и экологии.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.4rem;">Технологический стек</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:24px;">
        <span class="tech-pill">PyTorch 2.x</span>
        <span class="tech-pill">scikit-learn</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">NumPy</span>
        <span class="tech-pill">joblib</span>
        <span class="tech-pill">Python 3.12+</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Архитектура модели</div>
        <div class="card-body">
            В основе ARGUS — четырёхслойный многослойный перцептрон (MLP) с нормализацией по батчу
            и регуляризацией Dropout. Входной вектор содержит 20 признаков: 10 метеорологических
            и гидрологических параметров (включая производные — снеготаяние, накопленные осадки,
            индекс насыщения почвы) и 10 бинарных переменных для кодирования населённого пункта.
            Обучение выполняется с оптимизатором AdamW и расписанием скорости обучения
            Cosine Annealing за 60 эпох. На выходе — вероятность затопления от 0 до 100%.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.4rem;">Авторы</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Мирас Шамшидов</div>
            <div class="card-body">
                Разработка модели, инженерия признаков, архитектура приложения,
                деплой на Streamlit Cloud.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card">
            <div class="card-title">Бексултан Абдыхалык</div>
            <div class="card-body">
                Исследование предметной области, сбор данных по региону,
                тестирование и верификация результатов.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-accent">
        <div class="card-title">Научный руководитель</div>
        <div class="card-body">
            Ольга Шорникова Николаевна — методическое руководство, постановка задачи, научная экспертиза.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TAB 3 — FLOODS IN REGION
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Паводки в Акмолинской области</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-box">
            <div class="stat-number">2024</div>
            <div class="stat-label">Крупнейший паводок за 80 лет</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">400+</div>
            <div class="stat-label">Населённых пунктов пострадало</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">март — май</div>
            <div class="stat-label">Пиковый сезон паводков</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Географические особенности региона</div>
        <div class="card-body">
            Акмолинская область расположена в центре Казахстана и занимает площадь около 146 000 кв. км.
            Регион пересекают реки Ишим, Нура, Чаглинка и их притоки. Плоский рельеф степной зоны
            и преобладание глинистых почв с низкой водопроницаемостью создают условия, при которых
            даже умеренное снеготаяние приводит к разливу рек на десятки километров от их русел.
            Область включает Кокшетау, Степногорск, Щучинск и ещё более 20 районных центров,
            часть которых расположена в поймах или вблизи водоёмов.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Причины паводков</div>
        <div class="card-body">
            Основная причина весенних паводков — интенсивное таяние снега, накопленного за зиму.
            При среднесуточных температурах выше +5°C скорость снеготаяния резко возрастает,
            и объём воды превышает пропускную способность речных русел. Усугубляющие факторы:
            промёрзший грунт, не способный впитывать воду; дождевые осадки, совпадающие
            по времени со снеготаянием; высокий уровень грунтовых вод после многоснежной зимы;
            ледовые заторы, резко поднимающие уровень воды выше по течению.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Паводок 2024 года</div>
        <div class="card-body">
            Весной 2024 года Акмолинская область пережила наиболее масштабное наводнение за последние
            восемь десятилетий. Уровень реки Ишим превысил критические отметки на несколько метров.
            В регионе было подтоплено более 15 000 домов, эвакуировано свыше 100 000 человек.
            Ущерб сельскохозяйственному сектору составил несколько миллиардов тенге.
            Это событие наглядно показало критическую необходимость в системах раннего предупреждения,
            способных за 3–7 суток выдавать обоснованную оценку риска по конкретному населённому пункту.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Социально-экономические последствия</div>
        <div class="card-body">
            Паводки наносят комплексный ущерб: разрушение жилого фонда и инфраструктуры,
            гибель скота и потеря урожая, загрязнение питьевой воды, вспышки инфекционных заболеваний.
            Наиболее уязвимы сельские районы, где удалённость от служб реагирования затрудняет эвакуацию.
            По оценкам специалистов, каждый дополнительный день предупреждения снижает прямой
            экономический ущерб на 15–20%.
        </div>
    </div>

    <div class="card">
        <div class="card-title">Существующие меры и их ограничения</div>
        <div class="card-body">
            Мониторинг ведётся сетью гидрометеорологических станций Казгидромета,
            а оповещение осуществляется через акиматы и МЧС. Однако существующая система
            имеет существенные ограничения: данные со станций обновляются с задержкой,
            прогнозы носят региональный характер и не учитывают локальную специфику конкретного
            населённого пункта, а пороговые значения опасности заданы статически.
            ARGUS дополняет эту систему, предоставляя детализированный прогноз на уровне
            отдельного города или района .
        </div>
    </div>
    """, unsafe_allow_html=True)
