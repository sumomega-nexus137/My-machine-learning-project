import os
import pathlib

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG — must be the very first Streamlit call
# =========================================================
st.set_page_config(
    page_title="ARGUS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CONSTANTS
# =========================================================
CITIES = [
    "Кокшетау", "Степногорск", "Щучинск", "Атбасар",
    "Акколь", "Макинск", "Есиль", "Ерейментау", "Степняк", "Қосшы",
]

BASE_FEATURES = [
    "temperature", "rain", "snow", "soil_moisture", "river_level",
    "snow_melt", "precip_3d", "precip_7d", "temp_rain_inter", "soil_river",
]
CITY_FEATURES = [f"city_{c}" for c in CITIES]
FEATURE_COLUMNS = BASE_FEATURES + CITY_FEATURES
INPUT_DIM = len(FEATURE_COLUMNS)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "argus_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "argus_scaler.npz")

CITY_BIAS = {
    "Кокшетау": 0.10, "Степногорск": 0.08, "Щучинск": 0.12,
    "Атбасар": 0.28,  "Акколь": 0.16,      "Макинск": 0.18,
    "Есиль": 0.14,    "Ерейментау": 0.09,  "Степняк": 0.11,
    "Қосшы": 0.07,
}

LANG_MAP = {"Русский": "ru", "Қазақша": "kk", "English": "en"}

# =========================================================
# TRANSLATIONS
# =========================================================
T = {
    "ru": {
        "app_title": "ARGUS",
        "app_tagline": "Официальная система прогнозирования паводков для Акмолинской области",
        "app_subtitle": "Инструмент поддержки решений для оценки риска и планирования эвакуации",
        "sidebar_language": "Язык",
        "sidebar_city": "Город / район",
        "sidebar_note": "Проект разработан в 2026 году.",
        "sidebar_authors": "Авторы: Шамшидов Мирас, Бексултан Абдыхалык",
        "sidebar_supervisor": "Научный руководитель: Ольга Шорникова Николаевна",
        "tab_predict": "Прогноз",
        "tab_region": "Регион",
        "tab_about": "О проекте",
        "tab_guide": "Инструкция",
        "section_predict": "Прогноз риска затопления",
        "section_inputs": "Входные параметры",
        "section_results": "Результат",
        "section_charts": "Графики факторов риска",
        "temp": "Температура воздуха, °C",
        "rain": "Осадки за сутки, мм",
        "snow": "Снежный покров, см",
        "soil": "Влажность почвы, %",
        "river": "Уровень реки, см",
        "calculate": "Рассчитать риск",
        "risk": "Риск затопления",
        "low": "Низкий риск",
        "medium": "Средний риск",
        "high": "Высокий риск",
        "metric_snowmelt": "Снеготаяние",
        "metric_precip7": "Осадки за 7 суток",
        "metric_saturation": "Индекс насыщения",
        "metric_city": "Населённый пункт",
        "region_title": "Акмолинская область",
        "region_intro_title": "Почему регион уязвим",
        "region_intro_body": "Для Акмолинской области особенно характерны весенние паводки, связанные с быстрым снеготаянием, дождевыми осадками и высоким уровнем поверхностного стока. Плоский рельеф, промёрзший грунт и низкая водопроницаемость почв усиливают риск резкого подъёма воды.",
        "region_history_title": "Контекст последних лет",
        "region_history_body": "В 2024 году паводки в Казахстане стали одними из самых масштабных за последние десятилетия. Для региона также важен опыт 2017 года, когда разливы и разрушения показали уязвимость отдельных населённых пунктов и инженерной инфраструктуры.",
        "region_operational_title": "Оперативное значение",
        "region_operational_body": "Этот интерфейс создан как инструмент ранней оценки риска для служб реагирования, местных органов управления и жителей. Он помогает заранее определить зоны повышенного риска и поддержать решение по эвакуации.",
        "gallery_title": "Фотоматериалы и карта региона",
        "photo_1": "Карта рельефа Акмолинской области",
        "photo_2": "Ситуация с паводками в Казахстане, 2024",
        "photo_3": "Пример разрушений после наводнения",
        "about_title": "О проекте ARGUS",
        "about_goal_title": "Цель проекта",
        "about_goal_body": "ARGUS разработан для оценки вероятности затопления по гидрометеорологическим параметрам и для поддержки решений в вопросах предупреждения и эвакуации.",
        "about_model_title": "Что делает система",
        "about_model_body": "Система анализирует температуру, осадки, снежный покров, влажность почвы и уровень воды в реке, после чего выдаёт оценку риска в процентах.",
        "about_stack_title": "Технологический стек",
        "about_stack_body": "PyTorch, Streamlit, NumPy и Matplotlib.",
        "about_honesty_title": "Принцип честности",
        "about_honesty_body": "ARGUS является инструментом оценки риска и не заменяет официальные гидрометеорологические предупреждения и работу экстренных служб.",
        "about_authors_title": "Авторы и руководство",
        "author_1_name": "Шамшидов Мирас",
        "author_1_body": "Разработка модели, интерфейса и логики расчёта.",
        "author_2_name": "Бексултан Абдыхалык",
        "author_2_body": "Анализ предметной области, проверка структуры и презентация проекта.",
        "supervisor_name": "Ольга Шорникова Николаевна",
        "supervisor_body": "Научный руководитель проекта.",
        "guide_title": "Инструкция по использованию",
        "guide_step_1": "1. Выберите город или район из списка.",
        "guide_step_2": "2. Введите параметры: температуру, осадки, снежный покров, влажность почвы и уровень реки.",
        "guide_step_3": "3. Нажмите кнопку расчёта.",
        "guide_step_4": "4. Ознакомьтесь с процентом риска, интерпретацией и графиками.",
        "guide_for_title": "Для кого создано",
        "guide_for_body": "Для акиматов, служб гражданской защиты, экстренных подразделений, исследователей, преподавателей и жителей региона.",
        "guide_benefit_title": "Как помогает",
        "guide_benefit_body": "Система помогает заранее видеть рост риска, поддерживать решение об эвакуации и направлять внимание на наиболее опасные параметры.",
        "guide_note": "Результат следует использовать как оценку риска на основе введённых данных.",
        "risk_hint": "Чем выше процент, тем выше вероятность опасной паводковой ситуации.",
        "input_hint": "Все значения заданы в реалистичных для региона единицах.",
    },
    "kk": { ... },  # ← keep your full kk and en translations here (I shortened for message size)
    "en": { ... },
}

# =========================================================
# MODEL / SCALER (unchanged)
# =========================================================
class StandardScalerLite:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) / self.scale_

    def save(self, path: str):
        np.savez(path, mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        obj = cls()
        obj.mean_ = data["mean"]
        obj.scale_ = data["scale"]
        return obj


class FloodModel(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_features(temp, rain, snow, soil_percent, river_cm, city) -> np.ndarray:
    soil = float(soil_percent) / 100.0
    snow_melt = max(0.0, float(temp)) * float(snow) * 0.12
    vals = {
        "temperature": float(temp),
        "rain": float(rain),
        "snow": float(snow),
        "soil_moisture": soil,
        "river_level": float(river_cm),
        "snow_melt": snow_melt,
        "precip_3d": float(rain) * 3.0,
        "precip_7d": float(rain) * 7.0,
        "temp_rain_inter": float(temp) * float(rain),
        "soil_river": soil * float(river_cm),
        f"city_{city}": 1.0,
    }
    row = np.array([vals.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32)
    return row.reshape(1, -1)


def _generate_training_data(n: int = 12000):
    # ... (your original _generate_training_data function - keep it unchanged)
    pass  # ← put your full function here


@st.cache_resource(show_spinner=False)
def _load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler = StandardScalerLite.load(SCALER_PATH)
            mdl = FloodModel(INPUT_DIM)
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            mdl.load_state_dict(state)
            mdl.eval()
            return scaler, mdl
        except Exception:
            for p in (MODEL_PATH, SCALER_PATH):
                try:
                    os.remove(p)
                except Exception:
                    pass

    scaler, mdl = _generate_training_data()
    torch.save(mdl.state_dict(), MODEL_PATH)
    scaler.save(SCALER_PATH)
    return scaler, mdl


# =========================================================
# CSS (exactly as original)
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
    background: #ffffff; color: #111827;
}
.stApp { background: #ffffff; }
section[data-testid="stSidebar"] {
    background: #f7f8fa; border-right: 1px solid #e5e7eb;
}
.argus-brand {
    font-size: 2.6rem; font-weight: 800;
    letter-spacing: -0.04em; color: #111827; line-height: 1;
}
.argus-tagline {
    margin-top: 0.25rem; font-size: 0.82rem; font-weight: 600;
    color: #374151; letter-spacing: 0.08em; text-transform: uppercase;
}
.argus-subtitle { margin-top: 0.4rem; color: #4b5563; font-size: 0.95rem; line-height: 1.6; }
.section-title {
    margin: 1rem 0 1rem; padding-bottom: 0.55rem;
    border-bottom: 1px solid #d1d5db;
    font-size: 1.4rem; font-weight: 700; color: #111827;
}
.card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 18px; padding: 1.1rem 1.15rem; margin-bottom: 0.9rem;
    box-shadow: 0 6px 24px rgba(17,24,39,0.04);
}
.card-title { font-size: 1rem; font-weight: 700; color: #111827; margin-bottom: 0.45rem; }
.card-body { color: #374151; font-size: 0.94rem; line-height: 1.7; }
.result-panel {
    background: #111827; color: #fff;
    border-radius: 20px; padding: 1.2rem 1.25rem; margin-top: 1rem;
}
.result-label {
    font-size: 0.76rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: #cbd5e1; margin-bottom: 0.4rem;
}
.result-status { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.2rem; }
.result-number { font-size: 4.4rem; font-weight: 800; line-height: 1; letter-spacing: -0.06em; }
.small-note { color: #6b7280; font-size: 0.82rem; line-height: 1.5; }
.tech-pill {
    display: inline-block; background: #111827; color: #fff;
    padding: 0.3rem 0.75rem; border-radius: 999px;
    margin: 0.2rem 0.2rem 0 0; font-size: 0.75rem; letter-spacing: 0.04em;
}
.img-placeholder {
    background: #f3f4f6; border: 2px dashed #d1d5db;
    border-radius: 12px; height: 200px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #9ca3af; font-size: 0.85rem; text-align: center; padding: 1rem;
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem; background: transparent; border-bottom: 1px solid #d1d5db;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280;
    padding: 0.8rem 1rem; background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #111827 !important;
    background: transparent !important;
}
div[data-testid="stSlider"] label { font-size: 0.9rem; font-weight: 600; color: #374151; }
.stButton > button {
    width: 100%; height: 3.05rem; border: none;
    border-radius: 14px; background: #111827; color: #fff;
    font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
}
.stButton > button:hover { background: #1f2937; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS - ONLY THIS PART WAS CHANGED
# =========================================================
def tr(key: str) -> str:
    lang = st.session_state.get("lang", "ru")
    return T[lang].get(key, key)


def render_card(title: str, body: str):
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div>'
        f'<div class="card-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def risk_level(prob_pct: float):
    if prob_pct < 30:
        return tr("low"), "#16a34a"
    if prob_pct < 65:
        return tr("medium"), "#d97706"
    return tr("high"), "#dc2626"


# FIXED IMAGE FUNCTION
def show_image(filename: str, caption: str = ""):
    """Fixed version - looks for images in 'images/' folder"""
    image_path = os.path.join("images", filename)
    p = pathlib.Path(image_path)
    
    if p.exists():
        st.image(str(p), use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        filename_only = p.name
        st.markdown(
            f'<div class="img-placeholder">'
            f'<span style="font-size:2rem;">📷</span>'
            f'<span>{caption}</span>'
            f'<span style="color:#d1d5db;font-size:0.75rem;">Add <b>{filename_only}</b> to <b>images/</b> folder</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if caption:
            st.caption(caption)


# =========================================================
# BOOT
# =========================================================
os.makedirs("images", exist_ok=True)   # ← added this so folder is created automatically

with st.spinner("Loading ARGUS model…"):
    scaler, model = _load_model()

# =========================================================
# SIDEBAR (original)
# =========================================================
with st.sidebar:
    st.markdown("<div class='argus-brand'>ARGUS</div>", unsafe_allow_html=True)
    st.markdown("<div class='argus-tagline'>Flood Risk System</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'>", unsafe_allow_html=True)

    lang_label = st.selectbox(
        "Язык / Тіл / Language",
        ["Русский", "Қазақша", "English"],
        index=0,
        key="lang_selector",
    )
    st.session_state["lang"] = LANG_MAP[lang_label]

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'>", unsafe_allow_html=True)
    city = st.selectbox(tr("sidebar_city"), CITIES, key="city_selector")

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='argus-subtitle'>{tr('app_subtitle')}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_note')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_authors')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_supervisor')}</div>", unsafe_allow_html=True)

# =========================================================
# HEADER (original)
# =========================================================
st.markdown(f"<div class='argus-brand' style='font-size:3.3rem;'>{tr('app_title')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='argus-tagline' style='margin-top:0.4rem;'>{tr('app_tagline')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='argus-subtitle'>{tr('app_subtitle')}</div>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    tr("tab_predict"), tr("tab_region"), tr("tab_about"), tr("tab_guide"),
])

# =========================================================
# TAB 1 — PREDICTION (original)
# =========================================================
with tab1:
    # ... your original prediction tab code (unchanged) ...
    pass   # ← keep everything the same here

# =========================================================
# TAB 2 — REGION  ← ONLY THIS PART WAS FIXED
# =========================================================
with tab2:
    st.markdown(f"<div class='section-title'>{tr('region_title')}</div>", unsafe_allow_html=True)

    render_card(tr("region_intro_title"), tr("region_intro_body"))
    render_card(tr("region_history_title"), tr("region_history_body"))
    render_card(tr("region_operational_title"), tr("region_operational_body"))

    st.markdown(f"<div class='section-title' style='margin-top:1.2rem;'>{tr('gallery_title')}</div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        show_image("map_akmola.png", tr("photo_1"))
    with col_b:
        show_image("flood_2024.jpg", tr("photo_2"))
    with col_c:
        show_image("flood_damage.jpg", tr("photo_3"))

# =========================================================
# TAB 3 & TAB 4 (original - unchanged)
# =========================================================
with tab3:
    # your original About tab
    pass

with tab4:
    # your original Guide tab
    pass
