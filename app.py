"""
ARGUS — Flood Risk Prediction System
Single-file Streamlit app.

Requirements:
    streamlit
    torch
    numpy
    matplotlib
    Pillow

Put photos in images/ folder:
    images/map_akmola.png
    images/flood_2024.jpg
    images/flood_damage.jpg
"""

import os
import pathlib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# =========================================================
# PAGE CONFIG
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
        "sidebar_city": "Город / район",
        "sidebar_note": "Проект разработан в 2026 году.",
        "sidebar_authors": "Авторы: Шамшидов Мирас, Бексултан Абдыхалык",
        "sidebar_supervisor": "Научный руководитель: Ольга Шорникова Николаевна",
        "tab_predict": "Прогноз",
        "tab_region": "Регион",
        "tab_about": "О проекте",
        "tab_guide": "Инструкция",
        "section_predict": "Прогноз риска затопления",
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
        "region_title": "Акмолинская область",
        "region_intro_title": "Почему регион уязвим",
        "region_intro_body": "Для Акмолинской области особенно характерны весенние паводки, связанные с быстрым снеготаянием, дождевыми осадками и высоким уровнем поверхностного стока. Плоский рельеф, промёрзший грунт и низкая водопроницаемость почв усиливают риск резкого подъёма воды.",
        "region_history_title": "Контекст последних лет",
        "region_history_body": "В 2024 году паводки в Казахстане стали одними из самых масштабных за последние десятилетия. Для региона также важен опыт 2017 года, когда разливы и разрушения показали уязвимость отдельных населённых пунктов и инженерной инфраструктуры.",
        "region_operational_title": "Оперативное значение",
        "region_operational_body": "Этот интерфейс создан как инструмент ранней оценки риска для служб реагирования, местных органов управления и жителей. Он помогает заранее определить зоны повышенного риска и поддержать решение по эвакуации.",
        "gallery_title": "Фотоматериалы разрушений от паводков в Акмолинской области",
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
        "bolashaq_label": "Проект представлен от имени:",
        "bolashaq_city": "г. Кокшетау",
    },
    "kk": {
        "app_title": "ARGUS",
        "app_tagline": "Ақмола облысы үшін су тасқынын болжаудың ресми жүйесі",
        "app_subtitle": "Қауіпті бағалау және эвакуацияны жоспарлау үшін шешім қабылдауды қолдайтын құрал",
        "sidebar_city": "Қала / аудан",
        "sidebar_note": "Жоба 2026 жылы әзірленді.",
        "sidebar_authors": "Авторлар: Шамшидов Мирас, Бексултан Абдыхалык",
        "sidebar_supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна",
        "tab_predict": "Болжам",
        "tab_region": "Аймақ",
        "tab_about": "Жоба туралы",
        "tab_guide": "Нұсқаулық",
        "section_predict": "Су басу қаупін болжау",
        "section_results": "Нәтиже",
        "section_charts": "Қауіп факторларының графиктері",
        "temp": "Ауа температурасы, °C",
        "rain": "Тәуліктік жауын-шашын, мм",
        "snow": "Қар жамылғысы, см",
        "soil": "Топырақ ылғалдылығы, %",
        "river": "Өзен деңгейі, см",
        "calculate": "Қауіпті есептеу",
        "risk": "Су басу қаупі",
        "low": "Төмен қауіп",
        "medium": "Орташа қауіп",
        "high": "Жоғары қауіп",
        "metric_snowmelt": "Қар еруі",
        "metric_precip7": "7 күндік жауын-шашын",
        "metric_saturation": "Қанығу индексі",
        "region_title": "Ақмола облысы",
        "region_intro_title": "Неліктен аймақ осал",
        "region_intro_body": "Ақмола облысы үшін көктемгі су тасқыны ерекше тән. Ол қардың тез еруімен, жауын-шашынмен және жер бетімен ағып келетін судың көбеюімен байланысты. Жазық рельеф, тоңазытылған топырақ және төмен сіңіргіштік тәуекелді күшейтеді.",
        "region_history_title": "Соңғы жылдардағы жағдай",
        "region_history_body": "2024 жылы Қазақстандағы су тасқыны соңғы онжылдықтардағы ең ауқымдылардың бірі болды. Сондай-ақ 2017 жылғы жағдай өңірдің жекелеген елді мекендері мен инженерлік инфрақұрылымының осалдығын көрсетті.",
        "region_operational_title": "Жедел маңызы",
        "region_operational_body": "Бұл интерфейс төтенше жағдай қызметтері, жергілікті билік және тұрғындар үшін алдын ала тәуекелді бағалау құралы ретінде жасалған. Ол қауіпті аймақтарды ерте анықтауға және эвакуация туралы шешімді қолдауға көмектеседі.",
        "gallery_title": "Фотоматериалдар Ақмола облысындағы апаттардың",
        "photo_1": "Ақмола облысының жер бедері картасы",
        "photo_2": "Қазақстандағы су тасқыны жағдайы, 2024",
        "photo_3": "Су тасқынынан кейінгі қирау үлгісі",
        "about_title": "ARGUS жобасы туралы",
        "about_goal_title": "Жобаның мақсаты",
        "about_goal_body": "ARGUS гидрометеорологиялық параметрлер бойынша су басу ықтималдығын бағалау және ескерту мен эвакуация бойынша шешім қабылдауды қолдау үшін әзірленді.",
        "about_model_title": "Жүйе не істейді",
        "about_model_body": "Жүйе температураны, жауын-шашынды, қар жамылғысын, топырақ ылғалдылығын және өзен деңгейін талдап, қауіптің пайыздық бағасын береді.",
        "about_stack_title": "Технологиялық стек",
        "about_stack_body": "PyTorch, Streamlit, NumPy және Matplotlib.",
        "about_honesty_title": "Ашықтық қағидасы",
        "about_honesty_body": "ARGUS — тәуекелді бағалау құралы, ол ресми гидрометеорологиялық ескертулер мен төтенше қызметтердің жұмысын алмастырмайды.",
        "about_authors_title": "Авторлар және жетекші",
        "author_1_name": "Шамшидов Мирас",
        "author_1_body": "Модель, интерфейс және есептеу логикасы.",
        "author_2_name": "Бексултан Абдыхалык",
        "author_2_body": "Пәндік талдау, құрылым және жобаны таныстыру.",
        "supervisor_name": "Ольга Шорникова Николаевна",
        "supervisor_body": "Жобаның ғылыми жетекшісі.",
        "guide_title": "Қолдану нұсқаулығы",
        "guide_step_1": "1. Тізімнен қала немесе ауданды таңдаңыз.",
        "guide_step_2": "2. Параметрлерді енгізіңіз: температура, жауын-шашын, қар жамылғысы, топырақ ылғалдылығы және өзен деңгейі.",
        "guide_step_3": "3. Есептеу батырмасын басыңыз.",
        "guide_step_4": "4. Қауіп пайызын, түсіндірмені және графиктерді қараңыз.",
        "guide_for_title": "Кімге арналған",
        "guide_for_body": "Әкімдіктерге, азаматтық қорғау қызметтеріне, жедел бөлімшелерге, зерттеушілерге, мұғалімдерге және аймақ тұрғындарына арналған.",
        "guide_benefit_title": "Қалай көмектеседі",
        "guide_benefit_body": "Жүйе тәуекелдің өсуін алдын ала көрсетуге, эвакуация туралы шешімді қолдауға және ең қауіпті параметрлерге назар аударуға көмектеседі.",
        "guide_note": "Нәтижені енгізілген деректерге негізделген қауіп бағасы ретінде пайдалану керек.",
        "risk_hint": "Пайыз неғұрлым жоғары болса, қауіпті су тасқыны ықтималдығы соғұрлым жоғары.",
        "input_hint": "Барлық мәндер аймақ үшін шынайы өлшем бірліктерінде берілген.",
        "bolashaq_label": "Жоба атынан ұсынылды:",
        "bolashaq_city": "Көкшетау қ.",
    },
    "en": {
        "app_title": "ARGUS",
        "app_tagline": "Official flood prediction system for Akmola Region",
        "app_subtitle": "A decision-support tool for risk assessment and evacuation planning",
        "sidebar_city": "City / district",
        "sidebar_note": "The project was developed in 2026.",
        "sidebar_authors": "Authors: Miras Shamshidov, Beksultan Abdykhalyk",
        "sidebar_supervisor": "Supervisor: Olga Shornikova Nikolaevna",
        "tab_predict": "Prediction",
        "tab_region": "Region",
        "tab_about": "About",
        "tab_guide": "Guide",
        "section_predict": "Flood risk prediction",
        "section_results": "Result",
        "section_charts": "Risk factor charts",
        "temp": "Air temperature, °C",
        "rain": "Daily rainfall, mm",
        "snow": "Snow cover, cm",
        "soil": "Soil moisture, %",
        "river": "River level, cm",
        "calculate": "Calculate risk",
        "risk": "Flood risk",
        "low": "Low risk",
        "medium": "Medium risk",
        "high": "High risk",
        "metric_snowmelt": "Snowmelt",
        "metric_precip7": "7-day precipitation",
        "metric_saturation": "Saturation index",
        "region_title": "Akmola Region",
        "region_intro_title": "Why the region is vulnerable",
        "region_intro_body": "Akmola Region is especially exposed to spring floods driven by rapid snowmelt, rainfall, and high surface runoff. Flat terrain, frozen soil, and low infiltration increase the risk of sudden water rise.",
        "region_history_title": "Recent context",
        "region_history_body": "In 2024, floods in Kazakhstan were among the most extensive in decades. The 2017 floods also showed how vulnerable some settlements and engineering structures in the region can be.",
        "region_operational_title": "Operational value",
        "region_operational_body": "This interface is designed as an early risk assessment tool for emergency services, local authorities, and residents. It helps identify high-risk areas early and supports evacuation decisions.",
        "gallery_title": "Regional images — destruction in Akmola Region",
        "photo_1": "Relief map of Akmola Region",
        "photo_2": "Flood situation in Kazakhstan, 2024",
        "photo_3": "Example of post-flood destruction",
        "about_title": "About ARGUS",
        "about_goal_title": "Project purpose",
        "about_goal_body": "ARGUS was built to estimate flood likelihood from hydro-meteorological inputs and to support warning and evacuation decisions.",
        "about_model_title": "What the system does",
        "about_model_body": "The system analyzes temperature, rainfall, snow cover, soil moisture, and river level, then returns a probability estimate in percent.",
        "about_stack_title": "Tech stack",
        "about_stack_body": "PyTorch, Streamlit, NumPy, and Matplotlib.",
        "about_honesty_title": "Honesty principle",
        "about_honesty_body": "ARGUS is a risk-assessment tool and does not replace official hydrometeorological warnings or emergency services.",
        "about_authors_title": "Authors and supervisor",
        "author_1_name": "Miras Shamshidov",
        "author_1_body": "Model, interface, and calculation logic.",
        "author_2_name": "Beksultan Abdykhalyk",
        "author_2_body": "Domain analysis, structure, and project presentation.",
        "supervisor_name": "Olga Shornikova Nikolaevna",
        "supervisor_body": "Project supervisor.",
        "guide_title": "How to use",
        "guide_step_1": "1. Choose a city or district from the list.",
        "guide_step_2": "2. Enter temperature, rainfall, snow cover, soil moisture, and river level.",
        "guide_step_3": "3. Click the calculate button.",
        "guide_step_4": "4. Review the risk percentage, explanation, and charts.",
        "guide_for_title": "Who it is for",
        "guide_for_body": "For local authorities, civil protection services, emergency teams, researchers, teachers, and residents of the region.",
        "guide_benefit_title": "How it helps",
        "guide_benefit_body": "The system helps spot rising risk early, support evacuation decisions, and focus attention on the most dangerous parameters.",
        "guide_note": "Use the result as a risk estimate based on the entered data.",
        "risk_hint": "The higher the percentage, the higher the chance of a dangerous flood event.",
        "input_hint": "All values are given in realistic units for the region.",
        "bolashaq_label": "Project presented by:",
        "bolashaq_city": "Kokshetau city",
    },
}

# =========================================================
# MODEL
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
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_features(temp, rain, snow, soil_percent, river_cm, city) -> np.ndarray:
    soil = float(soil_percent) / 100.0
    snow_melt = max(0.0, float(temp)) * float(snow) * 0.12
    vals = {
        "temperature": float(temp), "rain": float(rain), "snow": float(snow),
        "soil_moisture": soil, "river_level": float(river_cm),
        "snow_melt": snow_melt,
        "precip_3d": float(rain) * 3.0, "precip_7d": float(rain) * 7.0,
        "temp_rain_inter": float(temp) * float(rain),
        "soil_river": soil * float(river_cm),
        f"city_{city}": 1.0,
    }
    row = np.array([vals.get(col, 0.0) for col in FEATURE_COLUMNS], dtype=np.float32)
    return row.reshape(1, -1)


def _generate_training_data(n: int = 12000):
    rng = np.random.default_rng(42)
    temp = rng.normal(loc=2.0, scale=12.0, size=n).clip(-35, 35)
    temp[rng.random(n) > 0.55] = rng.uniform(-5, 25, (rng.random(n) > 0.55).sum())
    rain = rng.gamma(shape=1.4, scale=14.0, size=n).clip(0, 300)
    rain[rng.random(n) < 0.07] = rng.uniform(40, 220, (rng.random(n) < 0.07).sum())
    snow = rng.gamma(shape=2.1, scale=18.0, size=n).clip(0, 300)
    snow = np.where(rng.random(n) < 0.10, rng.uniform(0, 40, n), snow)
    soil = rng.beta(2.4, 2.1, size=n) * 100.0
    river = rng.normal(loc=155.0, scale=50.0, size=n).clip(20, 450)
    cities = rng.choice(CITIES, size=n)

    X = np.vstack([build_features(t, r, s, so, rv, c)
                   for t, r, s, so, rv, c in zip(temp, rain, snow, soil, river, cities)])

    city_bias = np.array([CITY_BIAS[c] for c in cities], dtype=np.float32)
    snow_melt = np.maximum(temp, 0.0) * snow * 0.12
    risk_signal = (0.024*rain + 0.010*snow + 0.004*snow_melt
                   + 0.020*np.maximum(0.0, river-120.0)
                   + 0.90*(soil/100.0) + 0.018*np.maximum(0.0, -temp) + city_bias)
    y_prob = 1.0 / (1.0 + np.exp(-(risk_signal - 3.0)))
    y = torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1)

    scaler = StandardScalerLite().fit(X)
    X_t = torch.tensor(scaler.transform(X), dtype=torch.float32)
    mdl = FloodModel(INPUT_DIM)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    crit = nn.BCEWithLogitsLoss()
    mdl.train()
    for _ in range(80):
        opt.zero_grad()
        crit(mdl(X_t), y).backward()
        opt.step(); sch.step()
    mdl.eval()
    return scaler, mdl


@st.cache_resource(show_spinner=False)
def _load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            scaler = StandardScalerLite.load(SCALER_PATH)
            mdl = FloodModel(INPUT_DIM)
            mdl.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            mdl.eval()
            return scaler, mdl
        except Exception:
            for p in (MODEL_PATH, SCALER_PATH):
                try: os.remove(p)
                except: pass
    scaler, mdl = _generate_training_data()
    torch.save(mdl.state_dict(), MODEL_PATH)
    scaler.save(SCALER_PATH)
    return scaler, mdl


# =========================================================
# CSS  — ПОЛНЫЙ, все классы прописаны
# =========================================================
st.markdown("""
<style>
/* ---- base ---- */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif !important;
    background: #ffffff !important;
    color: #111827 !important;
}
.stApp { background: #ffffff !important; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child {
    background: #f7f8fa !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #111827 !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}

/* ---- hide Streamlit chrome ---- */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ---- header ---- */
.argus-brand {
    font-size: 3rem; font-weight: 800;
    letter-spacing: -0.04em; color: #111827 !important; line-height: 1.1;
}
.argus-tagline {
    margin-top: 0.3rem; font-size: 0.78rem; font-weight: 700;
    color: #374151 !important; letter-spacing: 0.1em; text-transform: uppercase;
}
.argus-subtitle {
    margin-top: 0.4rem; color: #4b5563 !important;
    font-size: 0.95rem; line-height: 1.6;
}

/* ---- section title ---- */
.section-title {
    margin: 1.2rem 0 0.8rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
    font-size: 1.3rem; font-weight: 700; color: #111827 !important;
}

/* ---- card ---- */
.card {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb;
    border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: 0.85rem;
    box-shadow: 0 4px 16px rgba(17,24,39,0.05);
}
.card-title {
    font-size: 0.98rem; font-weight: 700;
    color: #111827 !important; margin-bottom: 0.4rem;
}
.card-body {
    color: #374151 !important; font-size: 0.92rem; line-height: 1.7;
}

/* ---- result panel ---- */
.result-panel {
    background: #111827 !important;
    border-radius: 18px; padding: 1.2rem 1.4rem; margin-top: 1rem;
}
.result-label {
    font-size: 0.72rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #9ca3af !important; margin-bottom: 0.3rem;
}
.result-status {
    font-size: 1.35rem; font-weight: 700; margin-bottom: 0.15rem;
}
.result-number {
    font-size: 4rem; font-weight: 800; line-height: 1; letter-spacing: -0.05em;
}

/* ---- small note ---- */
.small-note { color: #6b7280 !important; font-size: 0.8rem; line-height: 1.5; }

/* ---- tech pills ---- */
.tech-pill {
    display: inline-block; background: #111827 !important; color: #fff !important;
    padding: 0.28rem 0.7rem; border-radius: 999px;
    margin: 0.2rem 0.2rem 0 0; font-size: 0.73rem; letter-spacing: 0.04em;
}

/* ---- bolashaq badge ---- */
.bolashaq-badge {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 12px; padding: 10px 14px; margin-bottom: 12px;
    font-size: 0.85rem; color: #1e3a8a !important; line-height: 1.6;
}
.bolashaq-badge strong { color: #1e40af !important; }

/* ---- image placeholder ---- */
.img-placeholder {
    background: #f3f4f6; border: 2px dashed #d1d5db;
    border-radius: 12px; height: 200px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: #9ca3af !important; font-size: 0.85rem;
    text-align: center; padding: 1rem; gap: 0.4rem;
}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: transparent;
    border-bottom: 2px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.75rem !important; font-weight: 700 !important;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #6b7280 !important; padding: 0.75rem 1.1rem;
    background: transparent !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #111827 !important;
    background: transparent !important;
}

/* ---- slider labels ---- */
div[data-testid="stSlider"] label p {
    font-size: 0.88rem !important; font-weight: 600 !important;
    color: #374151 !important;
}

/* ---- button ---- */
.stButton > button {
    width: 100%; height: 3rem; border: none !important;
    border-radius: 12px !important;
    background: #111827 !important; color: #ffffff !important;
    font-weight: 700 !important; font-size: 0.82rem !important;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.stButton > button:hover { background: #1f2937 !important; }

/* ---- selectbox labels ---- */
div[data-testid="stSelectbox"] label p {
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def tr(key: str) -> str:
    lang = st.session_state.get("lang", "ru")
    return T[lang].get(key, key)


def render_card(title: str, body: str):
    st.markdown(
        f'<div class="card">'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def risk_level(prob_pct: float):
    if prob_pct < 30:  return tr("low"),    "#16a34a"
    if prob_pct < 65:  return tr("medium"), "#d97706"
    return tr("high"), "#dc2626"


def show_image(local_path: str, caption: str = ""):
    p = pathlib.Path(local_path)
    if p.exists():
        st.image(str(p), use_container_width=True)
        if caption: st.caption(caption)
    else:
        st.markdown(
            f'<div class="img-placeholder">'
            f'<span style="font-size:2rem;">📷</span>'
            f'<span style="color:#374151;">{caption}</span>'
            f'<span style="color:#d1d5db;font-size:0.72rem;">Добавьте <b>{p.name}</b> в папку <b>images/</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if caption: st.caption(caption)


# =========================================================
# BOOT
# =========================================================
with st.spinner("Загрузка модели ARGUS…"):
    scaler, model = _load_model()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("<div class='argus-brand' style='font-size:2rem;'>ARGUS</div>", unsafe_allow_html=True)
    st.markdown("<div class='argus-tagline'>Flood Risk System</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Language — static label to avoid chicken-and-egg
    lang_label = st.selectbox(
        "Язык / Тіл / Language",
        ["Русский", "Қазақша", "English"],
        index=0, key="lang_selector",
    )
    st.session_state["lang"] = LANG_MAP[lang_label]

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.8rem 0;'>", unsafe_allow_html=True)
    city = st.selectbox(tr("sidebar_city"), CITIES, key="city_selector")
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.8rem 0;'>", unsafe_allow_html=True)

    # Bolashaq Saraiy badge
    st.markdown(
        f'<div class="bolashaq-badge">'
        f'<strong>{tr("bolashaq_label")}</strong><br>'
        f'<strong>Bolashaq Saraiy</strong><br>'
        f'<span style="color:#3b82f6;">{tr("bolashaq_city")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='argus-subtitle'>{tr('app_subtitle')}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_note')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_authors')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('sidebar_supervisor')}</div>", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown(f"<div class='argus-brand'>{tr('app_title')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='argus-tagline' style='margin-top:0.4rem;'>{tr('app_tagline')}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='argus-subtitle'>{tr('app_subtitle')}</div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    tr("tab_predict"), tr("tab_region"), tr("tab_about"), tr("tab_guide"),
])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab1:
    st.markdown(f"<div class='section-title'>{tr('section_predict')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('input_hint')}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider(tr("temp"), -50.0, 40.0, 5.0, step=0.5)
    with c2:
        rain = st.slider(tr("rain"), 0.0, 300.0, 8.0, step=0.5)
    with c3:
        snow = st.slider(tr("snow"), 0.0, 300.0, 25.0, step=1.0)

    c4, c5, _ = st.columns([1, 1, 1])
    with c4:
        soil_percent = st.slider(tr("soil"), 0.0, 100.0, 35.0, step=1.0)
    with c5:
        river = st.slider(tr("river"), 0.0, 500.0, 120.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        calculate = st.button(tr("calculate"), use_container_width=True)

    if calculate:
        x_raw    = build_features(temp, rain, snow, soil_percent, river, city)
        x_scaled = scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()
        risk_pct = round(prob * 100.0, 1)
        status, color = risk_level(risk_pct)

        st.markdown(
            f'<div class="result-panel" style="border-left:6px solid {color};">'
            f'  <div class="result-label">{city} — {tr("risk")}</div>'
            f'  <div class="result-status" style="color:{color};">{status}</div>'
            f'  <div class="result-number" style="color:{color};">{risk_pct}'
            f'    <span style="font-size:1.6rem;color:#6b7280;">%</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='small-note' style='margin-top:0.6rem;'>{tr('risk_hint')}</div>",
                    unsafe_allow_html=True)

        st.markdown(f"<div class='section-title' style='margin-top:1.2rem;'>{tr('section_results')}</div>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.metric(tr("metric_snowmelt"),  f"{round(max(0.0, temp)*snow*0.12, 1)} mm/day")
        with m2: st.metric(tr("metric_precip7"),   f"{round(rain*7.0, 1)} mm")
        with m3: st.metric(tr("metric_saturation"),f"{round((soil_percent/100.0)*river, 1)}")

        st.markdown(f"<div class='section-title' style='margin-top:1.2rem;'>{tr('section_charts')}</div>",
                    unsafe_allow_html=True)

        factors = {
            tr("temp"):  min(max(abs(temp)/40.0*100, 0), 100),
            tr("rain"):  min(rain/300.0*100, 100),
            tr("snow"):  min(snow/300.0*100, 100),
            tr("soil"):  soil_percent,
            tr("river"): min(river/500.0*100, 100),
        }
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        bars = ax1.bar(list(factors.keys()), list(factors.values()), color="#111827", width=0.5)
        ax1.set_ylim(0, 110)
        ax1.set_ylabel("% от максимума", fontsize=10, color="#374151")
        ax1.set_title(tr("section_charts"), fontsize=12, color="#111827", pad=8)
        ax1.tick_params(axis="x", rotation=10, labelsize=9, colors="#374151")
        ax1.tick_params(axis="y", labelsize=9, colors="#374151")
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#e5e7eb")
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h+1.5,
                     f"{h:.0f}%", ha="center", va="bottom", fontsize=8, color="#374151")
        fig1.tight_layout()
        st.pyplot(fig1, clear_figure=True)

        sensitivity = []
        rain_grid = np.linspace(0, 300, 60)
        for r in rain_grid:
            xr = build_features(temp, r, snow, soil_percent, river, city)
            xs = scaler.transform(xr)
            xt = torch.tensor(xs, dtype=torch.float32)
            with torch.no_grad():
                sensitivity.append(torch.sigmoid(model(xt)).item() * 100.0)

        fig2, ax2 = plt.subplots(figsize=(8, 3.2))
        ax2.plot(rain_grid, sensitivity, color="#111827", linewidth=2)
        ax2.fill_between(rain_grid, sensitivity, alpha=0.08, color="#111827")
        ax2.set_xlabel(tr("rain"), fontsize=10, color="#374151")
        ax2.set_ylabel(f"{tr('risk')}, %", fontsize=10, color="#374151")
        ax2.set_title(f"{tr('risk')} / {tr('rain')}", fontsize=12, color="#111827", pad=8)
        ax2.tick_params(labelsize=9, colors="#374151")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.spines[["left","bottom"]].set_color("#e5e7eb")
        ax2.set_ylim(0, 105)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True)

# =========================================================
# TAB 2 — REGION
# =========================================================
with tab2:
    st.markdown(f"<div class='section-title'>{tr('region_title')}</div>", unsafe_allow_html=True)
    render_card(tr("region_intro_title"),     tr("region_intro_body"))
    render_card(tr("region_history_title"),   tr("region_history_body"))
    render_card(tr("region_operational_title"), tr("region_operational_body"))



# =========================================================
# TAB 3 — ABOUT
# =========================================================
with tab3:
    st.markdown(f"<div class='section-title'>{tr('about_title')}</div>", unsafe_allow_html=True)
    render_card(tr("about_goal_title"),    tr("about_goal_body"))
    render_card(tr("about_model_title"),   tr("about_model_body"))
    render_card(tr("about_stack_title"),   tr("about_stack_body"))
    render_card(tr("about_honesty_title"), tr("about_honesty_body"))

    st.markdown(f"<div class='section-title' style='margin-top:1.2rem;'>{tr('about_authors_title')}</div>",
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        render_card(tr("author_1_name"), tr("author_1_body"))
        render_card(tr("author_2_name"), tr("author_2_body"))
    with right:
        render_card(tr("supervisor_name"), tr("supervisor_body"))
        st.markdown(
            f'<div class="bolashaq-badge" style="margin-top:0.1rem;">'
            f'<strong>{tr("bolashaq_label")}</strong><br>'
            f'<strong>Bolashaq Saraiy</strong><br>'
            f'<span style="color:#3b82f6;">{tr("bolashaq_city")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='margin-top:0.8rem;'>"
        "<span class='tech-pill'>PyTorch</span>"
        "<span class='tech-pill'>Streamlit</span>"
        "<span class='tech-pill'>NumPy</span>"
        "<span class='tech-pill'>Matplotlib</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================
# TAB 4 — GUIDE
# =========================================================
with tab4:
    st.markdown(f"<div class='section-title'>{tr('guide_title')}</div>", unsafe_allow_html=True)
    render_card(
        tr("guide_title"),
        f"{tr('guide_step_1')}<br>{tr('guide_step_2')}<br>{tr('guide_step_3')}<br>{tr('guide_step_4')}",
    )
    render_card(tr("guide_for_title"),     tr("guide_for_body"))
    render_card(tr("guide_benefit_title"), tr("guide_benefit_body"))
    render_card("", tr("guide_note"))
