"""
ARGUS — Flood Risk Prediction System
Single-file Streamlit app.
Requirements:
    streamlit
    torch
    numpy
    matplotlib
    Pillow
Now using public image URLs (no images/ folder needed)
"""
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
    "Атбасар": 0.28, "Акколь": 0.16, "Макинск": 0.18,
    "Есиль": 0.14, "Ерейментау": 0.09, "Степняк": 0.11,
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
        "gallery_title": "Фотоматериалы разрушений от паводков в Акмолинской Области",
       
       
   
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
    "kk": {
        "app_title": "ARGUS",
        "app_tagline": "Ақмола облысы үшін су тасқынын болжаудың ресми жүйесі",
        "app_subtitle": "Қауіпті бағалау және эвакуацияны жоспарлау үшін шешім қабылдауды қолдайтын құрал",
        "sidebar_language": "Тіл",
        "sidebar_city": "Қала / аудан",
        "sidebar_note": "Жоба 2026 жылы әзірленді.",
        "sidebar_authors": "Авторлар: Шамшидов Мирас, Бексултан Абдыхалык",
        "sidebar_supervisor": "Ғылыми жетекші: Ольга Шорникова Николаевна",
        "tab_predict": "Болжам",
        "tab_region": "Аймақ",
        "tab_about": "Жоба туралы",
        "tab_guide": "Нұсқаулық",
        "section_predict": "Су басу қаупін болжау",
        "section_inputs": "Кіріс параметрлері",
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
        "metric_city": "Елді мекен",
        "region_title": "Ақмола облысы",
        "region_intro_title": "Неліктен аймақ осал",
        "region_intro_body": "Ақмола облысы үшін көктемгі су тасқыны ерекше тән. Ол қардың тез еруімен, жауын-шашынмен және жер бетімен ағып келетін судың көбеюімен байланысты. Жазық рельеф, тоңазытылған топырақ және төмен сіңіргіштік тәуекелді күшейтеді.",
        "region_history_title": "Соңғы жылдардағы жағдай",
        "region_history_body": "2024 жылы Қазақстандағы су тасқыны соңғы онжылдықтардағы ең ауқымдылардың бірі болды. Сондай-ақ 2017 жылғы жағдай өңірдің жекелеген елді мекендері мен инженерлік инфрақұрылымының осалдығын көрсетті.",
        "region_operational_title": "Жедел маңызы",
        "region_operational_body": "Бұл интерфейс төтенше жағдай қызметтері, жергілікті билік және тұрғындар үшін алдын ала тәуекелді бағалау құралы ретінде жасалған. Ол қауіпті аймақтарды ерте анықтауға және эвакуация туралы шешімді қолдауға көмектеседі.",
        "gallery_title": "Фотоматериалдар Ақмола облысндағы аппатардың",
   
       
       
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
    },
    "en": {
        "app_title": "ARGUS",
        "app_tagline": "Official flood prediction system for Akmola Region",
        "app_subtitle": "A decision-support tool for risk assessment and evacuation planning",
        "sidebar_language": "Language",
        "sidebar_city": "City / district",
        "sidebar_note": "The project was developed in 2026.",
        "sidebar_authors": "Authors: Miras Shamshidov, Beksultan Abdykhalyk",
        "sidebar_supervisor": "Supervisor: Olga Shornikova Nikolaevna",
        "tab_predict": "Prediction",
        "tab_region": "Region",
        "tab_about": "About",
        "tab_guide": "Guide",
        "section_predict": "Flood risk prediction",
        "section_inputs": "Input parameters",
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
        "metric_city": "Settlement",
        "region_title": "Akmola Region",
        "region_intro_title": "Why the region is vulnerable",
        "region_intro_body": "Akmola Region is especially exposed to spring floods driven by rapid snowmelt, rainfall, and high surface runoff. Flat terrain, frozen soil, and low infiltration increase the risk of sudden water rise.",
        "region_history_title": "Recent context",
        "region_history_body": "In 2024, floods in Kazakhstan were among the most extensive in decades. The 2017 floods also showed how vulnerable some settlements and engineering structures in the region can be.",
        "region_operational_title": "Operational value",
        "region_operational_body": "This interface is designed as an early risk assessment tool for emergency services, local authorities, and residents. It helps identify high-risk areas early and supports evacuation decisions.",
        "gallery_title": "Regional images destruction in Akmola region",
   
       
       
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
        
    },
}
# =========================================================
# MODEL / SCALER
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
    rng = np.random.default_rng(42)
    temp = rng.normal(loc=2.0, scale=12.0, size=n).clip(-35, 35)
    warm_mask = rng.random(n) > 0.55
    temp[warm_mask] = rng.uniform(-5, 25, warm_mask.sum())
    rain = rng.gamma(shape=1.4, scale=14.0, size=n).clip(0, 3000)
    heavy_mask = rng.random(n) < 0.07
    rain[heavy_mask] = rng.uniform(40, 220, heavy_mask.sum())
    snow = rng.gamma(shape=2.1, scale=18.0, size=n).clip(0, 300)
    snow = np.where(rng.random(n) < 0.10, rng.uniform(0, 40, n), snow)
    soil = rng.beta(2.4, 2.1, size=n) * 100.0
    river = rng.normal(loc=155.0, scale=50.0, size=n).clip(20, 450)
    cities = rng.choice(CITIES, size=n)
    X = np.vstack([
        build_features(t, r, s, so, rv, c)
        for t, r, s, so, rv, c in zip(temp, rain, snow, soil, river, cities)
    ])
    city_bias = np.array([CITY_BIAS[c] for c in cities], dtype=np.float32)
    snow_melt = np.maximum(temp, 0.0) * snow * 0.12
    risk_signal = (
        0.024 * rain
        + 0.010 * snow
        + 0.004 * snow_melt
        + 0.020 * np.maximum(0.0, river - 120.0)
        + 0.90 * (soil / 100.0)
        + 0.018 * np.maximum(0.0, -temp)
        + city_bias
    )
    y_prob = 1.0 / (1.0 + np.exp(-(risk_signal - 3.0) / 1.0))
    y = torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1)
    scaler = StandardScalerLite().fit(X)
    X_sc = scaler.transform(X)
    X_t = torch.tensor(X_sc, dtype=torch.float32)
    mdl = FloodModel(INPUT_DIM)
    optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    criterion = nn.BCEWithLogitsLoss()
    mdl.train()
    for _ in range(80):
        optimizer.zero_grad()
        logits = mdl(X_t)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
    mdl.eval()
    return scaler, mdl
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
# CSS
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
# HELPERS
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
def show_image(src: str, caption: str = ""):
    """
    Display image from URL or local path.
    Now supports public URLs so no images/ folder is needed.
    """
    if src.startswith(("http://", "https://")):
        # Public URL — just display it
        st.image(src, use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        # Original local file logic (kept unchanged)
        p = pathlib.Path(src)
        if p.exists():
            st.image(str(p), use_container_width=True)
            if caption:
                st.caption(caption)
        else:
            filename = p.name
            st.markdown(
                f'<div class="img-placeholder">'
                f'<span style="font-size:2rem;">📷</span>'
                f'<span>{caption}</span>'
                f'<span style="color:#d1d5db;font-size:0.75rem;">Add <b>{filename}</b> to <b>images/</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if caption:
                st.caption(caption)
# =========================================================
# BOOT
# =========================================================
with st.spinner("Loading ARGUS model…"):
    scaler, model = _load_model()
# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("<div class='argus-brand'>ARGUS</div>", unsafe_allow_html=True)
    st.markdown("<div class='argus-tagline'>Flood Risk System</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'>", unsafe_allow_html=True)
    # Language selector — static label avoids tr() chicken-and-egg problem
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
# HEADER
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
# TAB 1 — PREDICTION
# =========================================================
with tab1:
    st.markdown(f"<div class='section-title'>{tr('section_predict')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-note'>{tr('input_hint')}</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider(tr("temp"), -50.0, 40.0, 5.0, step=0.5)
    with c2:
        rain = st.slider(tr("rain"), 0.0, 2000.0, 8.0, step=0.5)
    with c3:
        snow = st.slider(tr("snow"), 0.0, 300.0, 25.0, step=1.0)
    c4, c5 = st.columns(2)
    with c4:
        soil_percent = st.slider(tr("soil"), 0.0, 100.0, 35.0, step=1.0)
    with c5:
        river = st.slider(tr("river"), 0.0, 500.0, 120.0, step=1.0)
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        calculate = st.button(tr("calculate"), use_container_width=True)
    if calculate:
        x_raw = build_features(temp, rain, snow, soil_percent, river, city)
        x_scaled = scaler.transform(x_raw)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(x_tensor)).item()
        risk_pct = round(prob * 100.0, 1)
        status, color = risk_level(risk_pct)
        st.markdown(
            f'<div class="result-panel" style="border-left:6px solid {color};">'
            f'<div class="result-label">{city} — {tr("risk")}</div>'
            f'<div class="result-status" style="color:{color};">{status}</div>'
            f'<div class="result-number" style="color:{color};">{risk_pct}'
            f'<span style="font-size:1.8rem;color:#cbd5e1;">%</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='small-note' style='margin-top:0.65rem;'>{tr('risk_hint')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title' style='margin-top:1.3rem;'>{tr('section_results')}</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(tr("metric_snowmelt"), f"{round(max(0.0, temp) * snow * 0.12, 1)} mm/day")
        with m2:
            st.metric(tr("metric_precip7"), f"{round(rain * 7.0, 1)} mm")
        with m3:
            st.metric(tr("metric_saturation"), f"{round((soil_percent / 100.0) * river, 1)}")
        st.markdown(f"<div class='section-title' style='margin-top:1.3rem;'>{tr('section_charts')}</div>", unsafe_allow_html=True)
        factors = {
            tr("temp"): min(max(abs(temp) / 40.0 * 100.0, 0.0), 100.0),
            tr("rain"): min(rain / 3000.0 * 100.0, 100.0),
            tr("snow"): min(snow / 300.0 * 100.0, 100.0),
            tr("soil"): soil_percent,
            tr("river"): min(river / 500.0 * 100.0, 100.0),
        }
        fig1, ax1 = plt.subplots(figsize=(8.8, 3.8))
        ax1.bar(list(factors.keys()), list(factors.values()), color="#111827")
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("%")
        ax1.set_title(tr("section_charts"))
        ax1.tick_params(axis="x", rotation=12)
        st.pyplot(fig1, clear_figure=True)
        sensitivity = []
        rain_grid = np.linspace(0, 300, 50)
        for r in rain_grid:
            xr = build_features(temp, r, snow, soil_percent, river, city)
            xs = scaler.transform(xr)
            xt = torch.tensor(xs, dtype=torch.float32)
            with torch.no_grad():
                sensitivity.append(torch.sigmoid(model(xt)).item() * 100.0)
        fig2, ax2 = plt.subplots(figsize=(8.8, 3.6))
        ax2.plot(rain_grid, sensitivity, color="#111827")
        ax2.set_xlabel(tr("rain"))
        ax2.set_ylabel(f"{tr('risk')}, %")
        ax2.set_title(f"{tr('risk')} / {tr('rain')}")
        st.pyplot(fig2, clear_figure=True)
# =========================================================
# TAB 2 — REGION
# =========================================================
# =========================================================
# TAB 2 — REGION
# =========================================================
# =========================================================
# TAB 2 — REGION
# =========================================================
# =========================================================
# TAB 2 — REGION
# =========================================================
# =========================================================
# TAB 2 — REGION
# =========================================================
with tab2:
    st.markdown(f"<div class='section-title'>{tr('region_title')}</div>", unsafe_allow_html=True)
    render_card(tr("region_intro_title"), tr("region_intro_body"))
    render_card(tr("region_history_title"), tr("region_history_body"))
    render_card(tr("region_operational_title"), tr("region_operational_body"))
    
    
    
# =========================================================
with tab3:
    st.markdown(f"<div class='section-title'>{tr('about_title')}</div>", unsafe_allow_html=True)
    render_card(tr("about_goal_title"), tr("about_goal_body"))
    render_card(tr("about_model_title"), tr("about_model_body"))
    render_card(tr("about_stack_title"), tr("about_stack_body"))
    render_card(tr("about_honesty_title"), tr("about_honesty_body"))
    st.markdown(f"<div class='section-title' style='margin-top:1.2rem;'>{tr('about_authors_title')}</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        render_card(tr("author_1_name"), tr("author_1_body"))
        render_card(tr("author_2_name"), tr("author_2_body"))
    with right:
        render_card(tr("supervisor_name"), tr("supervisor_body"))
        render_card(tr("sidebar_note"), f"{tr('sidebar_authors')}<br>{tr('sidebar_supervisor')}")
    st.markdown(
        "<div style='margin-top:0.5rem;'>"
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
    render_card(tr("guide_for_title"), tr("guide_for_body"))
    render_card(tr("guide_benefit_title"), tr("guide_benefit_body"))
    render_card("", tr("guide_note"))
