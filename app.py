"""
ARGUS — Flood Risk Prediction System

Single-file Streamlit app.

FIXED VERSION for GitHub / Streamlit Cloud
✅ Images now load reliably using pathlib (this was the problem)
✅ Completed the incomplete _load_model function
✅ Everything else is exactly as you wrote it

Requirements:
streamlit
torch
numpy
matplotlib
Pillow

Put your photos in an images/ folder next to this file:
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
import matplotlib.pyplot as plt
from PIL import Image

# ========================================================
# PAGE CONFIG — must be the very first Streamlit call
# ========================================================
st.set_page_config(
    page_title="ARGUS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================================
# ROBUST IMAGE LOADING — FIXED FOR GITHUB
# ========================================================
BASE_DIR = pathlib.Path(__file__).parent.resolve()
IMAGES_DIR = BASE_DIR / "images"

def load_image(filename: str):
    """Fixed image loader that works on GitHub + Streamlit Cloud"""
    path = IMAGES_DIR / filename
    if not path.exists():
        st.error(f"❌ Image not found: {filename}")
        st.info(f"Expected path: {path}")
        st.info("Make sure the 'images' folder is in your GitHub repo root and all 3 files are committed.")
        return None
    return Image.open(str(path))

# ========================================================
# CONSTANTS
# ========================================================
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

# ========================================================
# TRANSLATIONS (exactly as you provided)
# ========================================================
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
        "gallery_title": "Фотоматериалдар және аймақ картасы",
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
        "gallery_title": "Regional images and map",
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
    },
}

# ========================================================
# MODEL / SCALER (exactly as you wrote)
# ========================================================
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

    rain = rng.gamma(shape=1.4, scale=14.0, size=n).clip(0, 300)
    heavy_mask = rng.random(n) < 0.07
    rain[heavy_mask] = rng.uniform(40, 220, heavy_mask.sum())

    snow = rng.gamma(shape=2.1, scale=18.0, size=n).clip(0, 300)
    snow = np.where(rng.random(n) < 0.10, rng.uniform(0, 40, n), snow)

    soil = rng.beta(2.4, 2.1, size=n) * 100.0
    river = rng.normal(loc=155.0, scale=50.0, size=n).clip(
