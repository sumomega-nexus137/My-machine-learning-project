import streamlit as st
import pathlib

st.set_page_config(page_title="Flood App", layout="wide")

# =========================
# TRANSLATION (example)
# =========================
T = {
    "ru": {
        "photo_1": "Карта Акмолинской области",
        "photo_2": "Паводок 2024",
        "photo_3": "Разрушения",
    },
    "en": {
        "photo_1": "Akmola region map",
        "photo_2": "Flood 2024",
        "photo_3": "Flood damage",
    },
    "kk": {
        "photo_1": "Ақмола облысының картасы",
        "photo_2": "2024 жылғы су тасқыны",
        "photo_3": "Зиян",
    }
}

LANG = "ru"  # можно потом менять через selectbox


def tr(key: str):
    return T.get(LANG, T["ru"]).get(key, key)


# =========================
# SAFE IMAGE (ТВОЙ ВАРИАНТ)
# =========================
def safe_image(path_or_url: str, caption: str = ""):
    p = pathlib.Path(path_or_url)
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.markdown(
            f"<div style='background:#e5e7eb;border-radius:12px;height:200px;"
            f"display:flex;align-items:center;justify-content:center;"
            f"color:#9ca3af;font-size:0.85rem;'>{caption}</div>",
            unsafe_allow_html=True,
        )
    if caption:
        st.caption(caption)


# =========================
# UI
# =========================
st.title("Flood Monitoring System")

st.subheader("Akmola Region")

col_a, col_b, col_c = st.columns(3)

with col_a:
    safe_image("images/map_akmola.png", tr("photo_1"))

with col_b:
    safe_image("images/flood_2024.jpg", tr("photo_2"))

with col_c:
    safe_image("images/flood_damage.jpg", tr("photo_3"))
