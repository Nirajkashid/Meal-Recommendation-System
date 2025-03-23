import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="Meal Recommender | Home", layout="wide", initial_sidebar_state="expanded")

# --------- LOAD CUSTOM CSS ---------
with open('style1.css') as f:
    css = f.read()

# Inject CSS (Add dark mode tweaks if you want directly here)
dark_mode_css = """
<style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .feature-box {
        background-color: #2c2c2c;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B !important;
    }
    .stMarkdown {
        color: white;
    }
</style>
"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.markdown(dark_mode_css, unsafe_allow_html=True)

# --------- RESET SESSION BUTTON (Optional) ---------
with st.sidebar:
    st.header("🔧 Settings")
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session reset! Starting fresh...")

# --------- LOAD LOTTIE ANIMATION ---------
def load_lottie(path: str):
    with open(path, "r") as p:
        return json.load(p)

lottie_path = load_lottie("./ani.json")

# --------- HOME PAGE LAYOUT ---------
st.title("🍽️ Meal Recommender System")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("""
    <div style='border-left: 5px solid #FF4B4B; padding-left: 1rem;'>
        <h3>Personalized Meal Recommendations Based on Your BMI</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
        <h4>✨ Features:</h4>
        <ul>
            <li><b>BMI-Based Personalized Recommendations</b></li>
            <li><b>Dynamic Content & Collaborative Filtering</b></li>
            <li><b>Detailed Nutrition Insights</b></li>
            <li><b>Interactive Data Visualizations</b></li>
            <li><b>Export PDF Reports & Charts</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        st.markdown("### 🍔 Ready to get started?")
        if st.button("Get Meal Recommendations", use_container_width=True):
            switch_page("Meal_Recommender")

    with col_btn2:
        st.markdown("### 📊 Visualize Nutrition Insights")
        if st.button("View Visualizations", use_container_width=True):
            switch_page("Visualizations")

with col2:
    st_lottie(lottie_path, height=400, key="home_animation")

# --------- FOOTER ---------
st.markdown("""
<hr style='border: 1px solid #444;' />
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    &copy; 2025 Meal Recommender | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
