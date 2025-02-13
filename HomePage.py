import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr


st.title('MEAL RECOMMENDATION SYSTEM')

# Load CSS from a file
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def get(path: str):
    with open(path, "r") as p:
        return json.load(p)

path = get("./ani.json")

# Use the full_width property to make the Lottie animation occupy the entire horizontal space
st_lottie(path, width=None)

# Use CSS to make the Lottie animation occupy the entire vertical space
st.markdown("""
    <style>
        div.stLottie {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Add a button to switch to the meal_recommender page
if st.button("Let's find the best for you!!"):
    switch_page("Meal_Recommender")

# Add BMI input in the homepage
st.sidebar.header("BMI Calculator")
weight = st.sidebar.number_input("Enter your weight (kg):", min_value=0.0)
height = st.sidebar.number_input("Enter your height (cm):", min_value=0.0)

if weight > 0 and height > 0:
    bmi = weight / ((height / 100) ** 2)
    st.sidebar.write(f"Your BMI is: {bmi:.2f}")
    if bmi < 18.5:
        st.sidebar.write("You are underweight.")
    elif 18.5 <= bmi < 24.9:
        st.sidebar.write("You have a normal weight.")
    elif 25 <= bmi < 29.9:
        st.sidebar.write("You are overweight.")
    else:
        st.sidebar.write("You are obese.")