import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from fpdf import FPDF
import io
import random

st.set_page_config(page_title="Dynamic Meal Recommender", layout="wide")

# Sample CSV dataset (replace with your CSV load if needed)
CSV_CONTENT = """item id,item,servesize,calories,protien,totalfat,satfat,transfat,cholestrol,carbs,sugar,addedsugar,sodium,menu,Ratings
0,McVeggie Burger,168 ,402,10.24,13.83,5.34,0.16,2.49,56.54,7.9,4.49,706.13,regular,1
1,McAloo Tikki Burger,146 ,339,8.5,11.31,4.27,0.2,1.47,5.27,7.05,4.07,545.34,regular,15
2,McSpicy Paneer Burger,199 ,652,20.29,39.45,17.12,0.18,21.85,52.33,8.35,5.27,1074.58,regular,19
3,Spicy Paneer Wrap,250 ,674,20.96,39.1,19.73,0.26,40.93,59.27,3.5,1.08,1087.46,regular,10
4,American Veg Burger,177 ,512,15.3,23.45,10.51,0.17,25.24,56.96,7.85,4.76,1051.24,regular,11
5,Veg Maharaja Mac ,306,832,24.17,37.94,16.83,0.28,36.19,93.84,11.52,6.92,1529.22,regular,17
6,Green Chilli Aloo Naan p,132  ,356,7.91,15.08,6.11,0.24,9.45,46.36,4.53,1.15,579.6,regular,1
7,Pizza Puff ,87 ,228,5.45,11.44,5.72,0.09,5.17,24.79,2.73,0.35,390.74,regular,20
8,Mc chicken Burger , 173 ,400,15.66,15.7,5.47,0.16,31.17,47.98,5.53,4.49,766.33,regular,7
9,FILLET-O-FISH Burger , 136 ,348,15.44,14.16,5.79,0.21,32.83,38.85,5.88,3.54,530.54,regular,6
10, Mc Spicy Chicken Burger ,186 ,451,21.46,19.36,7.63,0.18,66.04,46.08,2.52,4.49,928.52,regular,19
11, Spicy Chicken Wrap , 257 ,567,23.74,26.89,12.54,0.27,87.63,57.06,8.92,1.08,1152.38,regular,8
12, Chicken Maharaja Mac , 296  ,689,34,36.69,10.33,0.25,81.49,55.39,7.48,6.14,1854.71,regular,9
13,American Chicken Burger ,165  ,446,20.29,22.94,7.28,0.15,47.63,38.54,5.08,4.76,1132.3,regular,6
14,Chicken Kebab Burger ,138  ,357,8.64,14.02,4.84,0.13,1.51,47.9,3.64,3.49,548.79,regular,9
15,Green Chilli Kebab naan,138,230,5.67,9.32,3.27,0.19,8.74,31.06,4.89,1.15,410.78,regular,6
16,Mc Egg Masala Burger, 126.2 ,290,12.45,12.27,3.64,0.11,213.09,32.89,4.66,3.64,757.91,regular,1
17,Mc Egg Burger for Happy Meal,123,282,12.29,12.21,3.63,0.11,213.09,31.32,3.28,3.64,399.41,regular,5
18,Ghee Rice with Mc Spicy Fried Chicken 1 pc,325,720,26.91,29.2,5.08,0.3,31.32,77.47,0.58,0.35,2399.49,regular,8
19,McSpicy Fried Chicken 1 pc,115,248,17.33,14.29,2.82,0.06,31.11,12.7,0.32,0,873.89,regular,3
20,4 piece Chicken McNuggets ,64,169,10.03,9.54,4.45,0.1,24.66,10.5,0.72,0,313.25,regular,7
21,6 piece Chicken McNuggets ,96,254,15.04,14.3,6.68,0.14,36.99,15.74,0.29,0,469.87,regular,2
22,9 piece Chicken McNuggets ,144 ,381,22.56,21.46,10.02,0.06,55.48,23.62,0.44,0,704.81,regular,7
23,2 piece Chicken Strips ,58 ,164,10.17,12.38,11.41,0.09,30.1,2.68,0.72,0,477.22,regular,1
24,3 piece Chicken Strips , 87 ,246,15.26,18.57,17.12,75.26,45.15,4.02,0.39,0,715.83,regular,19
25,5 piece Chicken Strips , 145 ,411,25.43,28.54,0.15,0.08,6.7,0.73,0.55,0,1193.052,regular,8"""


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(CSV_CONTENT), on_bad_lines='skip', delimiter=',')
    df['calories'] = df['calories'].astype(float)
    df['protien'] = df['protien'].astype(float)
    df['totalfat'] = df['totalfat'].astype(float)
    df['carbs'] = df['carbs'].astype(float)
    return df

df = load_data()

# Generate fake user preferences for collaborative filtering
def generate_user_preferences(df):
    np.random.seed(42)
    num_users = 100
    user_preferences = pd.DataFrame(index=range(num_users), columns=df.index)
    for user in range(num_users):
        for meal in np.random.choice(df.index, size=int(0.2 * len(df)), replace=False):
            user_preferences.loc[user, meal] = np.random.randint(1, 6)
    return user_preferences.fillna(0)

user_preferences = generate_user_preferences(df)

def calculate_bmi(weight, height):
    if height == 0:
        return 0
    return weight / ((height / 100) ** 2)

def content_based_recommendations(bmi, df):
    df_norm = df.copy()
    for feature in ['calories', 'protien', 'totalfat', 'carbs']:
        df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / (df_norm[feature].max() - df_norm[feature].min())

    features = df_norm[['calories', 'protien', 'totalfat', 'carbs']].values

    calorie_weight = np.clip(1 - (bmi / 40), 0, 1) + np.random.uniform(-0.05, 0.05)
    protein_weight = np.clip((bmi / 40), 0, 1) + np.random.uniform(-0.05, 0.05)
    fat_weight = np.clip(1 - abs(22 - bmi) / 40, 0, 1) + np.random.uniform(-0.05, 0.05)
    carbs_weight = np.clip(1 - (bmi / 40), 0, 1) + np.random.uniform(-0.05, 0.05)

    user_profile = [round(w, 2) for w in [calorie_weight, protein_weight, fat_weight, carbs_weight]]
    profile_sum = sum(user_profile)
    user_profile = [round(w / profile_sum, 2) for w in user_profile]

    calorie_mean = df['calories'].mean()
    calorie_std = df['calories'].std()

    if bmi < 18.5:
        filtered = df[df['calories'] >= calorie_mean + (bmi / 18.5) * calorie_std]
        recommendation_type = f"High-Calorie, High-Protein Meals (BMI {bmi:.1f})"
        num_recommendations = 6
    elif bmi < 25:
        filtered = df[
            (df['calories'] >= calorie_mean - (bmi - 18.5) * calorie_std / 2) &
            (df['calories'] <= calorie_mean + (25 - bmi) * calorie_std / 2)
        ]
        recommendation_type = f"Balanced Meals (BMI {bmi:.1f})"
        num_recommendations = 4
    else:
        filtered = df[df['calories'] <= calorie_mean - (bmi - 25) * calorie_std / 2]
        recommendation_type = f"Low-Calorie, High-Protein Meals (BMI {bmi:.1f})"
        num_recommendations = 3

    filtered = filtered.sample(frac=1, random_state=int(bmi * 100))

    similarities = cosine_similarity([user_profile], features)
    indices = filtered.index.tolist()
    similarity_scores = [(i, similarities[0][i]) for i in indices]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in similarity_scores[:num_recommendations]]

    return filtered.loc[top_indices], recommendation_type, num_recommendations

def collaborative_filtering(user_preferences, content_recs, num_recs, bmi):
    cb_indices = content_recs.index.tolist()
    current_user = pd.Series(0, index=user_preferences.columns)
    current_user[cb_indices] = 5

    bmi_noise = np.random.normal(loc=bmi / 40, scale=0.1, size=len(current_user))
    current_user = current_user + bmi_noise

    user_similarities = cosine_similarity([current_user.values], user_preferences.values)[0]
    similar_users = user_preferences.iloc[np.argsort(user_similarities)[-10:]]

    collaborative_recs = pd.Series(0, index=user_preferences.columns)
    for _, user in similar_users.iterrows():
        collaborative_recs += user

    collaborative_recs[cb_indices] = 0
    collaborative_recs = collaborative_recs.sample(frac=1, random_state=int(bmi * 200))

    top_indices = collaborative_recs.nlargest(num_recs).index.tolist()

    return df.loc[top_indices]

def show_radar_chart(selected_meals):
    categories = ['calories', 'protien', 'totalfat', 'carbs']
    fig = go.Figure()

    for _, meal in selected_meals.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[meal['calories'], meal['protien'], meal['totalfat'], meal['carbs']],
            theta=categories,
            fill='toself',
            name=meal['item']
        ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def generate_diet_plan_pdf(user_name, bmi, content_recs, collab_recs):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{user_name}'s Personalized Diet Plan", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"BMI: {bmi:.1f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Content-Based Recommendations", ln=True)

    pdf.set_font("Arial", size=12)
    for _, row in content_recs.iterrows():
        pdf.multi_cell(0, 10, txt=f"- {row['item']} (Calories: {row['calories']}, Protein: {row['protien']}g)")

    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Collaborative Recommendations", ln=True)

    pdf.set_font("Arial", size=12)
    for _, row in collab_recs.iterrows():
        pdf.multi_cell(0, 10, txt=f"- {row['item']} (Calories: {row['calories']}, Protein: {row['protien']}g)")

    file_name = f"{user_name}_diet_plan.pdf"
    pdf.output(file_name)
    return file_name

# --- Streamlit App Layout ---
st.title("Dynamic BMI Meal Recommender 🍔🥗")

col1, col2 = st.columns(2)
with col1:
    weight = st.number_input("Enter your weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
with col2:
    height = st.number_input("Enter your height (cm)", min_value=0.0, max_value=250.0, value=170.0)

bmi = calculate_bmi(weight, height)
st.subheader(f"Your BMI: {bmi:.1f}")

if bmi < 18.5:
    st.warning("You are underweight.")
elif bmi < 25:
    st.success("You have a normal weight.")
else:
    st.error("You are overweight.")

if 'rec_history' not in st.session_state:
    st.session_state.rec_history = []

if st.button("Generate Meal Recommendations"):
    content_recs, rec_type, num_recs = content_based_recommendations(bmi, df)
    collab_recs = collaborative_filtering(user_preferences, content_recs, num_recs, bmi)

    st.session_state.rec_history.append({
        'bmi': bmi,
        'recommendations': content_recs['item'].tolist()
    })

    st.header(rec_type)
    st.dataframe(content_recs[['item', 'calories', 'protien', 'carbs', 'totalfat']])

    st.subheader("You Might Also Like (Collaborative Recs)")
    st.dataframe(collab_recs[['item', 'calories', 'protien', 'carbs', 'totalfat']])

    st.subheader("Nutrient Radar Chart")
    show_radar_chart(content_recs)

    user_name = st.text_input("Enter your name for the PDF")

    if st.button("Download Diet Plan as PDF"):
        pdf_file = generate_diet_plan_pdf(user_name or "User", bmi, content_recs, collab_recs)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Click to Download PDF",
                data=f,
                file_name=pdf_file,
                mime="application/octet-stream"
            )

st.sidebar.header("Recommendation History")
for rec in st.session_state.rec_history[-5:]:
    st.sidebar.write(f"BMI: {rec['bmi']:.1f}")
    for meal in rec['recommendations']:
        st.sidebar.write(f"• {meal}")
