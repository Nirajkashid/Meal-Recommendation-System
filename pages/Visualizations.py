import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import os

# Optional: switch_page helper (if using multi-page apps)
try:
    from streamlit_extras.switch_page_button import switch_page
    MULTI_PAGE = True
except ImportError:
    MULTI_PAGE = False

# Session state validation
if 'content_recs' not in st.session_state:
    st.warning("No recommendations yet. Please generate recommendations first.")
    if MULTI_PAGE:
        if st.button("Go to Meal Recommender"):
            switch_page("Meal_Recommender")
    st.stop()

content_recs = st.session_state['content_recs']
collab_recs = st.session_state['collaborative_recs']
recommendation_type = st.session_state['recommendation_type']
bmi = st.session_state['bmi']

st.set_page_config(page_title="Meal Recommendation Visualizations", layout="wide")
st.title("📊 Visual Insights into Your Personalized Meal Plan")

st.markdown(f"### Your BMI: `{bmi:.1f}` — {recommendation_type}")

# 🔹 NAVIGATION BUTTONS
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    if MULTI_PAGE and st.button("← Home", use_container_width=True):
        switch_page("HomePage")

with nav_col3:
    if MULTI_PAGE and st.button("Recommendations →", use_container_width=True):
        switch_page("Meal_Recommender")

# 🔹 Tabs for Visualizations
tab1, tab2, tab3 = st.tabs(["Calorie Distribution", "Macronutrient Breakdown", "Comparative Analysis"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Calorie Distribution of Your Recommended Meals")

    combined_recs = pd.concat([content_recs, collab_recs])

    fig_calorie_dist = px.bar(
        combined_recs,
        x='item',
        y='calories',
        color='category',
        title='Calories per Meal (Grouped by Category)',
        labels={'item': 'Meal', 'calories': 'Calories'},
        color_discrete_map={'Healthy': '#2ecc71', 'Treat': '#e74c3c'}
    )
    fig_calorie_dist.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_calorie_dist, use_container_width=True)

    # Export Chart as PNG
    st.download_button(
        label="📥 Download Calorie Chart as PNG",
        data=fig_calorie_dist.to_image(format="png"),
        file_name="calorie_distribution.png",
        mime="image/png"
    )

# ---------- TAB 2 ----------
with tab2:
    st.subheader("Macronutrient Composition of Recommended Meals")

    melted_df = content_recs.melt(
        id_vars='item',
        value_vars=['protien', 'carbs', 'totalfat'],
        var_name='Nutrient',
        value_name='Grams'
    )

    fig_macros = px.bar(
        melted_df,
        x='item',
        y='Grams',
        color='Nutrient',
        barmode='group',
        title='Protein, Carbs, and Fat per Meal',
        color_discrete_map={
            'protien': '#3498db',
            'carbs': '#f1c40f',
            'totalfat': '#e67e22'
        }
    )
    fig_macros.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_macros, use_container_width=True)

    # Export Chart as PNG
    st.download_button(
        label="📥 Download Macronutrient Chart as PNG",
        data=fig_macros.to_image(format="png"),
        file_name="macronutrient_breakdown.png",
        mime="image/png"
    )

    # Pie Chart for average macros
    avg_macros = {
        'Protein': content_recs['protien'].mean(),
        'Carbs': content_recs['carbs'].mean(),
        'Fat': content_recs['totalfat'].mean()
    }

    fig_macro_pie = px.pie(
        names=list(avg_macros.keys()),
        values=list(avg_macros.values()),
        title='Average Macronutrient Distribution',
        color_discrete_map={
            'Protein': '#3498db',
            'Carbs': '#f1c40f',
            'Fat': '#e67e22'
        }
    )
    st.plotly_chart(fig_macro_pie, use_container_width=True)

    # Export Pie Chart as PNG
    st.download_button(
        label="📥 Download Macro Pie Chart as PNG",
        data=fig_macro_pie.to_image(format="png"),
        file_name="macronutrient_pie_chart.png",
        mime="image/png"
    )

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Calories vs Protein Comparison")

    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=content_recs['item'],
        y=content_recs['calories'],
        mode='lines+markers',
        name='Calories',
        line=dict(color='#e74c3c', width=2)
    ))

    fig_compare.add_trace(go.Scatter(
        x=content_recs['item'],
        y=content_recs['protien'] * 10,
        mode='lines+markers',
        name='Protein (x10)',
        line=dict(color='#2ecc71', width=2)
    ))

    fig_compare.update_layout(
        title='Calories vs Protein (Scaled)',
        xaxis_title='Meal',
        yaxis_title='Value',
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Export Line Chart as PNG
    st.download_button(
        label="📥 Download Calories vs Protein Chart as PNG",
        data=fig_compare.to_image(format="png"),
        file_name="calories_vs_protein.png",
        mime="image/png"
    )

# ---------- Summary Panel ----------
st.markdown("## 📋 Quick Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric("Avg Calories", f"{content_recs['calories'].mean():.1f} kcal")
    st.metric("Avg Protein", f"{content_recs['protien'].mean():.1f} g")
    st.metric("Avg Carbs", f"{content_recs['carbs'].mean():.1f} g")
    st.metric("Avg Fat", f"{content_recs['totalfat'].mean():.1f} g")

with col2:
    st.info(f"Highest Rated Meal: **{content_recs.loc[content_recs['Ratings'].idxmax()]['item']}**")
    st.info(f"Lowest Calorie Meal: **{content_recs.loc[content_recs['calories'].idxmin()]['item']}**")
    st.info(f"Highest Protein Meal: **{content_recs.loc[content_recs['protien'].idxmax()]['item']}**")

# ---------- Personalized Insights ----------
st.markdown("---")
st.markdown("### 💡 Personalized Insights")

if bmi < 18.5:
    st.warning("You're underweight. Meals are optimized for higher calories and protein for healthy weight gain.")
elif bmi < 25:
    st.success("You're in the normal BMI range. Meals are balanced for maintenance and nutrition.")
else:
    st.error("You're overweight. Meals are optimized for lower calories and higher protein for weight management.")

# ---------- PDF Export Function ----------
def generate_visualizations_pdf(user_name, bmi, content_recs, collab_recs):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{user_name}'s Meal Recommendation Visual Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"BMI: {bmi:.1f}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Summary Metrics", ln=True)

    metrics = [
        f"Avg Calories: {content_recs['calories'].mean():.1f} kcal",
        f"Avg Protein: {content_recs['protien'].mean():.1f} g",
        f"Avg Carbs: {content_recs['carbs'].mean():.1f} g",
        f"Avg Fat: {content_recs['totalfat'].mean():.1f} g",
        f"Highest Rated Meal: {content_recs.loc[content_recs['Ratings'].idxmax()]['item']}",
        f"Lowest Calorie Meal: {content_recs.loc[content_recs['calories'].idxmin()]['item']}",
        f"Highest Protein Meal: {content_recs.loc[content_recs['protien'].idxmax()]['item']}"
    ]

    for metric in metrics:
        pdf.multi_cell(0, 10, txt=f"- {metric}")

    pdf.ln(10)
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Recommended Meals", ln=True)

    for _, row in content_recs.iterrows():
        pdf.multi_cell(0, 10, txt=f"- {row['item']} (Calories: {row['calories']}, Protein: {row['protien']}g)")

    # Save PDF
    file_name = f"{user_name}_visual_report.pdf"
    pdf.output(file_name)
    return file_name

# ---------- PDF Download Button ----------
user_name = st.text_input("Enter your name for the PDF Report")

if st.button("📄 Generate Visualizations PDF Report"):
    pdf_file = generate_visualizations_pdf(user_name or "User", bmi, content_recs, collab_recs)
    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name=pdf_file,
            mime="application/pdf"
        )
    # Optional cleanup: remove PDF after download (uncomment if needed)
    # os.remove(pdf_file)

