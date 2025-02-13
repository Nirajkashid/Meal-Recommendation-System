import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import MinMaxScaler

# Load data
st.header("🍕 Meal Recommendation System 🍔")
uploaded_file = st.file_uploader("Choose a file")

def calculate_calorie_needs(weight, height, age, gender):
    # Calculate BMR using Mifflin-St Jeor Equation without activity multiplier
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    return bmr

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', delimiter=',')

        # Preprocessing for nutrition-based filtering
        df['calories'] = df['calories'].astype(float)
        df['protien'] = df['protien'].astype(float)
        df['totalfat'] = df['totalfat'].astype(float)
        df['carbs'] = df['carbs'].astype(float)
        df['sugar'] = df['sugar'].astype(float)
        df['addedsugar'] = df['addedsugar'].astype(float)

        # Create normalized feature matrix for better comparison
        features_for_scaling = ['calories', 'protien', 'totalfat', 'carbs']
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[features_for_scaling]), 
                               columns=features_for_scaling)

        def get_recommendations(target_nutrients, num_recommendations=10):
            """
            Get meal recommendations based on target nutritional values
            target_nutrients: dict containing target values for calories, protein, fat, and carbs
            """
            # Create target vector
            target_vector = np.array([
                target_nutrients['calories'],
                target_nutrients['protein'],
                target_nutrients['fat'],
                target_nutrients['carbs']
            ]).reshape(1, -1)
            
            # Scale target vector using the same scaler
            target_vector_scaled = scaler.transform(target_vector)
            
            # Calculate similarity scores
            similarities = cosine_similarity(target_vector_scaled, df_scaled)
            
            # Get top recommendations
            top_indices = similarities[0].argsort()[::-1][:num_recommendations]
            return df.iloc[top_indices]

        def main():
            st.sidebar.header("Personal Information")
            
            # Get user information
            weight = st.sidebar.number_input("Weight (kg):", min_value=0.0)
            height = st.sidebar.number_input("Height (cm):", min_value=0.0)
            age = st.sidebar.number_input("Age:", min_value=0, max_value=120)
            gender = st.sidebar.selectbox("Gender:", ["Male", "Female"])

            if weight > 0 and height > 0:
                bmi = weight / ((height / 100) ** 2)
                base_calories = calculate_calorie_needs(weight, height, age, gender)
                
                st.sidebar.subheader("Health Metrics")
                st.sidebar.write(f"Your BMI: {bmi:.2f}")
                st.sidebar.write(f"Base Calorie Needs: {base_calories:.0f}")

                # Display BMI category and nutrition guidelines
                if bmi < 18.5:
                    st.sidebar.warning("BMI Category: Underweight")
                    default_calories = base_calories * 1.2
                    default_protein = 2.0  # g/kg bodyweight
                elif 18.5 <= bmi < 24.9:
                    st.sidebar.success("BMI Category: Normal Weight")
                    default_calories = base_calories
                    default_protein = 1.6
                elif 25 <= bmi < 29.9:
                    st.sidebar.warning("BMI Category: Overweight")
                    default_calories = base_calories * 0.85
                    default_protein = 1.8
                else:
                    st.sidebar.error("BMI Category: Obese")
                    default_calories = base_calories * 0.7
                    default_protein = 2.0

                st.subheader("Customize Your Nutritional Preferences")
                
                # Allow users to customize their nutritional targets
                target_calories = st.slider("Target Calories per Meal:", 
                                         int(default_calories * 0.2), 
                                         int(default_calories * 0.5), 
                                         int(default_calories * 0.33))
                
                target_protein = st.slider("Target Protein (g):", 
                                        10, 
                                        100, 
                                        int(weight * default_protein / 4))
                
                target_fat = st.slider("Target Fat (g):", 
                                     5, 
                                     50, 
                                     int((target_calories * 0.3) / 9))
                
                target_carbs = st.slider("Target Carbs (g):", 
                                       20, 
                                       150, 
                                       int((target_calories * 0.45) / 4))

                if st.button("Get Recommendations"):
                    target_nutrients = {
                        'calories': target_calories,
                        'protein': target_protein,
                        'fat': target_fat,
                        'carbs': target_carbs
                    }
                    
                    recommendations = get_recommendations(target_nutrients)
                    
                    if len(recommendations) > 0:
                        st.write("### Recommended Meals")
                        st.write("Meals are sorted by how well they match your nutritional targets:")
                        
                        for idx, row in recommendations.iterrows():
                            col1, col2 = st.columns([3, 2])
                            with col1:
                                st.write(f"**{row['item']}**")
                                st.write(f"Calories: {row['calories']:.0f} kcal")
                            with col2:
                                st.write(f"Protein: {row['protien']:.1f}g")
                                st.write(f"Fat: {row['totalfat']:.1f}g")
                                st.write(f"Carbs: {row['carbs']:.1f}g")
                            st.divider()
                    else:
                        st.warning("No suitable recommendations found. Try adjusting your preferences.")

            else:
                st.warning("Please enter your weight and height to get personalized recommendations.")

        if __name__ == "__main__":
            main()
    
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload a CSV file to start getting recommendations.")