import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Wellness Tourism Package Predictor")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
    duration = st.number_input("Duration of Pitch", min_value=1.0, max_value=150.0, value=15.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    followups = st.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    stars = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    trips = st.number_input("Number of Trips", min_value=0.0, max_value=30.0, value=3.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    children = st.number_input("Number of Children Visiting", min_value=0.0, max_value=10.0, value=1.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", min_value=0.0, max_value=200000.0, value=23000.0)

    submitted = st.form_submit_button("Predict Purchase")

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": stars,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income
    }])
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    if prediction:
        st.success(f"Likely to purchase — probability: {probability:.1%}")
    else:
        st.info(f"Unlikely to purchase — probability: {probability:.1%}")
