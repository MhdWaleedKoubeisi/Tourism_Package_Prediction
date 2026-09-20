import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🧳")

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


model = load_model(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase a travel package")

# In the training data every Designation is always pitched the same product (1-to-1 link),
# so the product is derived from the designation to avoid impossible combinations.
PRODUCT_BY_DESIGNATION = {
    "Executive": "Basic",
    "Manager": "Deluxe",
    "Senior Manager": "Standard",
    "AVP": "Super Deluxe",
    "VP": "King",
}

# Collect user input
col1, col2 = st.columns(2)
with col1:
    Age = st.slider("Age", 18, 70, 35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    Designation = st.selectbox("Designation", list(PRODUCT_BY_DESIGNATION.keys()))
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=22000.0, step=500.0)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
with col2:
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting (under 5)", 0, 3, 1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    NumberOfTrips = st.slider("Number of Trips per Year", 1, 22, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 5, 40, 14)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 4)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    ProductPitched = PRODUCT_BY_DESIGNATION[Designation]
    st.info(f"Product pitched (derived from designation): **{ProductPitched}**")

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold (must match the threshold used in train.py)
classification_threshold = 0.30

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    st.metric("Purchase probability", f"{prob:.1%}")
    if pred == 1:
        st.success("Prediction: Customer **will purchase** the travel package - prioritise for follow-up.")
    else:
        st.warning("Prediction: Customer is **unlikely to purchase** - low priority for follow-up.")
    with st.expander("Model input (dataframe sent to the model)"):
        st.dataframe(input_data)
