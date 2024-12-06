import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models
model_ctype_catl = joblib.load("random_forest_model_ctype_catl_98%.pkl")
model_ctypel = joblib.load("random_forest_model_ctypel_95_3%.pkl")
model_comp_gap_category = joblib.load("random_forest_model_comp_gap_category_97%.pkl")


# Mappings for predictions
ctypel_mapping = {
    0: "Acute / Chronic Respiratory Failure", 1: "Allergic Reaction", 2: "Atelectasis", 3: "Bleeding",
    4: "Blood Loss Requiring a Transfusion", 5: "Bronchopulmonary Fistula", 6: "Bronchospasm", 7: "Cardiac Arrest",
    8: "Cardiac Arrhythmia", 9: "Cerebral vascular accident (CVA) / Stroke", 10: "Congestive Heart Failure (CHF)",
    11: "Deep Venous Thrombosis (DVT)", 12: "Fever Requiring Antibiotics", 13: "Hemoptysis", 14: "Hemothorax",
    15: "Hospitalization", 16: "Hypokalemia", 17: "Hypotension / Vasovagal Reaction", 18: "Infection",
    19: "Infection and fever", 20: "Myocardial Infarction", 21: "Other Specify", 22: "Pain",
    23: "Pain Requiring Referral to an Anesthesiologist / Pain Specialist", 24: "Pneumothorax",
    25: "Pulmonary Embolus / Emboli", 26: "Respiratory Arrest", 27: "Rib Fracture(s)", 28: "Urinary",
    29: "Vocal Cord Immobility / Paralysis", 30: "Wound Dehiscence"
}

ctype_catl_mapping = {0: "Intermediate", 1: "Major", 2: "Minor"}
comp_gap_category_mapping = {0: "pre treatment", 1: "during treatment", 2: "post treatment"}


# Define user input fields
def user_input_form():
    st.sidebar.header("Enter Patient Data")
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    agelevel = st.sidebar.selectbox("Age Level", ["<= 59", "60-64", "65-69", ">= 70"])
    smoked_f = st.sidebar.selectbox("Smoked", ["No", "Yes"])
    cig_stat = st.sidebar.selectbox("Cigarette Status", ["Never Smoked", "Current Smoker", "Former Smoker"])
    cig_years = st.sidebar.number_input("Cigarette Years", min_value=0.0, value=0.0)
    pack_years = st.sidebar.number_input("Pack Years", min_value=0.0, value=0.0)
    num_cancl = st.sidebar.selectbox("Number of Primary Lung Cancers", ["No Primary Lung Cancer", "One Primary Lung Cancer", "Two Primary Lung Cancers"])
    intstatl_cat = st.sidebar.selectbox("Interval Status Category", ["No Cancer", "Control with Cancer", "Never Screened", "Post-Screening", "Interval", "Screen Dx"])
    lung_stage = st.sidebar.selectbox("Lung Stage", ["Stage I", "Stage IA", "Stage IB", "Stage II", "Stage IIA", "Stage IIB", "Stage IIIA", "Stage IIIB", "Stage IV", "Small Cell"])
    lung_stage_t = st.sidebar.selectbox("Lung Stage T", ["T1", "T2", "T3", "T4", "TX"])
    lung_stage_n = st.sidebar.selectbox("Lung Stage N", ["N0", "N1", "N2", "N3", "NX"])
    lung_stage_m = st.sidebar.selectbox("Lung Stage M", ["M0", "M1", "MX"])
    lung_cancer_type = st.sidebar.selectbox("Lung Cancer Type", ["Non-Small Cell Lung Cancer", "Small Cell Lung Cancer"])
    lung_grade = st.sidebar.selectbox("Lung Grade", [
        "Well differentiated; Grade I", "Moderately differentiated; Grade II",
        "Poorly differentiated; Grade III", "Undifferentiated; Grade IV", "T cell; T precursor", "Unknown"
    ])
    lung_histtype = st.sidebar.selectbox("Lung Histological Type", [
        "Squamous Cell Carcinoma", "Spindle Cell Carcinoma", "Small Cell Carcinoma", 
        "Intermediate Cell Carcinoma", "Adenocarcinoma", "Large Cell Carcinoma", "Other NSC carcinoma" ,"Carcinoma, NOS", "Other/Missing"
    ])
    lung_cancer_first = st.sidebar.selectbox("First Lung Cancer Diagnosis", ["No", "Yes"])
    lung_annyr = st.sidebar.number_input("Lung Annual Years", min_value=0.0, value=0.0)
    curative_pneuml = st.sidebar.selectbox("Curative Pneumonectomy", ["No", "Yes"])
    curative_wsll = st.sidebar.selectbox("Curative Wedge/Segmental Lobectomy", ["No", "Yes"])
    curative_chemol = st.sidebar.selectbox("Curative Chemotherapy", ["No", "Yes"])
    curative_radl = st.sidebar.selectbox("Curative Radiotherapy", ["No", "Yes"])
    neoadjuvantl = st.sidebar.selectbox("Neoadjuvant Treatment", ["No", "Yes"])
    trt_numl = st.sidebar.selectbox("Treatment Number", [
        "External photon beam", "Hyperfractionated", "Other radiation (specify)", "Unknown radiation", 
        "Exploratory thoracotomy w/o resection", "Mediansternotomy", "Mediastinoscopy", 
        "Lobectomy", "Lobectomy & resection of chest wall", "Bilobectomy", "Pneumonectomy", 
        "Pneumonectomy & resection of chest wall", "Wedge resection", "Segmental resection", 
        "Lymphadenectomy / lymph node sampling", "Chest wall resection", "Thoracentesis", 
        "Partial pleurectomy", "Exploratory laparoscopy/thoracoscopy without resection", 
        "Radio frequency ablation", "Cyber knife and gamma knife", "Other surgery (specify)", 
        "Cisplatin", "Doxorubicin", "Cyclophosphamide", "Methotrexate", "Vincristine", 
        "Etoposide (VP-16)", "CAV (cyclophosphamide/doxorubicin/vincristine)", 
        "VPP (or EP) (etoposide/cisplatin)", "Other chemotherapy (specify)", 
        "Laser therapy", "Other (specify)", "Radiation treatment, NOS", "Systemic treatment, NOS", 
        "Other treatment, NOS"
    ])
    trt_familyl = st.sidebar.selectbox("Treatment Family", [
        "Pneumonectomy or bilobectomy", "Wedge resection, segmental resection, or lobectomy", 
        "Radiation treatment", "Chemotherapy", "Non-curative treatment", "Pending"
    ])
    hyperten_f = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
    hearta_f = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
    stroke_f = st.sidebar.selectbox("Stroke", ["No", "Yes"])
    diabetes_f = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    # Add more fields as needed...
    
    # Collect data in a dictionary
    data = {
        "sex": sex,
        "agelevel": agelevel,
        "smoked_f": smoked_f,
        "cig_stat": cig_stat,
        "cig_years": cig_years,
        "pack_years": pack_years,
        "num_cancl": num_cancl,
        "intstatl_cat": intstatl_cat,
        "lung_stage": lung_stage,
        "lung_stage_t": lung_stage_t,
        "lung_stage_n": lung_stage_n,
        "lung_stage_m": lung_stage_m,
        "lung_cancer_type": lung_cancer_type,
        "lung_grade": lung_grade,
        "lung_histtype": lung_histtype,
        "lung_cancer_first": lung_cancer_first,
        "lung_annyr": lung_annyr,
        "curative_pneuml": curative_pneuml,
        "curative_wsll": curative_wsll,
        "curative_chemol": curative_chemol,
        "curative_radl": curative_radl,
        "neoadjuvantl": neoadjuvantl,
        "trt_numl": trt_numl,
        "trt_familyl": trt_familyl,
        "hyperten_f": hyperten_f,
        "hearta_f": hearta_f,
        "stroke_f": stroke_f,
        "diabetes_f": diabetes_f,
        # Add more features...
    }
    return pd.DataFrame([data])

# Main Streamlit app
st.title("Lung Cancer complication Predictor")
st.write("Predicts `ctype_catl`, `ctypel`, and `comp_gap_category` based on patient data.")

# Sidebar for user inputs
input_df = user_input_form()

# Preprocess inputs
def preprocess_input(input_df, model):
    expected_columns = model.feature_names_in_
    input_encoded = pd.get_dummies(input_df)
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    return input_encoded[expected_columns]

# Make predictions
if st.button("Predict"):
    try:
        # Check if Cigarette Years, Pack Years, and Lung Annual Years are all zero
        if input_df["cig_years"].iloc[0] == 0 and input_df["pack_years"].iloc[0] == 0 and input_df["lung_annyr"].iloc[0] == 0:
            st.warning("Not enough information to make predictions. Please provide additional details.")
        else:
            # Predict for each model
            input_processed_catl = preprocess_input(input_df, model_ctype_catl)
            input_processed_ctypel = preprocess_input(input_df, model_ctypel)
            input_processed_gap = preprocess_input(input_df, model_comp_gap_category)

            pred_catl = model_ctype_catl.predict(input_processed_catl)
            pred_ctypel = model_ctypel.predict(input_processed_ctypel)
            pred_gap = model_comp_gap_category.predict(input_processed_gap)

            # Get probabilities for ctypel to find top 3 classes
            pred_ctypel_proba = model_ctypel.predict_proba(input_processed_ctypel)[0]
            top_3_indices = np.argsort(pred_ctypel_proba)[-5:][::-1]

            # Map indices to class names and get probabilities
            top_3_classes_with_names = [(ctypel_mapping[idx], pred_ctypel_proba[idx]) for idx in top_3_indices]

            # Display predictions
            st.subheader("Predictions:")
            st.write(f"**Complication severity:** {ctype_catl_mapping[pred_catl[0]]}")
            st.write(f"**Complication name:** {ctypel_mapping[pred_ctypel[0]]}")
            st.write(f"**Complication Gap:** {comp_gap_category_mapping[pred_gap[0]]}")

            # Display top 3 ctypel predictions
            st.subheader("Top 3 Predictions for ctypel:")
            for i, (class_name, probability) in enumerate(top_3_classes_with_names, start=1):
                st.write(f"{i}. {class_name}: {probability * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Batch predictions via file upload
st.subheader("Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data.head())

    # Preprocess and predict for batch
    try:
        batch_processed_catl = preprocess_input(data, model_ctype_catl)
        batch_processed_ctypel = preprocess_input(data, model_ctypel)
        batch_processed_gap = preprocess_input(data, model_comp_gap_category)

        batch_pred_catl = model_ctype_catl.predict(batch_processed_catl)
        batch_pred_ctypel = model_ctypel.predict(batch_processed_ctypel)
        batch_pred_gap = model_comp_gap_category.predict(batch_processed_gap)

        # Add predictions to the dataframe
        data["ctype_catl"] = batch_pred_catl
        data["ctypel"] = batch_pred_ctypel
        data["comp_gap_category"] = batch_pred_gap

        st.subheader("Predictions for Uploaded Data:")
        st.write(data)
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
