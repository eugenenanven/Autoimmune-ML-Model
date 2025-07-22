import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# App Config
st.set_page_config(page_title="Autoimmune Disease Predictor", layout="wide")
st.title("🧠 Autoimmune Disease Prediction")
st.markdown("Welcome to the clinical-grade **Autoimmune Disease Prediction Tool**. Please enter lab values and symptoms below to assess potential autoimmune conditions.")

st.divider()

# Feature List (MUST match training)
features = ['Age', 'Gender', 'Sickness_Duration_Months', 'RBC_Count', 'Hemoglobin',
            'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 'Reticulocyte_Count', 'WBC_Count',
            'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils',
            'PLT_Count', 'MPV', 'ANA', 'Esbach', 'MBL_Level', 'ESR', 'C3', 'C4', 'CRP',
            'Anti-dsDNA', 'Anti-Sm', 'Rheumatoid factor', 'ACPA', 'Anti-TPO', 'Anti-Tg',
            'Anti-SMA', 'Low-grade fever', 'Fatigue or chronic tiredness', 'Dizziness',
            'Weight loss', 'Rashes and skin lesions', 'Stiffness in the joints',
            'Brittle hair or hair loss', 'Dry eyes and/or mouth', "General 'unwell' feeling",
            'Joint pain', 'Anti_dsDNA', 'Anti_enterocyte_antibodies', 'anti_LKM1',
            'Anti_RNP', 'ASCA', 'Anti_Ro_SSA', 'Anti_CBir1', 'Anti_BP230', 'Anti_tTG',
            'DGP', 'Anti_BP180', 'ASMA', 'Anti_IF', 'IgG_IgE_receptor', 'Anti_SRP',
            'Anti_desmoglein_3', 'Anti_La_SSB', 'Anti_Jo1', 'ANCA', 'anti_centromere',
            'Anti_desmoglein_1', 'EMA', 'Anti_type_VII_collagen', 'C1_inhibitor',
            'Anti_TIF1', 'Anti_epidermal_basement_membrane_IgA', 'Anti_OmpC', 'pANCA',
            'Anti_tissue_transglutaminase', 'anti_Scl_70', 'Anti_Mi2', 'Anti_parietal_cell',
            'Progesterone_antibodies', 'Anti_Sm']

# Split long list into chunks for display
def chunk_features(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

input_data = {}

# Input Form Layout
st.markdown("### 📥 Enter Patient Lab & Symptom Data:")
with st.form(key='input_form'):
    for chunk in chunk_features(features, 3):  # 3 columns per row
        cols = st.columns(3)
        for i, feature in enumerate(chunk):
            input_data[feature] = cols[i].number_input(f"{feature}", step=0.01, format="%.2f")
    submitted = st.form_submit_button("🧪 Predict Diagnosis")

# Prediction Logic
if submitted:
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = np.max(probabilities) * 100

        st.success(f"✅ **Predicted Diagnosis:** `{prediction}`")
        st.progress(confidence / 100)
        st.info(f"📊 **Confidence Level:** `{confidence:.2f}%`")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
