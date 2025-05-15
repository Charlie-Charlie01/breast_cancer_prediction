import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# App title and description
st.markdown("# 🎗️ Breast Cancer Prediction Model")
st.markdown("""
This app predicts breast cancer diagnosis (Benign/Malignant) using a Random Forest classifier. 
Enter tumor characteristics below and click **Predict**.
""")

# Load model and scaler
try:
    with open('breast_cancer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('breast_cancer_model.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input section with expandable tumor characteristics
with st.expander("🔬 Tumor Characteristics", expanded=True):
    st.markdown("### Tumor Measurements (Mean Values)")
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.slider("Radius (6-28)", 6.0, 28.0, 14.0, 
                        help="Mean distance from center to points on the perimeter")
        texture = st.slider("Texture (9-40)", 9.0, 40.0, 19.0,
                          help="Standard deviation of gray-scale values")
        perimeter = st.slider("Perimeter (40-190)", 40.0, 190.0, 90.0,
                            help="Perimeter measurement of tumor")
        area = st.slider("Area (150-2500)", 150.0, 2500.0, 650.0,
                       help="Area measurement of tumor")
        
    with col2:
        smoothness = st.slider("Smoothness (0.05-0.17)", 0.05, 0.17, 0.1, 0.01,
                             help="Local variation in radius lengths")
        compactness = st.slider("Compactness (0.02-0.35)", 0.02, 0.35, 0.1, 0.01,
                              help="(Perimeter² / (Area - 1.0))")
        concavity = st.slider("Concavity (0.0-0.43)", 0.0, 0.43, 0.1, 0.01,
                            help="Severity of concave portions of the contour")
        symmetry = st.slider("Symmetry (0.1-0.3)", 0.1, 0.3, 0.18, 0.01,
                           help="Symmetry measurement of tumor")
        
    fractal_dimension = st.slider("Fractal Dimension (0.05-0.1)", 0.05, 0.1, 0.06, 0.001,
                                help="Fractal dimension measurement")

# Prediction logic
if st.button("🔍 Predict Diagnosis", type="primary"):
    # Create input DataFrame
    input_data = pd.DataFrame([[radius, texture, perimeter, area, 
                               smoothness, compactness, concavity, 
                               symmetry, fractal_dimension]],
                             columns=['radius', 'texture', 'perimeter', 'area',
                                     'smoothness', 'compactness', 'concavity',
                                     'symmetry', 'fractal_dimension'])
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    proba = model.predict_proba(scaled_data)[0]
    
    # Display results
    st.divider()
    if prediction[0] == 0:
        st.success(f"## Benign (Probability: {proba[0]*100:.2f}%)")
        st.markdown("_Low likelihood of malignancy detected_")
    else:
        st.error(f"## Malignant (Probability: {proba[1]*100:.2f}%)")
        st.markdown("_High likelihood of malignancy detected_")
    
    # Confidence visualization
    st.progress(proba.max(), 
               text=f"Model Confidence: {proba.max()*100:.1f}%")

# Disclaimer
st.divider()
st.markdown("""
<div style='font-size: 0.8em; color: #666;'>
⚠️ <strong>Important Note:</strong> This prediction is for informational/educational purposes only. 
Always consult medical professionals for clinical diagnosis.
</div>
""", unsafe_allow_html=True)

# Optional: Show scaled input values
with st.expander("⚙️ Technical Details"):
    st.write("Scaled Input Values:", scaled_data)
    st.write("Model Classifier:", model.__class__.__name__)
    st.write("Feature Importance:", dict(zip(input_data.columns, model.feature_importances_)))
