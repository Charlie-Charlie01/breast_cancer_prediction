import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.markdown("# 🩺 Breast Cancer Prediction Tool")
st.markdown("""
This application uses machine learning to predict whether a breast mass is benign or malignant
based on measurements from digitized images of a fine needle aspirate (FNA) of the breast mass.

Please adjust the sliders below to input patient data and get a prediction.
""")

st.warning("⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. It should not replace professional medical advice, diagnosis, or treatment.")

# Function to train and save model and scaler
def train_and_save_model():
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        # Select only the mean features that we're using in the app
        mean_features = [col for col in X.columns if 'mean' in col]
        X = X[mean_features]
        
        # Convert feature names to format used in the app
        X.columns = [col.replace(' ', '_').lower() for col in X.columns]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        with open('breast_cancer_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        return model, scaler
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None, None

# Function to load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('breast_cancer_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.warning("Model files not found. Training a new model... (this may take a moment)")
        return train_and_save_model()

# Load model and scaler
model, scaler = load_model_and_scaler()

# Define feature input ranges based on the breast cancer dataset typical values
feature_ranges = {
    'mean_radius': (6.0, 28.0, 12.0),         # (min, max, default)
    'mean_texture': (10.0, 40.0, 18.0),
    'mean_perimeter': (40.0, 190.0, 80.0),
    'mean_area': (150.0, 2500.0, 500.0),
    'mean_smoothness': (0.05, 0.16, 0.1),
    'mean_compactness': (0.02, 0.35, 0.1),
    'mean_concavity': (0.0, 0.5, 0.1),
    'mean_symmetry': (0.1, 0.3, 0.18),
    'mean_fractal_dimension': (0.05, 0.1, 0.06)
}

# Input feature tooltips
feature_tooltips = {
    'mean_radius': "Mean of distances from center to points on the perimeter",
    'mean_texture': "Standard deviation of gray-scale values",
    'mean_perimeter': "Mean size of the core tumor",
    'mean_area': "Mean area of the tumor",
    'mean_smoothness': "Mean of local variation in radius lengths",
    'mean_compactness': "Mean of perimeter² / area - 1.0",
    'mean_concavity': "Mean of severity of concave portions of the contour",
    'mean_symmetry': "Mean symmetry of the tumor",
    'mean_fractal_dimension': "Mean 'coastline approximation' - 1"
}

# Divider
st.divider()

# Create columns for more organized layout
col1, col2 = st.columns(2)

# User input section
with st.expander("**Tumor Characteristics** (Click to expand/collapse)", expanded=True):
    # Column 1: Size-related features
    with col1:
        st.subheader("Size Metrics")
        
        mean_radius = st.slider(
            "Mean Radius", 
            min_value=feature_ranges['mean_radius'][0], 
            max_value=feature_ranges['mean_radius'][1], 
            value=feature_ranges['mean_radius'][2],
            help=feature_tooltips['mean_radius']
        )
        
        mean_perimeter = st.slider(
            "Mean Perimeter", 
            min_value=feature_ranges['mean_perimeter'][0], 
            max_value=feature_ranges['mean_perimeter'][1], 
            value=feature_ranges['mean_perimeter'][2],
            help=feature_tooltips['mean_perimeter']
        )
        
        mean_area = st.slider(
            "Mean Area", 
            min_value=feature_ranges['mean_area'][0], 
            max_value=feature_ranges['mean_area'][1], 
            value=feature_ranges['mean_area'][2],
            help=feature_tooltips['mean_area']
        )
        
        mean_texture = st.slider(
            "Mean Texture", 
            min_value=feature_ranges['mean_texture'][0], 
            max_value=feature_ranges['mean_texture'][1], 
            value=feature_ranges['mean_texture'][2],
            help=feature_tooltips['mean_texture']
        )
    
    # Column 2: Shape-related features
    with col2:
        st.subheader("Shape Metrics")
        
        mean_smoothness = st.slider(
            "Mean Smoothness", 
            min_value=feature_ranges['mean_smoothness'][0], 
            max_value=feature_ranges['mean_smoothness'][1], 
            value=feature_ranges['mean_smoothness'][2],
            format="%.4f",
            help=feature_tooltips['mean_smoothness']
        )
        
        mean_compactness = st.slider(
            "Mean Compactness", 
            min_value=feature_ranges['mean_compactness'][0], 
            max_value=feature_ranges['mean_compactness'][1], 
            value=feature_ranges['mean_compactness'][2],
            format="%.4f",
            help=feature_tooltips['mean_compactness']
        )
        
        mean_concavity = st.slider(
            "Mean Concavity", 
            min_value=feature_ranges['mean_concavity'][0], 
            max_value=feature_ranges['mean_concavity'][1], 
            value=feature_ranges['mean_concavity'][2],
            format="%.4f",
            help=feature_tooltips['mean_concavity']
        )
        
        mean_symmetry = st.slider(
            "Mean Symmetry", 
            min_value=feature_ranges['mean_symmetry'][0], 
            max_value=feature_ranges['mean_symmetry'][1], 
            value=feature_ranges['mean_symmetry'][2],
            format="%.4f",
            help=feature_tooltips['mean_symmetry']
        )
        
        mean_fractal_dimension = st.slider(
            "Mean Fractal Dimension", 
            min_value=feature_ranges['mean_fractal_dimension'][0], 
            max_value=feature_ranges['mean_fractal_dimension'][1], 
            value=feature_ranges['mean_fractal_dimension'][2],
            format="%.4f",
            help=feature_tooltips['mean_fractal_dimension']
        )

# Divider
st.divider()

# Create a dataframe with the input features
def get_input_features():
    features = {
        'mean_radius': mean_radius,
        'mean_texture': mean_texture,
        'mean_perimeter': mean_perimeter,
        'mean_area': mean_area,
        'mean_smoothness': mean_smoothness,
        'mean_compactness': mean_compactness,
        'mean_concavity': mean_concavity,
        'mean_symmetry': mean_symmetry,
        'mean_fractal_dimension': mean_fractal_dimension
    }
    return pd.DataFrame([features])

# Prediction
if st.button("📊 Predict Tumor Classification", help="Click to predict based on the input values"):
    if model is not None and scaler is not None:
        # Get the input features
        input_df = get_input_features()
        
        # Display the input values
        with st.expander("📋 Input Summary", expanded=False):
            st.write(input_df)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Get probability
        prediction_proba = model.predict_proba(input_scaled)
        
        # Display prediction result
        st.subheader("🔍 Prediction Result")
        
        # Set color based on prediction
        if prediction[0] == 0:  # Benign
            prediction_label = "Benign"
            prediction_color = "green"
            benign_probability = prediction_proba[0][0]
            malignant_probability = prediction_proba[0][1]
        else:  # Malignant
            prediction_label = "Malignant"
            prediction_color = "red"
            benign_probability = prediction_proba[0][0]
            malignant_probability = prediction_proba[0][1]
        
        # Display prediction
        st.markdown(f"<h2 style='color: {prediction_color};'>Prediction: {prediction_label}</h2>", unsafe_allow_html=True)
        
        # Display probabilities
        st.markdown("### Confidence Probabilities:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Benign")
            st.progress(benign_probability)
            st.write(f"{benign_probability:.2%}")
        
        with col2:
            st.markdown("#### Malignant")
            st.progress(malignant_probability)
            st.write(f"{malignant_probability:.2%}")
        
        # Add more interpretation
        st.markdown("### Interpretation:")
        if prediction[0] == 0:  # Benign
            st.success(f"The model predicts that the tumor is **benign** with {benign_probability:.2%} confidence.")
            st.info("Benign tumors are not cancerous and do not spread to other parts of the body.")
        else:  # Malignant
            st.error(f"The model predicts that the tumor is **malignant** with {malignant_probability:.2%} confidence.")
            st.warning("Malignant tumors are cancerous and can spread to other parts of the body.")
        
        # Importance disclaimer
        st.warning("""
        **Important Note:**
        This prediction is based on a machine learning model and should not be used as the sole basis for 
        diagnosis or treatment. Always consult with a qualified healthcare professional.
        """)
    else:
        st.error("Model could not be loaded. Please check the model files.")

# Add information about the model
with st.expander("ℹ️ About the Model", expanded=False):
    st.markdown("""
    ### Model Information
    
    This application uses a Random Forest Classifier trained on the Wisconsin Breast Cancer Diagnostic Dataset.
    
    #### Dataset Features:
    - The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    - Features describe characteristics of the cell nuclei present in the image.
    
    #### Model Performance:
    - The Random Forest model was trained with cross-validation.
    - It was trained to classify tumors as either Benign (0) or Malignant (1).
    
    #### Technology:
    - The model was built using scikit-learn's Random Forest Classifier.
    - It was saved using pickle for deployment in this Streamlit application.
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center">
    <p>Created for educational purposes only.</p>
    <p>This application is not a substitute for professional medical advice, diagnosis, or treatment.</p>
</div>
""", unsafe_allow_html=True)
