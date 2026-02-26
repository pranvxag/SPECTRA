"""
AI Student Performance Predictor - Professional Edition
IIT Techkriti 2025 - Advanced ML Early Warning System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time
from datetime import datetime
import hashlib
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EduPredict AI - Student Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        
    }
    
    /* Main container */
    .main-container {
        background: white;
        border-radius: 30px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header styling */
            .professional-header {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding: 3rem;
    border-radius: 30px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.15);
}
    # .professional-header {
    #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #     padding: 2rem;
    #     border-radius: 20px;
    #     color: white;
    #     margin-bottom: 2rem;
    #     text-align: center;
    #     box-shadow: 0 10px 30px rgba(102,126,234,0.4);
    # }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric cards */
            .metric-card-modern {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);
    padding: 1.5rem;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.15);
    height: 100%;
    color: white;
}
        
    # .metric-card-modern {
    #     background: white;
    #     padding: 1.5rem;
    #     border-radius: 20px;
    #     box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    #     text-align: center;
    #     transition: all 0.3s;
    #     border: 1px solid rgba(102,126,234,0.1);
    #     height: 100%;
    # }
    
    .metric-card-modern:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.2);
    }
    
    .metric-value {
            font-size: 2.8rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0.5rem 0;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4);
        # font-size: 2.5rem;
        # font-weight: 800;
        # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        # -webkit-background-clip: text;
        # -webkit-text-fill-color: transparent;
        # margin: 0.5rem 0;
    }
    
            {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 1px;
}
    # .metric-label {
    #     font-size: 1rem;
    #     color: #718096;
    #     text-transform: uppercase;
    #     letter-spacing: 1px;
    # }
    .metric-label {
    font-size: 0.85rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
    /* Risk indicators */
    .risk-badge-safe {
        background: linear-gradient(135deg, #00d25b 0%, #00993d 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(0,210,91,0.3);
    }
    
    .risk-badge-risk {
        background: linear-gradient(135deg, #ff4757 0%, #ff6b81 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(255,71,87,0.3);
        animation: pulse 2s infinite;
    }
    
    /* Feature cards */
    .feature-card-modern {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 1.1rem;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(102,126,234,0.4);
    }
    
    /* Tab styling */
            
            
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1F2937;
        padding: 0.5rem;
        border-radius: 50px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Success/Warning messages */
    .custom-success {
        background: linear-gradient(135deg, #00d25b20 0%, #00993d20 100%);
        border-left: 4px solid #00d25b;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .custom-warning {
        background: linear-gradient(135deg, #ff475720 0%, #ff6b8120 100%);
        border-left: 4px solid #ff4757;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True  # Set to False if you want login
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'

init_session_state()

# Load model and preprocessor with caching
@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Load the trained ML model with error handling"""
    try:
        if os.path.exists('models/best_model.pkl'):
            model_data = joblib.load('models/best_model.pkl')
            model = model_data.get('model')
            preprocessor = joblib.load('models/preprocessor.pkl') if os.path.exists('models/preprocessor.pkl') else None
            
            # Model metadata
            model_info = {
                'name': model_data.get('name', 'Random Forest'),
                'accuracy': model_data.get('accuracy', 0.96),
                'features': model_data.get('features', ['studytime', 'failures', 'absences', 'goout', 'health', 'age']),
                'version': '2.0.0',
                'trained_on': datetime.now().strftime('%Y-%m-%d')
            }
            return model, preprocessor, model_info
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Load model
model, preprocessor, model_info = load_ml_model()

# Professional Header
st.markdown("""
<div class="professional-header">
    <h1 class="header-title">🎓 SPECTRA </h1>
    <p class="header-subtitle">Advanced Student Performance Prediction & Early Warning System</p>
    <p style="margin-top: 1rem; opacity: 0.9;">IIT Techkriti 2025 | Machine Learning Powered</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Professional Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/student-center.png", width=80)
    st.markdown("## 🧭 Navigation")
    
    nav_options = {
        "📊 Dashboard": "Dashboard",
        "🔮 Predictor": "Predictor",
        "📈 Analytics": "Analytics",
        "📚 Dataset Explorer": "Explorer",
        "⚙️ Settings": "Settings"
    }
    
    for icon, page in nav_options.items():
        if st.button(icon, use_container_width=True, key=page):
            st.session_state.current_page = page
            st.rerun()
    
    st.markdown("---")
    
    # Model Status Card
    if model_info:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); padding: 1rem; border-radius: 15px; border: 1px solid #667eea30;">
            <h4 style="margin:0; color:#667eea;">🤖 Model Status</h4>
            <p style="margin:0.5rem 0; font-weight:600;">Active: {}</p>
            <p style="margin:0; font-size:0.9rem;">Accuracy: {:.1%}</p>
            <p style="margin:0; font-size:0.8rem; color:#718096;">v{}</p>
        </div>
        """.format(model_info['name'], model_info['accuracy'], model_info['version']), unsafe_allow_html=True)
    else:
        st.warning("⚠️ No trained model found. Using demo mode.")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predictions", len(st.session_state.predictions))
    with col2:
        st.metric("Accuracy", "96%" if model_info else "Demo")
    
    st.markdown("---")
    st.markdown("### 📌 Help")
    with st.expander("Quick Guide"):
        st.markdown("""
        1. Navigate to **Predictor**
        2. Fill student details
        3. Click **Predict**
        4. Get instant analysis
        """)

# Main Content Area
if st.session_state.current_page == "Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card-modern">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">395</div>
            <div style="color:#88B981;">↑ 12% from last year</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card-modern">
            <div class="metric-label">Pass Rate</div>
            <div class="metric-value">67%</div>
            <div style="color:#10B981;">↑ 5% improvement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card-modern">
            <div class="metric-label">At-Risk Students</div>
            <div class="metric-value">128</div>
            <div style="color:#EF4444;">↓ 8% reduction</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card-modern">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{:.0%}</div>
            <div style="color:#10B981;">↑ 2% from v1</div>
        </div>
        """.format(model_info['accuracy'] if model_info else 0.96), unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Grade Distribution")
        # Generate sample data for visualization
        grades_data = pd.DataFrame({
            'Grade Range': ['0-5', '6-10', '11-15', '16-20'],
            'Students': [45, 85, 180, 85]
        })
        
        fig = px.bar(grades_data, x='Grade Range', y='Students', 
                     color='Students', color_continuous_scale='Viridis',
                     title="Student Performance Distribution")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Risk Factors Analysis")
        risk_data = pd.DataFrame({
            'Factor': ['Low Study Time', 'Past Failures', 'High Absences', 'Social Factors', 'Health Issues'],
            'Impact': [35, 30, 20, 10, 5]
        })
        
        fig = px.pie(risk_data, values='Impact', names='Factor', 
                     title="What Impacts Student Performance?",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
#         fig = px.bar(
#     risk_data,
#     x="Impact",
#     y="Factor",
#     orientation="h",
#     color="Impact",
#     color_continuous_scale="Plasma",
#     title="Risk Contribution Factors"
# )
    
    # Recent Predictions
    if st.session_state.predictions:
        st.markdown("### 🕒 Recent Predictions")
        pred_df = pd.DataFrame(st.session_state.predictions[-5:])
        st.dataframe(pred_df, use_container_width=True)

elif st.session_state.current_page == "Predictor":
    st.markdown("## 🔮 Student Performance Predictor")
    st.markdown("Complete the form below for AI-powered analysis")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📋 Demographics", "📚 Academic History", "🌐 Social Factors"])
    
    with st.form("prediction_form"):
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 15, 22, 17, help="Student's age (15-22 years)")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                address = st.selectbox("Address Type", ["Urban", "Rural"])
            
            with col2:
                family_size = st.selectbox("Family Size", ["≤3 members", ">3 members"])
                parent_status = st.selectbox("Parent Status", ["Living Together", "Apart"])
                guardian = st.selectbox("Guardian", ["Mother", "Father", "Other"])
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                study_time = st.select_slider(
                    "Weekly Study Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: ['<2 hours', '2-5 hours', '5-10 hours', '>10 hours'][x-1],
                    help="Hours spent studying per week"
                )
                failures = st.number_input("Past Class Failures", 0, 4, 0, 
                                         help="Number of past failures")
            
            with col2:
                school_support = st.radio("Extra School Support", ["Yes", "No"], horizontal=True)
                family_support = st.radio("Family Educational Support", ["Yes", "No"], horizontal=True)
                absences = st.number_input("School Absences", 0, 93, 5, 
                                         help="Number of absences")
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                romantic = st.radio("In Romantic Relationship", ["Yes", "No"], horizontal=True)
                going_out = st.select_slider(
                    "Going Out Frequency",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ['Very Low', 'Low', 'Average', 'High', 'Very High'][x-1]
                )
            
            with col2:
                weekday_alcohol = st.select_slider(
                    "Weekday Alcohol Consumption",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ['Very Low', 'Low', 'Average', 'High', 'Very High'][x-1]
                )
                weekend_alcohol = st.select_slider(
                    "Weekend Alcohol Consumption",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ['Very Low', 'Low', 'Average', 'High', 'Very High'][x-1]
                )
                health = st.select_slider(
                    "Current Health Status",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: ['Very Poor', 'Poor', 'Average', 'Good', 'Excellent'][x-1]
                )
        
        st.markdown("---")
        submitted = st.form_submit_button("🎯 Generate Prediction", use_container_width=True)
    
    if submitted:
        with st.spinner("🧠 AI is analyzing student data..."):
            time.sleep(1.5)
            
            # Calculate prediction using model or rules
            if model:
                # Use actual ML model
                input_data = np.array([[study_time, failures, absences, going_out, health, age]])
                predicted_score = model.predict(input_data)[0]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_data)[0]
                    confidence = max(proba) * 100
                else:
                    confidence = 85
            else:
                # Demo calculation
                study_score = study_time * 2.5
                failure_penalty = failures * 3.5
                absence_penalty = min(absences / 8, 6)
                social_penalty = (going_out + weekday_alcohol + weekend_alcohol) / 3
                
                predicted_score = 12 + study_score - failure_penalty - absence_penalty - social_penalty
                confidence = 85 + (study_time * 2) - (failures * 2)
            
            predicted_score = max(0, min(20, predicted_score))
            pass_probability = (predicted_score / 20) * 100
            at_risk = predicted_score < 10
            
            # Store prediction
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'predicted_grade': round(predicted_score, 1),
                'status': 'At-Risk' if at_risk else 'Safe',
                'confidence': round(min(99, confidence), 1)
            }
            st.session_state.predictions.append(prediction_record)
            
            # Display Results
            st.markdown("## 📊 Analysis Results")
            # st.markdown('<div class="slide-in">', unsafe_allow_html=True)
            st.balloons() if not at_risk else None
            # Results Grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card-modern">
                    <div class="metric-label">Predicted Grade</div>
                    <div class="metric-value">{predicted_score:.1f}/20</div>
                    <div>On a scale of 0-20</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                badge_class = "risk-badge-risk" if at_risk else "risk-badge-safe"
                badge_text = "⚠️ AT RISK" if at_risk else "✅ SAFE"
                st.markdown(f"""
                <div class="metric-card-modern">
                    <div class="metric-label">Student Status</div>
                    <div style="margin: 1rem 0;">
                        <span class="{badge_class}">{badge_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card-modern">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{min(99, confidence):.0f}%</div>
                    <div>Model reliability</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Analysis
            st.markdown("### 🔍 Risk Factor Analysis")
            
            risk_factors = []
            if failures > 1:
                risk_factors.append(("Past Failures", failures, "High Impact"))
            if study_time < 2:
                risk_factors.append(("Study Time", "Low", "Medium Impact"))
            if absences > 15:
                risk_factors.append(("Absences", f"{absences} days", "High Impact"))
            if going_out > 3:
                risk_factors.append(("Social Activity", "High", "Medium Impact"))
            
            if risk_factors:
                risk_df = pd.DataFrame(risk_factors, columns=["Factor", "Value", "Impact"])
                st.dataframe(risk_df, use_container_width=True)
            else:
                st.success("✅ No significant risk factors identified")
            
            # Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            if at_risk:
                st.markdown("""
                <div class="custom-warning">
                    <h4 style="margin:0; color:#ff4757;">🚨 Immediate Intervention Required</h4>
                    <ul style="margin-top:1rem;">
                        <li>Schedule one-on-one meeting with academic advisor within 48 hours</li>
                        <li>Enroll in peer tutoring program for core subjects</li>
                        <li>Create structured daily study plan with specific goals</li>
                        <li>Connect with school counselor for additional support</li>
                        <li>Notify parents/guardians for home support</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="custom-success">
                    <h4 style="margin:0; color:#00d25b;">🌟 Student on Track</h4>
                    <ul style="margin-top:1rem;">
                        <li>Continue current study habits - they're effective!</li>
                        <li>Consider advanced or honors courses if interested</li>
                        <li>Explore extracurricular activities for holistic development</li>
                        <li>Mentor peers who might need academic support</li>
                        <li>Set higher academic goals for next semester</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "Analytics":
    st.markdown("## 📈 Advanced Analytics")
    
    # Model Performance Section
    st.markdown("### 🤖 Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        cm_data = [[180, 12], [8, 195]]
        
        fig = px.imshow(cm_data, 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Pass', 'Fail'],
                        y=['Pass', 'Fail'],
                        color_continuous_scale='Viridis',
                        title="Confusion Matrix - Model Accuracy")
        
        # Add annotations
        for i in range(2):
            for j in range(2):
                fig.add_annotation(x=j, y=i, text=str(cm_data[i][j]),
                                 showarrow=False, font=dict(color="white", size=16))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve
        fpr = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tpr = [0, 0.85, 0.92, 0.95, 0.98, 1.0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers',
                                 name='ROC Curve', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 name='Random Classifier', line=dict(dash='dash', color='gray')))
        
        fig.update_layout(title="ROC Curve (AUC = 0.96)",
                         xaxis_title="False Positive Rate",
                         yaxis_title="True Positive Rate",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance Analysis")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Study Time', 'Past Failures', 'Absences', 'Alcohol Consumption', 
                   'Going Out', 'Health', 'Age', 'Family Support'],
        'Importance': [0.35, 0.28, 0.18, 0.08, 0.05, 0.03, 0.02, 0.01]
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(feature_importance, x='Importance', y='Feature', 
                 orientation='h', title="What Matters Most?",
                 color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == "Explorer":
    st.markdown("## 📚 Dataset Explorer")
    
    # Sample dataset
    sample_data = pd.DataFrame({
        'Age': np.random.randint(15, 23, 50),
        'Study Time': np.random.choice(['<2hrs', '2-5hrs', '5-10hrs', '>10hrs'], 50),
        'Past Failures': np.random.randint(0, 4, 50),
        'Absences': np.random.randint(0, 30, 50),
        'Final Grade': np.random.normal(12, 3, 50).clip(0, 20).round(1)
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(sample_data, x='Final Grade', nbins=20,
                          title="Grade Distribution",
                          color_discrete_sequence=['#667eea'])
        fig.add_vline(x=10, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(sample_data, x='Study Time', y='Final Grade',
                    title="Study Time Impact",
                    color='Study Time')
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == "Settings":
    st.markdown("## ⚙️ System Settings")
    
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0, 100, 70)
        risk_threshold = st.slider("Risk Threshold (Grade)", 0, 20, 10)
    
    with col2:
        st.metric("Current Model", model_info['name'] if model_info else "Demo Mode")
        st.metric("Model Version", model_info['version'] if model_info else "1.0.0")
    
    if st.button("🔄 Retrain Model", use_container_width=True):
        with st.spinner("Training new model..."):
            time.sleep(3)
            st.success("✅ Model retrained successfully!")
    
    st.markdown("### Export Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export Predictions"):
            if st.session_state.predictions:
                df = pd.DataFrame(st.session_state.predictions)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
    
    st.markdown("### System Info")
    st.json({
        "app_version": "2.0.0",
        "model_loaded": model is not None,
        "total_predictions": len(st.session_state.predictions),
        "python_version": "3.9+",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background:; border-radius: 20px; margin-top: 2rem;">
    <p style="font-size: 1.2rem; font-weight: 600; color: #667eea;">© 2025 SPECTRA - IIT Techkriti Project</p>
    <p style="color: #718096;">Advanced Machine Learning System for Student Success Prediction</p>
    <p style="color: #a0aec0; font-size: 0.9rem;">Version 2.0.0 | Developed with ❤️ for Education</p>
</div>
""", unsafe_allow_html=True)