import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load model
model = joblib.load('student_performance_model.pkl')

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for cards & responsive design
st.markdown("""
<style>
/* Hide Streamlit default buttons */
button[title="Share"], button[title="Star"], button[title="Edit"], a[title="View source on GitHub"] {
    display: none;
}

/* Card style */
.card {
    background-color: #1e1e2f;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
    color: white;
    margin-bottom: 20px;
}

/* Prediction result style */
.result {
    background-color: #0f5132;
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    text-align: center;
    margin-top: 10px;
}

/* Button hover */
.stButton>button:hover {
    background-color: #0d6efd;
    color: white;
    transition: 0.3s;
}

/* Responsive text */
@media (max-width: 768px){
    .card {
        padding: 15px;
    }
    .result {
        font-size: 18px;
        padding: 15px;
    }
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict exam score using academic & lifestyle factors</p>", unsafe_allow_html=True)
st.write("---")

# Layout
left_col, right_col = st.columns([1,1])

# ---------------- LEFT COLUMN (Inputs) ----------------
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## 📊 Student Inputs")

    # Input fields
    study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 2.0)
    attendance = st.slider("🏫 Attendance (%)", 0.0, 100.0, 80.0)
    mental_health = st.slider("🧠 Mental Health Rating (1–10)", 1, 10, 5)
    sleep_hours = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0)
    part_time_job = st.selectbox("💼 Part-Time Job", ['Yes', 'No'])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT COLUMN (Prediction & Result) ----------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    # Buttons side by side
    col1, col2 = st.columns([1, 0.3])
    with col1:
        predict_btn = st.button("🚀 Predict")
    with col2:
        refresh_btn = st.button("🔄 Refresh")

    if predict_btn:
        ptj_encoded = 1 if part_time_job == "Yes" else 0
        input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))

        st.markdown(f"<div class='result'>🎯 Predicted Exam Score<br><b>{prediction:.2f} / 100</b></div>", unsafe_allow_html=True)

    elif refresh_btn:
        
        st.rerun()
    else:
        st.info("👈 Enter details and click **Predict**")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
