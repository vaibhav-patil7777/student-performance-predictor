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

# Custom CSS for cards
st.markdown("""
<style>
/* Hide Share button */
button[title="Share"] {
    display: none;
}

/* Hide Star button */
button[title="Star"] {
    display: none;
}

/* Hide Edit button */
button[title="Edit"] {
    display: none;
}

/* Hide GitHub button */
a[title="View source on GitHub"] {
    display: none;
}
.card {
    background-color: #1e1e2f;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}
.result {
    background-color: #0f5132;
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict exam score using academic & lifestyle factors</p>", unsafe_allow_html=True)

st.write("---")

# Layout
left_col, right_col = st.columns(2)

# ---------------- LEFT COLUMN (Inputs) ----------------
with left_col:
    st.markdown("<div class='card'> ", unsafe_allow_html=True)

    st.markdown("## 📊 Student Inputs")

    study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 2.0)
    attendance = st.slider("🏫 Attendance (%)", 0.0, 100.0, 80.0)
    mental_health = st.slider("🧠 Mental Health Rating (1–10)", 1, 10, 5)
    sleep_hours = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0)
    part_time_job = st.selectbox("💼 Part-Time Job", ['Yes', 'No'])

    predict_btn = st.button("🚀 Predict Exam Score")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT COLUMN (Result) ----------------
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:
        ptj_encoded = 1 if part_time_job == "Yes" else 0

        input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        st.markdown(
            f"<div class='result'>🎯 Predicted Exam Score<br><b>{prediction:.2f} / 100</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.info("👈 Enter details and click **Predict Exam Score**")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
