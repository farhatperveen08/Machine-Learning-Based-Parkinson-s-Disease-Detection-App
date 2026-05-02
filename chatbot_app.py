
 #file name app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Parkinson Disease Detection",
    page_icon="🧠",
    layout="wide"
)

# Feature Names
feature_names = [
"MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)",
"MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP",
"MDVP:PPQ","Jitter:DDP","MDVP:Shimmer",
"MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5",
"MDVP:APQ","Shimmer:DDA","NHR","HNR",
"RPDE","DFA","spread1","spread2","D2","PPE"
]

# Demo Samples
parkinson_sample = [
119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,
0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,
0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654
]

healthy_sample = [
197.076,206.896,192.055,0.00289,0.00001,0.00166,0.00168,0.00498,
0.01098,0.097,0.00563,0.0068,0.00802,0.01689,0.00339,26.775,
0.422229,0.741367,-7.3483,0.177551,1.743867,0.085569
]

# Header
st.markdown("""
<h1 style='text-align:center; color:#6C63FF;'>🧠 Parkinson Disease Detection System</h1>
<h4 style='text-align:center;'>AI Based Voice Analysis Healthcare Dashboard</h4>
""", unsafe_allow_html=True)

# Load Model Files
model = joblib.load("parkinson_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_means = joblib.load("feature_means.pkl")

st.markdown("<h5 style='text-align:center;'>Voice-based Machine Learning Detection System</h5>", unsafe_allow_html=True)

st.divider()

# Sidebar
st.sidebar.title("Patient Voice Features")
st.sidebar.markdown("### Demo Samples")

# Buttons with unique keys ✅
if st.sidebar.button("Load Healthy Sample", key="healthy_btn"):
    st.session_state["demo_data"] = healthy_sample

if st.sidebar.button("Load Parkinson Sample", key="parkinson_btn"):
    st.session_state["demo_data"] = parkinson_sample

if st.sidebar.button("Reset Inputs", key="reset_btn"):
    if "demo_data" in st.session_state:
        del st.session_state["demo_data"]

# Input Fields
features = []

for i, name in enumerate(feature_names):
    if "demo_data" in st.session_state:
        default_value = st.session_state["demo_data"][i]
    else:
        default_value = float(feature_means[name])

    val = st.sidebar.number_input(name, value=float(default_value))
    features.append(val)

# Prediction
if st.sidebar.button("Predict Now", key="predict_btn"):

    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    prob = probability[0][1]

    st.write("Input Data:", input_data)
    st.write("Scaled Data:", input_scaled)

    st.divider()
    st.markdown("### 📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("⚠ Parkinson Detected")
        else:
            st.success("✅ Healthy")

    with col2:
        st.metric("Risk Probability", f"{prob*100:.2f}%")

    # PDF Report
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200,10,txt="Parkinson Disease Prediction Report",ln=True)
    pdf.cell(200,10,txt=f"Prediction: {'Parkinson' if prediction[0]==1 else 'Healthy'}",ln=True)
    pdf.cell(200,10,txt=f"Risk Probability: {prob*100:.2f}%",ln=True)

    pdf.output("report.pdf")

    with open("report.pdf","rb") as file:
        st.download_button(
            label="Download Patient Report",
            data=file,
            file_name="parkinson_report.pdf",
            mime="application/pdf"
        )

# Feature Importance (Safe)
st.divider()
st.markdown("### 📊 Feature Importance")

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

# Footer
st.markdown(
    "<p style='text-align: center;'>Developed by Farhat 💛 | ML Healthcare App</p>",
    unsafe_allow_html=True
)

