import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Admission Prediction Dashboard",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Graduate Admission Prediction")
st.write("Interactive dashboard for predicting the Chance of Admit.")

st.divider()

# User inputs
st.subheader("Student Parameters")

cgpa = st.slider(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=8.0,
    step=0.01
)

gre = st.slider(
    "GRE Score",
    min_value=260,
    max_value=340,
    value=315,
    step=1
)

toefl = st.slider(
    "TOEFL Score",
    min_value=0,
    max_value=120,
    value=100,
    step=1
)

university_rating = st.number_input(
    "University Rating",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)

# Regression coefficient model
prediction = (
    -1.40549394
    + (0.14332317 * cgpa)
    + (0.00172822 * gre)
    + (0.00292854 * toefl)
    + (0.01123791 * university_rating)
)

# Keep prediction within valid probability range
prediction = np.clip(prediction, 0, 1)

st.divider()

# Prediction result
st.subheader("Prediction Result")

st.metric(
    label="Chance of Admit",
    value=f"{prediction:.2%}"
)

# Matplotlib visualization
fig, ax = plt.subplots()

ax.bar(
    ["Chance of Admit"],
    [prediction]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Predicted Admission Probability")

ax.text(
    0,
    prediction + 0.03,
    f"{prediction:.2%}",
    ha="center"
)

st.pyplot(fig)

st.caption(
    "Prediction is calculated using a regression coefficient model "
    "derived from the SageMaker Canvas batch prediction outputs."
)
