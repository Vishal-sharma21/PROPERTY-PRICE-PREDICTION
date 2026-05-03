import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("newmodel.pkl", "rb"))
scaler = pickle.load(open("newscaler.pkl", "rb"))
encoder = pickle.load(open("newencoder.pkl", "rb"))


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .main-box {
        background: rgba(0, 0, 0, 0.6);
        padding: 30px;
        border-radius: 15px;
        color: white;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
    }

    h1 {
        text-align: center;
        color: white;
    }

    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
    }

    .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>🏠 House Price Prediction</h1>", unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bhk = st.selectbox("BHK", [1,2,3,4,5,6,7,8,9,10])
        bhk_type = st.selectbox("Type", ["BHK", "RK"])
        square_ft = st.number_input("Square Feet", 100, 50000)

    with col2:
        ready = st.selectbox("Ready to Move", ["Yes", "No"])
        under = st.selectbox("Under Construction", ["Yes", "No"])
        rera = st.selectbox("RERA Status", ["Approved", "Not Approved"])

   
    bhk_type_enc = encoder.transform([bhk_type])[0]
    ready = 1 if ready == "Yes" else 0
    under = 1 if under == "Yes" else 0
    rera = 1 if rera == "Approved" else 0

    square_ft = np.log1p(square_ft)

    features = np.array([[bhk, bhk_type_enc, square_ft, ready, under, rera]])
    features = scaler.transform(features)

    
    if st.button(" Predict Price"):
        pred = model.predict(features)
        price_lakhs = np.expm1(pred[0])

        if price_lakhs >= 100:
            price_cr = price_lakhs / 100
            st.success(f" Estimated Price: ₹ {price_cr:.2f} Cr")
        else:
            st.success(f" Estimated Price: ₹ {price_lakhs:.2f} Lakhs")

    st.markdown('</div>', unsafe_allow_html=True)