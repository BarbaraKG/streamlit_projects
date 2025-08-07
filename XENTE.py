import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(page_title="Loan Default Predictor - Xente", layout="wide")

# ------------------------------------------------
# CUSTOM STYLING (Yellow and Black Theme)
# ------------------------------------------------
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #FFF9C4 !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            border-right: none;
        }
        html, body, [data-testid="stAppViewContainer"] > .main {
            background-color: #FFFDE7 !important;
            color: #000000 !important;
        }
        .stSelectbox > div,
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #FFF9C4 !important;
            border-radius: 8px;
        }
        input[type="number"],
        input[type="text"] {
            background-color: #FFF9C4 !important;
            color: #000000 !important;
            border-radius: 8px;
            padding: 0.4rem;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(2) {
            background: #FBC02D !important;
        }
        div[data-baseweb="slider"] > div > div > div:nth-child(3) {
            background: #d0d0d0 !important;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #FBC02D !important;
        }
        div.stButton > button {
            background-color: #FBC02D !important;
            color: black !important;
            border-radius: 8px !important;
            height: 3em;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #F9A825 !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------
model = joblib.load("loan_default_prediction.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Predictor", "About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=1,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#FFF9C4"},
            "icon": {"color": "#FBC02D", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#FFF176",
                "color": "#000000"
            },
            "nav-link-selected": {
                "background-color": "#FDD835",
                "color": "#000000"
            },
        },
    )

# ------------------------------------------------
# HOME TAB
# ------------------------------------------------
if selected == "Home":
    st.title("🏠 Welcome to the Xente Loan Default Predictor")
    st.write("This app uses machine learning to predict the likelihood of a customer defaulting on a loan.")

# ------------------------------------------------
# PREDICT TAB
# ------------------------------------------------
elif selected == "Predictor":
    st.title("📊 Predict Loan Default")
    st.write("Enter customer loan details below:")

    # Input fields
    product_category = st.selectbox("Product Category", ['Airtime', 'Data Bundles', 'Retail', 'Utility Bills', 'TV', 'Financial Services', 'Movies'])
    amount_loan = st.number_input("Amount of Loan", min_value=50.0, max_value=100000.0, value=5000.0)
    total_amount = st.number_input("Total Amount", min_value=50.0, max_value=100000.0, value=5000.0)

    if st.button("Predict Default Probability"):
        # Encoding
        product_category_mapping = {
            'Airtime': 0,
            'Data Bundles': 1,
            'Retail': 2,
            'Utility Bills': 3,
            'TV': 4,
            'Financial Services': 5,
            'Movies': 6
        }
        product_category_encoded = product_category_mapping.get(product_category, -1)

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'ProductCategory': product_category_encoded,
            'AmountLoan': amount_loan,
            'TotalAmount': total_amount
        }])

        input_data = input_data[X_columns]
        scaled_input = scaler.transform(input_data)
        final_input = pd.DataFrame(scaled_input, columns=X_columns)

        # Predict
        prob = model.predict_proba(final_input)[0][1] * 100
        st.success(f"Predicted Default Probability: {prob:.2f}%")

# ------------------------------------------------
# ABOUT TAB
# ------------------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
        This Streamlit app predicts the probability that a customer may default on a loan based on features from Xente's financial dataset.

        **Built by**: Group 7  
        **Final Model Used**: Logistic Regression (after RandomOverSampler)  
        **F1 Score (defaulters)**: 0.77  
        **Recall**: 0.94  
        **Libraries**: scikit-learn, pandas, numpy, streamlit
    """)
