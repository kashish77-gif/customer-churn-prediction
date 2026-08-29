import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL FILES
# =========================================================

model = joblib.load("churn_model.pkl")
scaler = joblib.load("churn_scaler.pkl")
features = joblib.load("churn_features.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6b7a;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 15px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
        text-align: center;
    }

    .probability {
        font-size: 38px;
        font-weight: 700;
    }

    .risk {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 50px;
        padding: 20px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a telecom customer is likely to churn using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col3:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# =========================================================
# SERVICES
# =========================================================

st.markdown(
    '<div class="section-title">🌐 Services</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# =========================================================
# BILLING INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0,
    step=10.0
)


# =========================================================
# PREDICTION
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    new_customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # One-hot encode categorical columns
    new_customer = pd.get_dummies(
        new_customer,
        columns=new_customer.select_dtypes(
            include=["object", "string"]
        ).columns,
        drop_first=True,
        dtype=int
    )

    # Match the exact training feature structure
    new_customer = new_customer.reindex(
        columns=features,
        fill_value=0
    )

    # Scale data
    customer_scaled = scaler.transform(new_customer)

    # Predict probability
    churn_probability = model.predict_proba(
        customer_scaled
    )[0][1]

    # Predict class
    prediction = model.predict(
        customer_scaled
    )[0]

    probability_percent = churn_probability * 100


    # =====================================================
    # RISK CATEGORY
    # =====================================================

    if probability_percent < 30:
        risk = "Low Risk"
    elif probability_percent < 60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )

    with result_col2:
        st.metric(
            "Risk Category",
            risk
        )

    with result_col3:
        prediction_text = (
            "Churn" if prediction == 1 else "No Churn"
        )

        st.metric(
            "Prediction",
            prediction_text
        )


    # Detailed message
    if prediction == 1:
        st.error(
            "⚠️ This customer is predicted to churn. "
            "Consider a retention strategy."
        )
    else:
        st.success(
            "✅ This customer is predicted to stay."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built using Python • Pandas • Scikit-learn • Logistic Regression • Streamlit
        <br>
        Customer Churn Prediction Project
    </div>
    """,
    unsafe_allow_html=True
)