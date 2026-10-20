import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏠 California House Price Predictor")
st.markdown("Fill in the housing details below and click **Predict** to get an estimated median house value.")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📍 Location")
    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
    with col2:
        latitude = st.number_input("Latitude", value=37.88, format="%.4f")

    st.subheader("🏘️ Housing Details")
    col3, col4 = st.columns(2)
    with col3:
        housing_median_age = st.number_input("Housing Median Age (years)", min_value=1.0, max_value=100.0, value=41.0, step=1.0)
        total_rooms = st.number_input("Total Rooms", min_value=1.0, value=880.0, step=1.0)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, value=129.0, step=1.0)
    with col4:
        population = st.number_input("Population", min_value=1.0, value=322.0, step=1.0)
        households = st.number_input("Households", min_value=1.0, value=126.0, step=1.0)
        median_income = st.number_input("Median Income (tens of thousands $)", min_value=0.0, value=8.33, step=0.01, format="%.4f")

    st.subheader("🌊 Ocean Proximity")
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        index=3,
    )

    st.divider()
    submitted = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                price = result["predicted_median_house_value"]
                st.success("Prediction successful!")
                st.metric(
                    label="Estimated Median House Value",
                    value=f"${price:,.2f}",
                )
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Make sure it is running with:\n\nuvicorn app:app --reload")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
