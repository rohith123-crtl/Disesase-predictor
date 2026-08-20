import streamlit as st
import requests

# --- Configuration ---
API_URL = "http://localhost:8000"

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

# --- Fetch Symptoms from Backend ---
@st.cache_data
def fetch_symptoms():
    try:
        response = requests.get(f"{API_URL}/symptoms")
        response.raise_for_status()
        return response.json()["symptoms"]
    except requests.exceptions.ConnectionError:
        return None

symptoms = fetch_symptoms()

# --- Header ---
st.title("🩺 Medical Disease Predictor")
st.markdown("Select your symptoms below to get a predicted diagnosis powered by machine learning.")
st.divider()

# --- Check Backend Connection ---
if symptoms is None:
    st.error("❌ Cannot connect to the backend API. Make sure the FastAPI server is running:")
    st.code("uvicorn main:app --reload", language="bash")
    st.stop()

# --- Symptom Selection ---
st.subheader("📋 Select Your Symptoms")
user_symptoms = st.multiselect(
    "Choose one or more symptoms:",
    options=symptoms,
    placeholder="Start typing to search symptoms..."
)

st.caption(f"✅ {len(user_symptoms)} symptom(s) selected")

# --- Diagnosis Button ---
st.divider()
if st.button("🔍 Diagnose", use_container_width=True, type="primary"):
    if user_symptoms:
        with st.spinner("Analyzing symptoms..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"symptoms": user_symptoms}
                )
                response.raise_for_status()
                result = response.json()

                # Show result
                st.divider()
                st.subheader("📊 Diagnosis Result")
                st.success(f"**Predicted Disease:** {result['predicted_disease']}")

                # Show selected symptoms summary
                with st.expander("🧾 Symptoms you provided"):
                    for s in result["provided_symptoms"]:
                        st.markdown(f"- {s}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error communicating with the API: {e}")
    else:
        st.warning("⚠️ Please select at least one symptom before diagnosing.")

# --- Footer ---
st.divider()
st.caption("⚠️ Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical advice.")