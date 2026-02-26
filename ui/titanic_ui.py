import streamlit as st
import requests

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival App", page_icon="🚢")

# --- SIDEBAR MENU ---
menu = st.sidebar.radio(
    "Menu",
    ("Titanic Survival Predictor", "Titanic Survival Summary")
)
st.sidebar.info("This app uses a Supervised Learning model (Logistic Regression) hosted on a FastAPI backend.")

# --- MAIN CONTENT ---
if menu == "Titanic Survival Predictor":
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("Enter passenger details below to see if they would have survived the disaster.")
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1st = Luxury, 3rd = Economy")
        age = st.slider("Age of Passenger", 0, 100, 25)
    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    if st.button("Predict Survival Status"):
        api_url = "http://127.0.0.1:8000/predict"
        params = {
            "pclass": pclass,
            "age": age,
            "sibsp": sibsp,
            "parch": parch
        }
        try:
            response = requests.get(api_url, params=params)
            result = response.json()
            if result["prediction"] == "Survived":
                st.success(f"✅ Result: {result['prediction']}")
                st.balloons()
            else:
                st.error(f"❌ Result: {result['prediction']}")
            st.write(f"**Confidence Level:** {result['probability_of_survival']}")
        except Exception as e:
            st.error("Error: Is the FastAPI server running? Make sure to start the FastAPI script first!")

elif menu == "Titanic Survival Summary":
    st.title("📊 Titanic Survival Summary")
    prompt_options = [
        "What is the survival rate for luxury class?",
        "What is the survival rate for passengers aged 20 to 30?",
        "Show cluster summary for passengers.",
        "How many children survived?",
        "How many seniors survived?",
        "Show survival rate by gender.",
        "Show survival rate by family.",
        "What is the survival rate for each class?",
        "What is the average age of survivors?",
        "What features influence survival?"
    ]
    selected_prompt = st.selectbox("Select a summary prompt or type your own:", prompt_options)
    custom_prompt = st.text_input("Or enter your own question/statement:", value=selected_prompt)
    if st.button("Get Summary"):
        summary_url = "http://127.0.0.1:8000/summary"
        params = {"prompt": custom_prompt} if custom_prompt else {}
        try:
            summary_response = requests.get(summary_url, params=params)
            summary = summary_response.json()
            st.subheader("Summary Result")
            if "error" in summary:
                st.error(summary["error"])
            else:
                for k, v in summary.items():
                    st.write(f"{k}: {v}")
        except Exception as e:
            st.error("Error fetching summary. Is the FastAPI server running?")