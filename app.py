# app.py - Streamlit Dashboard Code

import streamlit as st
import pandas as pd
import joblib
import os # To construct file paths reliably
import warnings

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Slow Learner Prediction Tool",
    page_icon="🎓",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Constants ---
SCALER_FILENAME = 'scaler.joblib'
MODEL_FILENAME = 'random_forest_model.joblib'
FEATURE_NAMES = ['Standardized_Test_Score', 'Average_Grade', 'Attendance_Percentage', 'Times_Late', 'Participation_Rating']


# --- Load Model and Scaler ---
# Use caching to load only once
@st.cache_resource
def load_artifacts(scaler_path, model_path):
    """Loads the scaler and model, handling potential errors."""
    try:
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        print("Scaler and Model loaded successfully.")
        return scaler, model
    except FileNotFoundError:
        st.error(f"Error: Could not find required files ('{SCALER_FILENAME}', '{MODEL_FILENAME}'). "
                 "Please ensure they are in the same directory as the app script.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        return None, None

# Construct absolute paths (more robust for deployment)
script_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(script_dir, SCALER_FILENAME)
model_path = os.path.join(script_dir, MODEL_FILENAME)

scaler, final_model = load_artifacts(scaler_path, model_path)

# --- Helper Functions (Prediction and Suggestions) ---
# These should be the same functions developed in Steps 7 and 8

def predict_slow_learner(data, scaler_obj, model_obj):
    """ Predicts using loaded scaler and model. """
    if scaler_obj is None or model_obj is None:
        return None, None # Return None if artifacts didn't load

    # Create input DataFrame
    input_data = pd.DataFrame([data], columns=FEATURE_NAMES)

    # Scale the input data
    # Ignore feature names warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_data_scaled = scaler_obj.transform(input_data)

        # Make prediction
        prediction = model_obj.predict(input_data_scaled)[0]

        # Get probability
        probabilities = model_obj.predict_proba(input_data_scaled)[0]
        probability_slow_learner = probabilities[1]

    return prediction, probability_slow_learner

def get_remedial_suggestions(prediction, probability, standardized_test_score, average_grade, attendance_percentage, participation_rating):
    """ Generates remedial suggestions based on prediction and inputs. """
    suggestions = []
    if prediction == 1:
        suggestions.append(f"**Student identified as potentially needing support (Model Probability: {probability:.2%})**") # Use percentage
        suggestions.append("---") # Separator
        suggestions.append("**Recommended Actions:**")

        # General Suggestions
        suggestions.append("* Schedule a one-on-one meeting to discuss challenges and learning style.")
        suggestions.append("* Provide simplified explanations and break down complex topics.")
        suggestions.append("* Offer extra practice exercises tailored to weak areas.")
        suggestions.append("* Utilize visual aids and hands-on activities.")
        suggestions.append("* Encourage asking questions in a supportive environment.")
        suggestions.append("* Recommend peer tutoring or study groups if appropriate.")

        # Specific Suggestions based on input values (Adjust thresholds as needed)
        suggestions.append("\n**Observations & Targeted Suggestions:**") # Sub-header
        triggered_specific = False
        if standardized_test_score < 45:
            suggestions.append("* *Observation:* Low Standardized Test Score.")
            suggestions.append("    * *Suggestion:* Focus on foundational concepts related to test topics.")
            triggered_specific = True
        if average_grade < 70:
            suggestions.append("* *Observation:* Low Average Grade.")
            suggestions.append("    * *Suggestion:* Review recent assignments/tests for specific weaknesses.")
            suggestions.append("    * *Suggestion:* Offer opportunities for re-assessment or extra credit focused on improvement.")
            triggered_specific = True
        if attendance_percentage < 85:
            suggestions.append("* *Observation:* Low Attendance.")
            suggestions.append("    * *Suggestion:* Discuss potential barriers to attendance.")
            triggered_specific = True
        if participation_rating <= 2:
            suggestions.append("* *Observation:* Low Class Participation.")
            suggestions.append("    * *Suggestion:* Create low-pressure participation opportunities (e.g., think-pair-share).")
            suggestions.append("    * *Suggestion:* Positively reinforce participation attempts.")
            triggered_specific = True

        if not triggered_specific:
             suggestions.append("* No specific low metrics triggered additional suggestions based on current thresholds, focus on general strategies.")

        suggestions.append("\n---") # Separator
        suggestions.append("**Follow Up:** Monitor progress closely and adjust strategies as needed.")

    return suggestions


# --- Streamlit App UI ---
st.title("🎓 Student Support Predictor")
st.caption("Predicting potential needs for student support using academic and behavioral data.")

# Add some instructions or context in the sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This tool uses a Random Forest machine learning model (trained on sample data) "
    "to identify students who might benefit from additional academic support. "
    "Enter the student's details below and click 'Predict'."
)
st.sidebar.header("Input Student Details")


# --- Input Fields ---
# Use columns for a potentially cleaner layout, or just stack them
# col1, col2 = st.columns(2)

# with col1:
score = st.sidebar.number_input(
    "Standardized Test Score",
    min_value=0.0, max_value=100.0, value=50.0, step=0.5, format="%.1f",
    help="Enter the student's most recent standardized test score (0-100)."
)
grade = st.sidebar.number_input(
    "Average Grade (%)",
    min_value=0.0, max_value=100.0, value=75.0, step=0.5, format="%.1f",
    help="Enter the student's current average grade percentage (0-100)."
)
attendance = st.sidebar.number_input(
    "Attendance Percentage (%)",
    min_value=0.0, max_value=100.0, value=90.0, step=0.5, format="%.1f",
    help="Enter the student's attendance rate (0-100)."
)
# with col2:
late = st.sidebar.number_input(
    "Times Late to Class (Count)",
    min_value=0, value=2, step=1,
    help="Enter the number of times the student has been late recently."
)
participation = st.sidebar.number_input(
    "Participation Rating (1-5)",
    min_value=1, max_value=5, value=3, step=1,
    help="Rate the student's class participation (1=Very Low, 5=Very High)."
)


# --- Prediction Button ---
predict_button = st.sidebar.button("✨ Predict Support Need", type="primary", use_container_width=True)


# --- Display Results ---
st.header("Prediction Results")

if scaler is None or final_model is None:
    st.warning("Model artifacts not loaded. Please check file paths and logs.")
elif predict_button:
    # Prepare input data dictionary
    input_features = {
        'Standardized_Test_Score': score,
        'Average_Grade': grade,
        'Attendance_Percentage': attendance,
        'Times_Late': late,
        'Participation_Rating': participation
    }

    # Get prediction and probability
    prediction, probability = predict_slow_learner(input_features, scaler, final_model)

    if prediction is not None:
        # Display prediction status
        if prediction == 1:
            st.warning("Prediction: **Potential Need for Support Identified**")
        else:
            st.success("Prediction: **No Specific Support Need Identified by Model**")

        # Display probability using a metric or progress bar
        st.metric(label="Model Confidence (Probability of Needing Support)", value=f"{probability:.1%}")
        # st.progress(probability, text=f"{probability:.1%}") # Alternative visualization

        # Get and display suggestions
        suggestions = get_remedial_suggestions(prediction, probability, score, grade, attendance, participation)

        if suggestions:
            with st.expander("💡 View Recommended Remedial Suggestions", expanded=True):
                for suggestion in suggestions:
                    st.markdown(suggestion) # Use markdown for formatting (like bullets)
    else:
        st.error("Prediction could not be made. Check artifact loading.")

else:
    st.info("Enter student details in the sidebar and click 'Predict Support Need' to see the results.")


# --- Footer (Optional) ---
st.markdown("---")
st.caption("Disclaimer: This tool provides suggestions based on a predictive model. "
           "Always use professional judgment and consider individual student circumstances.")