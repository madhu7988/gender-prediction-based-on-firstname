import pickle
import joblib
import streamlit as st

# Load the input vectoriser from the pickle file
vectorizer_filename = 'C:\\Users\\User\\Desktop\\Projects(Self)\\Gender prediction with name\\vectoriser.pkl'
loaded_vectorizer = joblib.load(vectorizer_filename)

# Load the trained model from the pickle file
model_filename = 'C:\\Users\\User\\Desktop\\Projects(Self)\\Gender prediction with name\\trained_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Create a function to predict gender based on a name
def predict_gender(name):
    name=name.lower()
    X=loaded_vectorizer.transform([name]).toarray()
    gender = (model.predict(X)[0]>=0.5)
    if gender==0 :
       return "female"
    return "male"

# Create the Streamlit app
st.title("Name Gender Predictor")

# Get user input for the name
name = st.text_input("Enter a name:")

# Predict gender when the user clicks the button
if st.button("Predict Gender"):
    if not name:
        st.warning("Please enter a name")
    else:
        gender = predict_gender(name)
        st.success(f"Predicted gender: {gender}")
