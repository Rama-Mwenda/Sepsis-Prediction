import streamlit as st
import requests
import time


# Define the backend server url
backend_server_url = "http://127.0.0.1:8000"

# Page configurations
st.set_page_config(
    page_title = ('Prediction'),
    layout = 'wide',
    page_icon = '🧠'
)

st.title("SEPSIS PREDICTION APP")
st.divider()

st.write('Please Select a model to predict')
 
def model_select(): 
    model = None
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button('Adaboost', use_container_width=True):
            model = "http://api:8000/adapredict"
            st.session_state['model'] = model
        
    with colB:
        if st.button('Logistic Regression', use_container_width=True):
            model = "http://api:8000/lrpredict"
            st.session_state['model'] = model
        
    with colC:
        if st.button('Random Forest', use_container_width=True):
            model = "http://api:8000/rfpredict"
            st.session_state['model'] = model
    
    return model

def show_form():
    with st.form('input_features'):
        # text_fields for sepsis features
        st.header("Enter Patient Attributes")

        # Split form into three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            plasma = st.number_input('Plasma Glucose', min_value=0.0, step=0.1, key='plasma')
            bt1 = st.number_input("Blood Work Result-1 (mu U/ml)", min_value=0.0, step=0.1, key='btn1')
            pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, step=0.1, key='pressure')
        with col2:
            bt2 = st.number_input("Blood Work Result-2 (mm)", min_value=0.0, step=0.1, key='bt2')
            bt3 = st.number_input("Blood Work Result-3 (mu U/ml)", min_value=0.0, step=0.1, key='bt3')
            bmi = st.number_input("Body Mass Index (BMI) ", min_value=0.0, step=0.1, key='bmi')
        with col3:    
            bt4 = st.number_input("Blood Work Result-4 (mu U/ml)", min_value=0.0, step=0.1, key='bt4')
            age = st.number_input("Patient Age", min_value=0, step=1, max_value=100, key='age')
            insurance = st.radio("Insurance", ('Positive', 'Negative'))

        # Prediction button
        if st.form_submit_button("Make Prediction"):
            # input data dictionary
            input_data = {
                "plasma": plasma,
                "bt1": bt1,
                "pressure": pressure,
                "bt2": bt2,
                "bt3": bt3,
                "age": age,
                "insurance": insurance,
                "bmi": bmi,
                "bt4": bt4
            }              
            
            response = requests.post(st.session_state['model'], json=input_data)
            
            if response.status_code == 200:
                prediction = response.json().get('results')
                return prediction
            
def prediction():
    model = model_select()
    prediction = show_form()
    if prediction:
        if prediction['prediction'] == 'Negative':
            st.success(f"The patient is likely to be sepsis {prediction['prediction']} with a probability of {prediction['probability']}%")
        else:
            st.error(f"The patient is likely to be sepsis {prediction['prediction']} with a probability of {prediction['probability']}%")

if __name__ == '__main__':
    prediction()
