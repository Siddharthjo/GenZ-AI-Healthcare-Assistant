# GenZ Healthcare

## Overview
GenZ Healthcare is an AI-powered healthcare assistant designed to interact with users, predict diseases based on symptoms, and provide information about diseases and their symptoms. The assistant utilizes natural language processing (NLP) techniques and machine learning models to assist users in understanding their health conditions.

## Features
- User authentication and registration
- Symptom-based disease prediction
- Retrieval of disease symptoms
- Integration with Google search for nearby hospitals

## Technologies Used
- Python
- Flask (web framework)
- SQLAlchemy (database ORM)
- Gradio (for building the user interface)
- Pandas (data manipulation)
- NumPy (numerical computing)
- Scikit-learn (machine learning)

## Project Structure
- `app.py`: Main Flask application file containing routes and logic for the healthcare assistant.
- `msgConstant.py`: Constants file containing welcome messages for user interaction.
- `dataset.xlsx`: Dataset containing disease and symptom information.
- `model/random_forest.joblib`: Pre-trained machine learning model for disease prediction.
- `requirements.txt`: File listing all Python dependencies.

## How to Use
1. Clone this repository to your local machine.
2. Install the required Python dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the application in your web browser at `http://localhost:3000`.

## Team Members
- Siddharth Hiriyan
- Tanvi Shekhar Sawant 

## Project Workflow Video
Check out our project workflow in action by watching the video:
[Watch Video](https://drive.google.com/file/d/1TpEK3a_ALANkpAfgI5Uq4cPSoL69ebTp/view?usp=drive_link)

## Note
Ensure that you have Python installed on your system and the necessary dependencies are installed before running the application.

