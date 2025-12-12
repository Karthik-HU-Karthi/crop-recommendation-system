# 2. System Architecture

## 2.1 Overview
The system follows a standard Machine Learning pipeline architecture, integrated with a web-based frontend.

## 2.2 Key Components
1.  **Data Source**: A dataset representing historical records of soil conditions, weather patterns, and the crops grown in those conditions.
2.  **Preprocessing Module**: Cleans and prepares the data for modeling. This includes feature selection and data splitting.
3.  **Machine Learning Model**: The core engine that learns patterns from the data. We utilize a **Random Forest Classifier** due to its robustness and high accuracy in multi-class classification problems.
4.  **Web Application (Frontend)**: Built with **Streamlit**, this component accepts user input and communicates with the trained model to display predictions.

## 2.3 Data Flow
1.  User enters data (N, P, K, etc.) in the Web App.
2.  App sends data to the loading Model.
3.  Model processes input and returns a prediction (Crop Name).
4.  App displays the result to the User.
