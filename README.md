# Crop Recommendation System for Farmers 🌾

## Overview
A Machine Learning-based application that recommends the best crop to grow based on soil nutrients (Nitrogen, Phosphorus, Potassium) and environmental conditions (Temperature, Humidity, pH, Rainfall).

## Repository Structure
-   `data/`: Contains the dataset (`Crop_recommendation.csv`).
-   `src/`: Source code for EDA and training.
-   `models/`: Saved Random Forest model.
-   `report/`: Project report documentation and images.
-   `app.py`: Streamlit web application.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd "Crop Recommendation System for Farmers"
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Application**:
    ```bash
    python -m streamlit run app.py
    ```
    The app will open in your browser at `http://localhost:8501`.

2.  **Retrain Model** (Optional):
    ```bash
    python src/train_model.py
    ```

## Project Report
The detailed project report can be found in the `report/` directory.
