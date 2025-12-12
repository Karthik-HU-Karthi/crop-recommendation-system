# 4. Challenges and Solutions

## 4.1 Challenges
-   **Imbalanced Data**: Initially, we were concerned about class imbalance, which can lead to biased models.
-   **Feature Selection**: Deciding which environmental factors were most critical for the prediction.

## 4.2 Solutions
-   **Dataset Verification**: Upon EDA, we verified that the dataset was perfectly balanced (100 samples per crop), eliminating the need for resampling techniques like SMOTE.
-   **Model Choice**: We chose Random Forest because it automatically handles feature importance and interactions well, reducing the need for manual feature engineering.
