# 3. Implementation

## 3.1 Technology Stack
-   **Programming Language**: Python
-   **Libraries**: Pandas (Data Manipulation), Scikit-learn (Machine Learning), Streamlit (Web App), Seaborn/Matplotlib (Visualization).

## 3.2 Approach

### Step 1: Exploratory Data Analysis (EDA)
We began by analyzing the dataset to understand the distribution of features.
*(Insert `report/images/feature_distributions.png` here)*
The correlation matrix helped us understand relationships between variables.
*(Insert `report/images/correlation_matrix.png` here)*

### Step 2: Data Preprocessing
The dataset was split into training (80%) and testing (20%) sets. No missing values were found, so imputation was not necessary.

### Step 3: Model Training
We trained a **Random Forest Classifier**. Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time. It was chosen for its high accuracy and ability to handle non-linear relationships.

Code Snippet (Model Training):
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### Step 4: Application Development
The web interface was built using Streamlit. It loads the saved model (`random_forest_model.pkl`) and provides a form for user input.
