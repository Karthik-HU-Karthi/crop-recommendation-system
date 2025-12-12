import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directories if not exist
os.makedirs('report/images', exist_ok=True)

# Load data
df = pd.read_csv('data/Crop_recommendation.csv')

print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget Class Distribution:")
print(df['label'].value_counts())

# Detailed Stats
print("\nDescriptive Statistics:")
print(df.describe())

# Correlation Matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('report/images/correlation_matrix.png')
print("\nSaved correlation_matrix.png")

# Distribution of features
features = numeric_df.columns
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('report/images/feature_distributions.png')
print("\nSaved feature_distributions.png")
