import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and prepare dataset
df = pd.read_csv(r"C:\Users\RAVI\Downloads\ground_water_quality_in_andhra_pradesh.csv", encoding="latin")
df = df.rename(columns={
    'STATION CODE': 'Station_Code',
    'LOCATIONS': 'Location',
    'STATE': 'State',
    'pH : Mean : 6.5-8.5': 'pH',
    'B.O.D. (mg/l) : Mean : < 3 mg/l': 'BOD',
    'TOTAL COLIFORM (MPN/100ml) : Mean : < 5000 MPN/100ml': 'Coliform'
})

# Step 2: Select important features
df = df[['BOD', 'pH', 'Coliform']]

# Step 3: Handle missing values
df.dropna(subset=['BOD', 'pH', 'Coliform'], inplace=True)

# Step 4: Define safety rule (label generation)
df['Safe'] = ((df['BOD'] < 3) & (df['pH'].between(6.5, 8.5)) & (df['Coliform'] < 50)).astype(int)

# Step 5: Feature scaling and train-test split
features = df[['BOD', 'pH', 'Coliform']]
target = df['Safe']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Visualizations

# Pie chart of Safe vs Unsafe
df['Safe'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['Safe', 'Unsafe'], colors=['green', 'red'], startangle=90
)
plt.title("Water Safety Distribution")
plt.ylabel("")
plt.show()

# Bar chart of Safe vs Unsafe
safe_counts = df['Safe'].value_counts().sort_index()
labels = ['Unsafe', 'Safe']
colors = ['crimson', 'seagreen']

plt.figure(figsize=(6, 5))
plt.bar(labels, safe_counts, color=colors)
plt.title("Count of Safe vs Unsafe Water Samples")
plt.ylabel("Number of Samples")
plt.xlabel("Water Safety Status")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['BOD'], color='skyblue')
plt.title("BOD Distribution")
plt.ylabel("BOD (mg/l)")

plt.subplot(1, 3, 2)
sns.boxplot(y=df['pH'], color='lightgreen')
plt.title("pH Distribution")
plt.ylabel("pH Level")

plt.subplot(1, 3, 3)
sns.boxplot(y=df['Coliform'], color='salmon')
plt.title("Coliform Distribution")
plt.ylabel("Coliform Count (MPN/100ml)")

plt.tight_layout()
plt.show()

# Step 9: Manual Prediction
sample = pd.DataFrame([[2.5, 7.0, 45]], columns=features.columns)

# Scale the sample
sample_scaled = scaler.transform(sample)

# ML Model Prediction
ml_result = model.predict(sample_scaled)[0]

# Rule-Based Prediction
bod, ph, coliform = sample.iloc[0]
rule_result = int((bod < 3) and (6.5 <= ph <= 8.5) and (coliform < 50))

# Display Results
print("\n💧 Manual Prediction Result:")
print("✅ Water is SAFE" if rule_result == 1 else " ⚠ Water is UNSAFE")