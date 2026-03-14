import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
# Load dataset
df = pd.read_csv('data/car_preprocessed.csv')

# Split features & target
X = df.drop('class', axis=1)
y = df['class']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/car_price_model.pkl')
print("Model saved to models/car_price_model.pkl")

buying_high=int(input("buying_high:"))
buying_low=int(input("buying_low :"))
buying_med=int(input("buying_med:"))
buying_vhigh=int(input("buying_vhigh:"))
maint_high=int(input("maint_high:"))
maint_low=int(input("maint_low:"))
maint_med=int(input("maint_med:"))
maint_vhigh=int(input("maint_vhigh:"))
doors_2=int(input("doors_2:"))
doors_3=int(input("doors_3:"))
doors_4=int(input("doors_4:"))
doors_5more=int(input("doors_5more:"))
persons_2=int(input("persons_2:"))
persons_4=int(input("persons_4:"))
persons_more=int(input("persons_more:"))
lug_boot_big=int(input("lug_boot_big:"))
lug_boot_med=int(input("lug_boot_med:"))
lug_boot_small=int(input("lug_boot_small:"))
safety_high=int(input("safety_high:"))
safety_low=int(input("safety_low:"))
safety_med=int(input("safety_med:"))
print(model.predict([[buying_high,buying_low,buying_med,buying_vhigh,maint_high,maint_low,maint_med,maint_vhigh,doors_2,
                      doors_3,doors_4,doors_5more,persons_2,persons_4,persons_more,lug_boot_big,
                      lug_boot_med,lug_boot_small,safety_high,safety_low,safety_med]]))