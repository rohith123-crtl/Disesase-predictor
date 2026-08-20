import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. LOAD DATA ---
print("Loading dataset...")
df = pd.read_csv("dataset.csv")

# --- 2. CLEAN & TRANSFORM DATA ---
print("Cleaning and reshaping data...")
# Extract all unique symptoms from the dataset
symptom_columns = df.columns[1:] 
all_symptoms = pd.unique(df[symptom_columns].values.ravel('K'))
unique_symptoms = [str(s).strip() for s in all_symptoms if pd.notna(s)]
unique_symptoms = list(set(unique_symptoms))

# Create a new matrix of 0s
binary_df = pd.DataFrame(0, index=df.index, columns=unique_symptoms)

# Populate the matrix with 1s where a symptom exists for a patient
for col in symptom_columns:
    for idx, val in df[col].items():
        if pd.notna(val):
            symptom = str(val).strip()
            binary_df.at[idx, symptom] = 1

# Combine the target ('Disease') with the new binary symptom features
clean_df = pd.concat([df[['Disease']], binary_df], axis=1)

# --- 3. TRAIN/TEST SPLIT ---
X = clean_df.drop('Disease', axis=1)
y = clean_df['Disease']

# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAIN THE MODEL ---
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. EVALUATE ---
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report Preview:")
# Only printing the first few lines of the report to keep the output clean
print('\n'.join(classification_report(y_test, y_pred).split('\n')[:10]))
import joblib

# Save the model
joblib.dump(model, 'disease_model.pkl')
# Save the symptom list so the frontend knows the exact columns
joblib.dump(unique_symptoms, 'symptoms_list.pkl')
print("Model and symptoms saved successfully!")