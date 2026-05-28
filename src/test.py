"""
Test model_simple on TRUE TEST DATASET
Compare performance between training data and test data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEST model_simple ON TEST DATASET")
print("=" * 80)

# Load test data
print("\n📊 Loading test data...")
df_test = pd.read_csv("wind_turbine_maintenance_test_data.csv")
print(f"✅ Test data loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

# Load model
print("\n🤖 Loading model_simple.pkl...")
model_simple = joblib.load("model_simple.pkl")
print(f"✅ Model loaded: {type(model_simple).__name__}")

# Prepare features
print("\n🔧 Preparing features...")
feature_cols = ['Rotor_Speed_RPM', 'Wind_Speed_mps', 'Power_Output_kW', 
                'Gearbox_Oil_Temp_C', 'Generator_Bearing_Temp_C', 
                'Vibration_Level_mmps', 'Ambient_Temp_C', 'Humidity_pct']

X_test = df_test[feature_cols].fillna(0)
y_test = df_test['Maintenance_Label'].values

print(f"✅ Features prepared: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Make predictions
print("\n🎯 Making predictions...")
predictions = model_simple.predict(X_test)
confidence = model_simple.predict_proba(X_test).max(axis=1)
print(f"✅ Predictions made: {len(predictions)} samples")

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE METRICS ON TEST SET")
print("=" * 80)

accuracy = accuracy_score(y_test, predictions)
f1_macro = f1_score(y_test, predictions, average='macro')
f1_weighted = f1_score(y_test, predictions, average='weighted')
precision_macro = precision_score(y_test, predictions, average='macro', zero_division=0)
recall_macro = recall_score(y_test, predictions, average='macro', zero_division=0)

print(f"\n📊 Overall Metrics:")
print(f"   Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score (Macro):      {f1_macro:.4f}  ⭐ IMPORTANT")
print(f"   F1-Score (Weighted):   {f1_weighted:.4f}")
print(f"   Precision (Macro):     {precision_macro:.4f}")
print(f"   Recall (Macro):        {recall_macro:.4f}")

# Confusion Matrix
print(f"\n📋 Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)

print(f"\n{'Predicted\\Actual':<20} {'Healthy':<15} {'Maintenance':<15} {'Critical':<15}")
print("-" * 65)

class_names = ["Healthy", "Maintenance", "Critical"]
for i, pred_name in enumerate(class_names):
    print(f"{pred_name:<20} {cm[i][0]:<15} {cm[i][1]:<15} {cm[i][2]:<15}")

# Distribution
print(f"\n📊 Prediction Distribution:")
unique_preds, counts = np.unique(predictions, return_counts=True)

for pred, count in zip(unique_preds, counts):
    pct = 100 * count / len(predictions)
    status = class_names[pred] if pred < 3 else f"Unknown({pred})"
    print(f"   {status:<20} {count:<10} ({pct:>5.1f}%)")

# Actual distribution
print(f"\n📊 Actual Distribution (Test Set):")
unique_actual, counts_actual = np.unique(y_test, return_counts=True)

for actual, count in zip(unique_actual, counts_actual):
    pct = 100 * count / len(y_test)
    status = class_names[actual] if actual < 3 else f"Unknown({actual})"
    print(f"   {status:<20} {count:<10} ({pct:>5.1f}%)")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON: Training Data vs Test Set")
print("=" * 80)

print(f"\n{'Metric':<35} {'Full Data (Train)':<20} {'Test Set':<20} {'Difference':<20}")
print("-" * 75)
print(f"{'F1-Score (Macro)':<35} {'0.7655':<20} {f1_macro:<20.4f} {f1_macro - 0.7655:<20.4f}")
print(f"{'Accuracy':<35} {'0.8778':<20} {accuracy:<20.4f} {accuracy - 0.8778:<20.4f}")
print(f"{'Precision (Macro)':<35} {'0.7300':<20} {precision_macro:<20.4f} {precision_macro - 0.7300:<20.4f}")
print(f"{'Recall (Macro)':<35} {'0.8337':<20} {recall_macro:<20.4f} {recall_macro - 0.8337:<20.4f}")

# ============================================================================
# VERDICT
# ============================================================================

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

drop = 0.7655 - f1_macro
drop_pct = (drop / 0.7655) * 100

print(f"""
📊 ANALYSIS:

Training Data F1-Score:  0.7655
Test Data F1-Score:      {f1_macro:.4f}
Drop:                    {drop:.4f} ({drop_pct:.1f}%)

{"❌ OVERFITTING DETECTED!" if drop > 0.1 else "✅ Good generalization"}

""")

if drop > 0.2:
    print("""
🚨 SEVERE OVERFITTING!
The model memorized training data but fails on test data.
This is a CRITICAL ISSUE!

Recommendations:
1. Use turbine1_lastmodel instead
2. Retrain model_simple with regularization
3. Use cross-validation properly
""")
elif drop > 0.1:
    print("""
⚠️  MODERATE OVERFITTING
The model shows some overfitting but still generalizes somewhat.

Recommendations:
1. Consider using turbine1_lastmodel
2. Increase model regularization
3. Use more training data
""")
else:
    print("""
✅ GOOD GENERALIZATION
The model performs similarly on test data as training data.
No significant overfitting detected.
""")

print("=" * 80)