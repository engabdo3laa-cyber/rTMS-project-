# ============================================================
# Random Forest Model to Predict Patient Response to Treatment
# ============================================================

# -------------------- 1. IMPORT LIBRARIES --------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# -------------------- 2. LOAD DATA --------------------
print("=" * 60)
print("RANDOM FOREST - TREATMENT RESPONSE PREDICTION")
print("=" * 60)

# Load the processed dataset
file_path = r"C:\Users\Drago\Documents\rTMS project\Dipression_prideacte_project\TDBRAIN_data\patients_with_full_processed_data.csv"
df = pd.read_csv(file_path)

print(f"\n✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📊 Number of patients: {len(df)}")
print(f"📊 Number of features: {len(df.columns)}")

# -------------------- 3. EXPLORE DATA --------------------
print("\n" + "=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

# First few rows
print("\n📋 First 5 rows:")
print(df.head())

# Data types
print("\n📋 Data types:")
print(df.dtypes.value_counts())

# Check for missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

print(f"\n📊 Missing values in dataset: {missing_df.shape[0]} columns have missing values")
if missing_df.shape[0] > 0:
    print(missing_df)

# Target variable distribution
if 'Responder' in df.columns:
    responder_counts = df['Responder'].value_counts()
    responder_percent = df['Responder'].value_counts(normalize=True) * 100
    
    print("\n🎯 Target Variable: Responder")
    print(f"   Responders (1): {responder_counts.get(1, 0)} patients ({responder_percent.get(1, 0):.1f}%)")
    print(f"   Non-responders (0): {responder_counts.get(0, 0)} patients ({responder_percent.get(0, 0):.1f}%)")
else:
    print("\n❌ 'Responder' column not found in dataset!")
    print("Available columns:", df.columns.tolist())
    exit()

# -------------------- 4. IDENTIFY FEATURES --------------------
print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Columns to exclude from features
exclude_cols = ['participants_ID', 'ID', 'Responder', 'Remitter', 'condition', 'session', 
                'BDI_post', 'BDI_change', 'nrSessions', 'Unnamed: 0']

# EEG features (based on your processed data)
eeg_features = ['FAA', 'coherence', 'alpha', 'theta', 'beta']

# Clinical features (add any that exist in your data)
clinical_features = []
for col in df.columns:
    if col not in exclude_cols + eeg_features:
        if col not in ['participants_ID', 'ID', 'Responder']:
            clinical_features.append(col)

# All features
all_features = eeg_features + clinical_features
available_features = [f for f in all_features if f in df.columns]

print(f"\n✅ EEG Features ({len([f for f in available_features if f in eeg_features])}):")
print([f for f in available_features if f in eeg_features])

print(f"\n✅ Clinical Features ({len([f for f in available_features if f in clinical_features])}):")
print([f for f in available_features if f in clinical_features])

print(f"\n📊 Total features available: {len(available_features)}")

# -------------------- 5. PREPARE DATA --------------------
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Create feature matrix X and target vector y
X = df[available_features].copy()
y = df['Responder'].copy()

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# Check for missing values
missing_cols = X.columns[X.isnull().any()].tolist()
if missing_cols:
    print(f"\n📊 Columns with missing values: {len(missing_cols)}")
    for col in missing_cols[:10]:  # Show first 10
        missing_pct = X[col].isnull().mean() * 100
        print(f"   {col}: {missing_pct:.1f}% missing")

# Handle missing values
if X.isnull().sum().sum() > 0:
    print("\n🔄 Handling missing values...")
    
    # Remove columns that are completely empty
    empty_cols = X.columns[X.isnull().all()].tolist()
    if empty_cols:
        print(f"   Removing completely empty columns: {empty_cols}")
        X = X.drop(columns=empty_cols)
        available_features = [f for f in available_features if f not in empty_cols]
    
    # For remaining missing values, use median imputation
    if X.isnull().sum().sum() > 0:
        # Select only numeric columns for imputation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            X_numeric_imputed = imputer.fit_transform(X[numeric_cols])
            X_numeric = pd.DataFrame(X_numeric_imputed, columns=numeric_cols, index=X.index)
            
            # Replace numeric columns with imputed values
            for col in numeric_cols:
                X[col] = X_numeric[col]
        
        # For categorical columns, fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    print(f"   Missing values after imputation: {X.isnull().sum().sum()}")
    print(f"✅ New X shape: {X.shape}")

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"\n🔄 Encoding categorical variables: {categorical_cols.tolist()}")
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes

# Scale features (optional for Random Forest, but good for comparison)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\n📊 Feature statistics after scaling:")
print(X_scaled.describe().loc[['mean', 'std']].round(2))


# -------------------- 6. SPLIT DATA --------------------
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print(f"✅ Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✅ Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n📊 Training set distribution:")
print(y_train.value_counts())
print(f"\n📊 Test set distribution:")
print(y_test.value_counts())

# -------------------- 7. TRAIN RANDOM FOREST MODEL --------------------
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

# Create base model
base_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Train the model
print("🔄 Training base model...")
base_rf.fit(X_train, y_train)
print("✅ Model training complete!")

# Predictions
y_pred_train = base_rf.predict(X_train)
y_pred_test = base_rf.predict(X_test)
y_pred_proba_train = base_rf.predict_proba(X_train)[:, 1]
y_pred_proba_test = base_rf.predict_proba(X_test)[:, 1]

# -------------------- 8. EVALUATE MODEL --------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_auc = roc_auc_score(y_train, y_pred_proba_train)
test_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"\n📊 Training Set Performance:")
print(f"   Accuracy: {train_accuracy:.4f}")
print(f"   AUC-ROC: {train_auc:.4f}")

print(f"\n📊 Test Set Performance:")
print(f"   Accuracy: {test_accuracy:.4f}")
print(f"   AUC-ROC: {test_auc:.4f}")

# Detailed classification report
print("\n📊 Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Non-Responder', 'Responder']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Responder', 'Responder'],
            yticklabels=['Non-Responder', 'Responder'])
plt.title('Confusion Matrix - Random Forest', fontsize=14)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Random Forest', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("✅ ROC curve saved as 'roc_curve.png'")

# -------------------- 9. FEATURE IMPORTANCE --------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': base_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.title('Feature Importance - Random Forest (Top 20)', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("✅ Feature importance plot saved as 'feature_importance.png'")

# -------------------- 10. HYPERPARAMETER TUNING --------------------
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("🔄 Performing grid search with 5-fold CV...")
print(f"   Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid,
    cv=5,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV AUC: {grid_search.best_score_:.4f}")

# Best model
best_rf = grid_search.best_estimator_

# Evaluate best model
y_pred_best = best_rf.predict(X_test)
y_pred_proba_best = best_rf.predict_proba(X_test)[:, 1]
best_accuracy = accuracy_score(y_test, y_pred_best)
best_auc = roc_auc_score(y_test, y_pred_proba_best)

print(f"\n📊 Tuned Model Performance:")
print(f"   Accuracy: {best_accuracy:.4f} (vs base: {test_accuracy:.4f})")
print(f"   AUC-ROC: {best_auc:.4f} (vs base: {test_auc:.4f})")

# Improvement
imp_accuracy = ((best_accuracy - test_accuracy) / test_accuracy) * 100
imp_auc = ((best_auc - test_auc) / test_auc) * 100

print(f"\n📈 Improvement:")
print(f"   Accuracy: {imp_accuracy:+.2f}%")
print(f"   AUC-ROC: {imp_auc:+.2f}%")

# -------------------- 11. CROSS-VALIDATION --------------------
print("\n" + "=" * 60)
print("CROSS-VALIDATION")
print("=" * 60)

# Perform cross-validation
cv_scores = cross_val_score(best_rf, X_scaled, y, cv=5, scoring='roc_auc')

print(f"✅ 5-Fold CV AUC scores: {cv_scores}")
print(f"✅ Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# -------------------- 12. SAVE MODEL AND RESULTS --------------------
print("\n" + "=" * 60)
print("SAVING MODEL AND RESULTS")
print("=" * 60)

import joblib
import datetime

# Save model
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"random_forest_responder_{timestamp}.pkl"
joblib.dump(best_rf, model_filename)
print(f"✅ Model saved as: {model_filename}")

# Save scaler
scaler_filename = f"scaler_{timestamp}.pkl"
joblib.dump(scaler, scaler_filename)
print(f"✅ Scaler saved as: {scaler_filename}")

# Save feature list
features_filename = f"features_{timestamp}.txt"
with open(features_filename, 'w') as f:
    for feature in available_features:
        f.write(f"{feature}\n")
print(f"✅ Features list saved as: {features_filename}")

# Save results summary
results = {
    'Model': 'Random Forest',
    'Target': 'Responder',
    'Features': len(available_features),
    'Train Size': len(X_train),
    'Test Size': len(X_test),
    'Base Accuracy': test_accuracy,
    'Base AUC': test_auc,
    'Tuned Accuracy': best_accuracy,
    'Tuned AUC': best_auc,
    'CV Mean AUC': cv_scores.mean(),
    'CV Std AUC': cv_scores.std(),
    'Best Parameters': grid_search.best_params_,
    'Top Features': feature_importance.head(10).to_dict()
}

results_df = pd.DataFrame([results])
results_df.to_csv(f"model_results_{timestamp}.csv", index=False)
print(f"✅ Results saved as: model_results_{timestamp}.csv")

# -------------------- 13. PREDICTION FUNCTION --------------------
print("\n" + "=" * 60)
print("PREDICTION FUNCTION")
print("=" * 60)

def predict_response(new_data, model, scaler, feature_names):
    """
    Predict treatment response for new patients
    
    Parameters:
    -----------
    new_data : pandas DataFrame or dict
        Patient data with same features as training
    model : trained model
        The trained Random Forest model
    scaler : fitted scaler
        The fitted StandardScaler
    feature_names : list
        List of feature names used in training
    
    Returns:
    --------
    predictions : array
        Predicted class (0 or 1)
    probabilities : array
        Prediction probabilities
    """
    # Convert to DataFrame if dict
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])
    
    # Ensure correct features
    X_new = new_data[feature_names].copy()
    
    # Handle missing values
    if X_new.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        X_new = pd.DataFrame(imputer.fit_transform(X_new), columns=feature_names)
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)
    
    return predictions, probabilities

print("✅ Prediction function defined!")
print("\n📝 Example usage:")
print("""
# Load model and scaler
model = joblib.load('random_forest_responder_20250215_123456.pkl')
scaler = joblib.load('scaler_20250215_123456.pkl')

# New patient data
new_patient = {
    'FAA': 0.35,
    'coherence': 0.72,
    'alpha': 8500,
    'theta': 45000,
    'beta': 12000,
    'age': 45,
    'BDI_pre': 28
}

# Predict
pred, prob = predict_response(new_patient, model, scaler, feature_names)
print(f"Prediction: {'Responder' if pred[0] == 1 else 'Non-Responder'}")
print(f"Probability of response: {prob[0][1]:.3f}")
""")

# -------------------- 14. SUMMARY --------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"""
✅ RANDOM FOREST MODEL COMPLETED SUCCESSFULLY!

📊 Dataset: {len(df)} patients, {len(available_features)} features
🎯 Target: Responder

📈 Best Model Performance:
   - Accuracy: {best_accuracy:.4f}
   - AUC-ROC: {best_auc:.4f}
   - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})

📁 Saved Files:
   - Model: {model_filename}
   - Scaler: {scaler_filename}
   - Features: {features_filename}
   - Results: model_results_{timestamp}.csv
   - Plots: confusion_matrix.png, roc_curve.png, feature_importance.png

🔍 Top 5 Most Important Features:
""")

for i, row in feature_importance.head(5).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "=" * 60)
print("🎉 MODEL READY FOR USE!")
print("=" * 60)



# ============================================================
# PREDICT MULTIPLE PATIENTS FROM FILE
# ============================================================

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def load_model_files():
    """Load the latest model, scaler, and features"""
    
    # Find the latest model file (you can also specify a specific file)
    model_files = list(Path('.').glob('random_forest_responder_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No model files found!")
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    timestamp = str(latest_model).replace('random_forest_responder_', '').replace('.pkl', '')
    
    model_file = f'random_forest_responder_{timestamp}.pkl'
    scaler_file = f'scaler_{timestamp}.pkl'
    features_file = f'features_{timestamp}.txt'
    
    print(f"📁 Using model: {model_file}")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    with open(features_file, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return model, scaler, feature_names

def predict_patients_file(input_file, output_file=None):
    """
    Predict response for all patients in a CSV file
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with patient data
    output_file : str, optional
        Path to save results. If None, auto-generated name will be used
    """
    
    print("\n" + "=" * 60)
    print("PREDICT PATIENTS FROM FILE")
    print("=" * 60)
    
    # Load model
    try:
        model, scaler, feature_names = load_model_files()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please specify model files manually or train the model first.")
        return None
    
    # Load patient data
    try:
        patients_df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(patients_df)} patients from: {input_file}")
        print(f"📊 File columns: {list(patients_df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # Check which features are available in the input file
    available_features = [f for f in feature_names if f in patients_df.columns]
    missing_features = [f for f in feature_names if f not in patients_df.columns]
    
    print(f"\n📊 Features available: {len(available_features)}/{len(feature_names)}")
    if missing_features:
        print(f"⚠️ Missing features: {missing_features[:10]}... (will be filled with 0)")
    
    # Prepare data for prediction
    X_pred = pd.DataFrame(index=patients_df.index)
    
    for feature in feature_names:
        if feature in patients_df.columns:
            X_pred[feature] = patients_df[feature]
        else:
            X_pred[feature] = 0  # Fill missing with 0
    
    # Ensure correct column order
    X_pred = X_pred[feature_names]
    
    # Check for missing values
    if X_pred.isnull().sum().sum() > 0:
        print("⚠️ Missing values detected, filling with 0")
        X_pred = X_pred.fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Create results dataframe
    results_df = patients_df[['participants_ID']].copy() if 'participants_ID' in patients_df.columns else pd.DataFrame()
    
    # Add ID if not present
    if 'participants_ID' not in results_df.columns:
        results_df['participants_ID'] = [f"Patient_{i}" for i in range(len(patients_df))]
    
    # Add predictions
    results_df['Responder_Predicted'] = predictions
    results_df['Responder_Label'] = results_df['Responder_Predicted'].map({1: 'Yes', 0: 'No'})
    results_df['Probability_Response'] = probabilities[:, 1]
    results_df['Probability_NonResponse'] = probabilities[:, 0]
    
    # Calculate confidence
    results_df['Confidence'] = np.max(probabilities, axis=1)
    results_df['Confidence_Level'] = pd.cut(
        results_df['Confidence'],
        bins=[0, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Add original features for reference
    for feature in ['age', 'BDI_pre', 'FAA', 'alpha']:
        if feature in patients_df.columns:
            results_df[feature] = patients_df[feature]
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    n_responders = (predictions == 1).sum()
    n_non_responders = (predictions == 0).sum()
    
    print(f"📊 Total patients: {len(patients_df)}")
    print(f"✅ Predicted Responders: {n_responders} ({n_responders/len(patients_df)*100:.1f}%)")
    print(f"❌ Predicted Non-responders: {n_non_responders} ({n_non_responders/len(patients_df)*100:.1f}%)")
    print(f"\n📈 Average probability of response: {probabilities[:, 1].mean():.2%}")
    print(f"📊 Confidence levels:")
    print(f"   High confidence: {(results_df['Confidence_Level'] == 'High').sum()} patients")
    print(f"   Medium confidence: {(results_df['Confidence_Level'] == 'Medium').sum()} patients")
    print(f"   Low confidence: {(results_df['Confidence_Level'] == 'Low').sum()} patients")
    
    # Save results
    if output_file is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Show first few predictions
    print("\n📋 First 10 predictions:")
    display_cols = ['participants_ID', 'Responder_Label', 'Probability_Response', 'Confidence_Level']
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].head(10).to_string())
    
    return results_df

def predict_single_patient_interactive():
    """Interactive mode to input single patient data"""
    
    print("\n" + "=" * 60)
    print("INTERACTIVE SINGLE PATIENT PREDICTION")
    print("=" * 60)
    
    # Load model
    try:
        model, scaler, feature_names = load_model_files()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    
    print(f"\n📝 Enter patient data (press Enter to use default value 0)")
    
    # Get important features interactively
    important_features = ['age', 'BDI_pre', 'FAA', 'coherence', 'alpha', 'theta', 'beta', 'gender']
    patient_data = {}
    
    for feature in important_features:
        if feature in feature_names:
            try:
                val = input(f"Enter {feature}: ")
                if val.strip():
                    patient_data[feature] = float(val)
                else:
                    patient_data[feature] = 0
                    print(f"   Using 0 for {feature}")
            except:
                patient_data[feature] = 0
                print(f"   Invalid input, using 0 for {feature}")
    
    # Fill remaining features with 0
    for feature in feature_names:
        if feature not in patient_data:
            patient_data[feature] = 0
    
    # Create DataFrame
    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df[feature_names]
    
    # Scale and predict
    patient_scaled = scaler.transform(patient_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Responder: {'✅ YES' if prediction == 1 else '❌ NO'}")
    print(f"Probability of response: {probability[1]:.2%}")
    print(f"Probability of non-response: {probability[0]:.2%}")
    print(f"Confidence: {max(probability):.2%}")
    
    # Show input values
    print("\n📋 Patient data used:")
    for feature in important_features:
        if feature in feature_names:
            print(f"   {feature}: {patient_data[feature]}")
    
    return prediction, probability

def batch_predict_folder(input_folder, output_folder=None):
    """
    Predict for all CSV files in a folder
    
    Parameters:
    -----------
    input_folder : str
        Folder containing patient CSV files
    output_folder : str, optional
        Folder to save results
    """
    
    import glob
    
    if output_folder is None:
        output_folder = "predictions"
    
    os.makedirs(output_folder, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files in {input_folder}")
    
    all_results = []
    
    for i, file in enumerate(csv_files):
        print(f"\n📊 Processing file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
        
        # Extract patient ID from filename
        patient_id = os.path.basename(file).replace('.csv', '')
        
        # Read patient data
        try:
            patient_df = pd.read_csv(file)
            
            # If file has multiple rows, take first row
            if len(patient_df) > 1:
                patient_df = patient_df.iloc[[0]]
            
            # Add patient ID
            patient_df['participants_ID'] = patient_id
            
            # Save temp file with single patient
            temp_file = os.path.join(output_folder, f"temp_{patient_id}.csv")
            patient_df.to_csv(temp_file, index=False)
            
            # Predict
            results = predict_patients_file(temp_file, os.path.join(output_folder, f"result_{patient_id}.csv"))
            
            if results is not None:
                all_results.append(results)
            
            # Clean up temp file
            os.remove(temp_file)
            
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(output_folder, "all_predictions.csv")
        combined.to_csv(combined_file, index=False)
        print(f"\n✅ Combined results saved to: {combined_file}")
        
        return combined
    
    return None

# ============================================================
# MAIN INTERACTIVE MENU
# ============================================================

def prediction_menu():
    """Main interactive menu for prediction"""
    
    print("\n" + "=" * 60)
    print("PATIENT RESPONSE PREDICTION MENU")
    print("=" * 60)
    
    while True:
        print("\n📋 Choose an option:")
        print("   1. Predict from CSV file (batch prediction)")
        print("   2. Predict single patient (interactive input)")
        print("   3. Predict all files in a folder")
        print("   4. View prediction statistics")
        print("   5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            file_path = input("Enter path to CSV file: ").strip()
            if os.path.exists(file_path):
                predict_patients_file(file_path)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == '2':
            predict_single_patient_interactive()
        
        elif choice == '3':
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                batch_predict_folder(folder_path)
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        elif choice == '4':
            # Show statistics from saved predictions
            pred_files = list(Path('.').glob('predictions_*.csv'))
            if pred_files:
                latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
                print(f"\n📊 Loading latest predictions: {latest_pred}")
                df = pd.read_csv(latest_pred)
                print(f"   Total patients: {len(df)}")
                print(f"   Responders: {(df['Responder_Predicted'] == 1).sum()} ({(df['Responder_Predicted'] == 1).mean()*100:.1f}%)")
                print(f"   Avg probability: {df['Probability_Response'].mean():.2%}")
            else:
                print("No prediction files found.")
        
        elif choice == '5':
            print("\n👋 Exiting prediction menu.")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

# Run the menu if this file is executed directly
if __name__ == "__main__":
    prediction_menu()