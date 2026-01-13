import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

KEEP_COLUMNS = [
    #Significant predictors (Coeff > 0.10)
    'age', 'avg_daily_uv', 'skin_tone', 'family_history', 
    'immunosuppressed', 'number_of_lesions', 'sunscreen_freq',
    'sunburns_last_year', 
    
    #Moderate Predictors (Coeff > 0.02)
    'outdoor_job', 'skin_photosensitivity', 'tanning_bed_use',
    'clothing_protection', 'hat_use', 'lesion_size_mm',
    'lesion_location', 'years_lived_at_address',
    
    #Borderline but had specific flags/levels selected
    'occupation', 'favorite_cuisine','urban_rural', 'income'
]

train_raw = pd.read_csv('SkinCancerTrain.csv')
test_raw = pd.read_csv('SkinCancerTestNoY.csv')

train_raw['is_train'] = True
test_raw['is_train'] = False
test_raw['Cancer'] = np.nan 

df = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)


important_flags = [
    'sunburns_last_year', 'number_of_lesions', 'avg_daily_uv', 
    'zip_code_last_digit', 'monthly_screen_time_minutes'
]

for col in important_flags:
    if col in df.columns:
        df[f'{col}_missing_flag'] = df[col].isnull().astype(int)

flag_cols = [c for c in df.columns if '_missing_flag' in c]
final_cols = ['ID', 'Cancer', 'is_train'] + KEEP_COLUMNS + flag_cols

available_cols = [c for c in final_cols if c in df.columns]
df = df[available_cols].copy()

num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

#Categorical NAs (Flagged as "Unknown")
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols = [c for c in cat_cols if c not in ['ID', 'Cancer']]
df[cat_cols] = df[cat_cols].fillna("Unknown")

#Label Encoding for XGBoost
for col in cat_cols:
    df[col] = df[col].astype('category').cat.codes

df['target'] = df['Cancer'].apply(lambda x: 1 if x == 'Malignant' else 0)

feature_cols = [c for c in df.columns if c not in ['ID', 'Cancer', 'target', 'is_train']]

X_full = df[df['is_train'] == True][feature_cols]
y_full = df[df['is_train'] == True]['target']
X_test_submission = df[df['is_train'] == False][feature_cols]

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

#Best hyperparameters from grid search
best_params = {
    'max_depth': 4,
    'learning_rate': 0.01,
    'min_child_weight': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.6
}


#Split data 80/20 for validation
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_full, y_full, test_size=0.2, random_state=101, stratify=y_full
)


#Train with early stopping to find optimal n_estimators
tuned_model = xgb.XGBClassifier(
    n_estimators=10000,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=1.0,
    reg_lambda=2.0,
    gamma=1.0,
    eval_metric='error', 
    early_stopping_rounds=250,
    random_state=101
)

tuned_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
    verbose=100
)



val_probs_tuned = tuned_model.predict_proba(X_val_final)[:, 1]

print(f"Best iteration: {tuned_model.best_iteration}")

best_threshold_tuned = 0.50
best_acc_tuned = 0
for threshold in np.arange(0.40, 0.60, 0.01):
    preds = (val_probs_tuned > threshold).astype(int)
    acc = accuracy_score(y_val_final, preds)
    if acc > best_acc_tuned:
        best_acc_tuned = acc
        best_threshold_tuned = threshold



val_preds_tuned = (val_probs_tuned > best_threshold_tuned).astype(int)
cm_tuned = confusion_matrix(y_val_final, val_preds_tuned)


#Retrain on 100% of data
best_n_estimators_tuned = tuned_model.best_iteration

final_tuned_model = xgb.XGBClassifier(
    n_estimators=best_n_estimators_tuned,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=1.0,
    reg_lambda=2.0,
    gamma=1.0,
    eval_metric='error',
    random_state=101
)

final_tuned_model.fit(X_full, y_full, verbose=False)

final_tuned_probs = final_tuned_model.predict_proba(X_test_submission)[:, 1]
final_tuned_preds = np.where(final_tuned_probs > best_threshold_tuned, "Malignant", "Benign")

#Create submission file
submission_final = pd.DataFrame({
    'ID': test_raw['ID'],
    'Cancer': final_tuned_preds
})

submission_final.to_csv('submission_final_tuned.csv', index=False)

#Final summary
print(f"File: 'submission_final_tuned.csv'")
print(f"Samples: {len(submission_final)}")
print(f"\nValidation Performance:")
print(f"  Accuracy: {best_acc_tuned:.4f}")
