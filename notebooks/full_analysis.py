import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
import joblib

from utils import prepare_dataset, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from predict import predict_confirmation

X, y, df = prepare_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
    ('num', StandardScaler(), NUMERIC_FEATURES),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)),
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print('=== MODEL CALIBRATION (BRIER SCORE) ===')
bs = brier_score_loss(y_test, y_proba)
print(f'Brier Score: {bs:.4f}  (0=perfect, 0.25=random, lower=better)')
print()

# 5-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print('=== 5-FOLD CROSS-VALIDATION ===')
print(f'AUC  per fold: {cv_auc.round(4)}')
print(f'AUC  mean: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}')
print(f'ACC  per fold: {cv_acc.round(4)}')
print(f'ACC  mean: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}')
print()

# Overfitting check: train vs test AUC
train_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_proba)
test_auc = roc_auc_score(y_test, y_proba)
print('=== OVERFITTING CHECK ===')
print(f'Train AUC: {train_auc:.4f}')
print(f'Test  AUC: {test_auc:.4f}')
print(f'Gap: {abs(train_auc - test_auc):.4f}  (>0.05 = overfitting concern)')
print()

# Waitlisted-only test subset
waitlisted_mask = X_test['WL Number'] > 0
X_wl = X_test[waitlisted_mask]
y_wl = y_test[waitlisted_mask]
y_pred_wl = pipeline.predict(X_wl)
y_proba_wl = pipeline.predict_proba(X_wl)[:, 1]
cm = confusion_matrix(y_wl, y_pred_wl)
print(f'=== WAITLISTED-ONLY TEST SUBSET ({len(X_wl)} samples) ===')
print(f'AUC      : {roc_auc_score(y_wl, y_proba_wl):.4f}')
print(f'Accuracy : {accuracy_score(y_wl, y_pred_wl):.4f}')
print(f'Brier    : {brier_score_loss(y_wl, y_proba_wl):.4f}')
print(f'Confusion Matrix:\n{cm}')
print()

# Seat availability issue check
print('=== SEAT AVAILABILITY in WAITLISTED rows ===')
wl_mask_all = X['WL Number'] > 0
print(X[wl_mask_all]['Seat Availability'].value_counts().head())
print()

# Sanity predict tests
m = joblib.load('models/pnr_logistic_regression_pipeline.pkl')
test_cases = [
    dict(wl=20, days=120, seat=0,   label='WL20  120days seat=0'),
    dict(wl=20, days=120, seat=100, label='WL20  120days seat=100 [SHOULD BE SAME AS ABOVE]'),
    dict(wl=20, days=0,   seat=0,   label='WL20    0days seat=0'),
    dict(wl=100, days=0,  seat=0,   label='WL100   0days seat=0'),
    dict(wl=5,  days=90,  seat=0,   label='WL5    90days seat=0'),
    dict(wl=1,  days=5,   seat=0,   label='WL1     5days seat=0'),
    dict(wl=0,  days=30,  seat=100, label='WL0    30days seat=100 [NOT WAITLISTED]'),
]

print('=== SANITY: END-TO-END PROBABILITY TESTS ===')
for tc in test_cases:
    r = predict_confirmation(m, '3AC', 'General', 'Adult', 'IRCTC Website', 'Express', 'No', 'No',
                             2, 800, 10, 12, tc['seat'], tc['wl'], tc['days'])
    print(f"  {tc['label']}: {r['probability']:.2%}")

print()

# Feature importance via coefficients
clf = m.named_steps['classifier']
feature_names = m.named_steps['preprocessor'].get_feature_names_out()
coefs = clf.coef_[0]
ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
print('=== TOP 10 LOGISTIC REGRESSION COEFFICIENTS (by magnitude) ===')
for name, coef in ranked[:10]:
    print(f'  {coef:+.4f}  {name}')
