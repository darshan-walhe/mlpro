import sys
sys.path.insert(0, 'src')
import joblib
from predict import predict_confirmation

m = joblib.load('models/pnr_logistic_regression_pipeline.pkl')

cases = [
    ('WL20  120days [should be HIGH]',   '3AC','Tatkal','Adult','IRCTC Website','Superfast','No','No', 2,800,10,12, 20,120),
    ('WL20  0days   [should be MED]',    '3AC','Tatkal','Adult','IRCTC Website','Superfast','No','No', 2,800,10,12, 20,0),
    ('WL100 0days   [should be LOW]',    '3AC','General','Adult','IRCTC Website','Express','No','No',  2,800,10,12,100,0),
    ('WL5   90days  [should be VERY HIGH]','3AC','General','Adult','IRCTC Website','Express','No','No',2,800,10,12,  5,90),
    ('WL15  120days [original case]',    '2AC','General','Adult','IRCTC Website','Express','No','No',  2,800,10,12, 15,120),
    ('WL15  0days   [same, no advance]', '2AC','General','Adult','IRCTC Website','Express','No','No',  2,800,10,12, 15,0),
    ('WL1   5days   [very low WL]',      '3AC','General','Adult','IRCTC Website','Express','No','No',  1,500,5,8,    1,5),
    ('WL180 0days   [should be ~0%]',    '3AC','General','Adult','IRCTC Website','Express','Yes','No', 4,800,10,12,180,0),
]

print('=== FINAL SANITY CHECK ===')
for row in cases:
    label = row[0]
    args = row[1:]
    r = predict_confirmation(m, *args)
    print(f'  {label}: {r["probability"]:.0%}  ({r["label"]})')

# Also verify top coefficients
clf = m.named_steps['classifier']
feat_names = m.named_steps['preprocessor'].get_feature_names_out()
coefs = clf.coef_[0]
ranked = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)
print()
print('=== TOP COEFFICIENTS ===')
for name, coef in ranked[:8]:
    print(f'  {coef:+.4f}  {name}')
