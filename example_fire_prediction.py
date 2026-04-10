import pickle
import pandas as pd

# Load the model
with open('fire_model_selector.pkl', 'rb') as f:
    model_data = pickle.load(f)

model         = model_data['model']
scaler        = model_data['scaler']
feature_names = model_data['feature_names']

# INPUT: fire scenario parameters
df = pd.DataFrame({
    'Fuel_load':        [420],    # MJ/m²
    'Floor_area':       [70],     # m²
    'Thermal_inertia':  [423.5],  # J/(m²·s^0.5·K)
    'Opening_factor':   [0.08],   # m^0.5
    'building_height':  [3.5],    # m
})

# PROCESS: Scale and predict
X        = df[feature_names].values
X_scaled = scaler.transform(X)
proba    = model.predict_proba(X_scaled)

# OUTPUT: Probability of which fire model should be used
#   Class 0 = Eurocode Parametric Fire (EN)
#   Class 1 = Travelling Fire (TF)
en_pct = proba[0][0] * 100
tf_pct = proba[0][1] * 100

print(f"{'EN (Eurocode Parametric Fire)':<30} {en_pct:.1f}%")
print(f"{'TF (Travelling Fire)':<30} {tf_pct:.1f}%")
