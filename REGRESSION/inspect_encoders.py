import joblib
import os

# Prefer the existing file produced by regression pipeline
script_dir = os.path.dirname(__file__)
default_path = os.path.join(script_dir, "regression_encoders.pkl")
fallback_path = os.path.join(script_dir, "regression_curve_encoders.pkl")

if os.path.exists(default_path):
    file_path = default_path
elif os.path.exists(fallback_path):
    file_path = fallback_path
else:
    print(f"Error: Neither {default_path} nor {fallback_path} was found.")
    raise SystemExit(1)

print(f"Using encoder file: {file_path}")
print(f"--- Inspecting: {os.path.basename(file_path)} ---")
try:
    # Load the binary encoders
    encoders = joblib.load(file_path)
    
    # Encoders is usually a dictionary of LabelEncoder objects
    for col, le in encoders.items():
        print(f"\nFeature: {col}")
        # print first 10 classes to avoid overwhelming output
        classes = le.classes_
        print(f"  Classes ({len(classes)} total): {classes[:10]} {'...' if len(classes) > 10 else ''}")

except Exception as e:
    print(f"Could not read the pickle file: {e}")
