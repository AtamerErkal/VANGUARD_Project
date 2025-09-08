# src/check_paths.py
import os

print("--- Path Check Script Running ---")

# This is the directory where this script (check_paths.py) is located.
# It should be the 'src' folder.
script_dir = os.path.dirname(__file__)
print(f"Script directory is: {os.path.abspath(script_dir)}")

# This is the relative path logic our app.py uses to find the 'models' folder.
# It goes up one level from 'src' and then into 'models'.
models_dir_path = os.path.join(script_dir, "..", "models")
print(f"The app is looking for the 'models' folder at this absolute path: {os.path.abspath(models_dir_path)}")

# Now, let's check if that path actually exists.
if os.path.isdir(models_dir_path):
    print("\n[SUCCESS] The 'models' folder was found at the expected location.")
else:
    print("\n[ERROR] The 'models' folder was NOT found at that location.")
    print("Please ensure your folder structure is correct ('src' and 'models' should be sibling folders).")

# Finally, let's check for a specific model file.
model_file_path = os.path.join(models_dir_path, 'vanguard_classifier.joblib')
print(f"\nThe app is looking for the model file at: {os.path.abspath(model_file_path)}")
if os.path.exists(model_file_path):
    print("[SUCCESS] The model file 'vanguard_classifier.joblib' was found!")
else:
    print("[ERROR] The model file was NOT found in the 'models' folder.")