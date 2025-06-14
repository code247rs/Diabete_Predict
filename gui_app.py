import tkinter as tk
from tkinter import messagebox
from joblib import load
import numpy as np

# Load the model
model = load('diabetes_model.joblib')

# Feature labels (must match model input order)
feature_names = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

# Create the main window
root = tk.Tk()
root.title("Diabetes Predictor")
root.geometry("400x600")

entries = {}

# Add entry fields dynamically
for idx, feature in enumerate(feature_names):
    label = tk.Label(root, text=feature)
    label.grid(row=idx, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(root, width=30)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = entry

# Prediction function
def predict():
    try:
        values = [float(entries[feat].get()) for feat in feature_names]
        input_array = np.array(values).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # Map numeric prediction to class name
        class_map = {
            0: "Non-Diabetic",
            1: "Pre-Diabetic",
            2: "Diabetic"
        }
        result = class_map.get(prediction, "Unknown")
        messagebox.showinfo("Prediction Result", f"The model predicts: {result}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers in all fields.")

# Predict Button
btn = tk.Button(root, text="Predict", command=predict, bg="blue", fg="white", width=20)
btn.grid(row=len(feature_names), column=0, columnspan=2, pady=20)

# Run the GUI loop
root.mainloop()



# Predict Button
btn = tk.Button(root, text="Predict", command=predict, width=20)
btn.configure(bg="blue", fg="white", activebackground="darkblue", activeforeground="white")
btn.grid(row=len(feature_names), column=0, columnspan=2, pady=20)

# Run the GUI loop
root.mainloop()

    







