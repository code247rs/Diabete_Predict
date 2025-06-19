import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
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
    if feature == 'Gender':
        entry = ttk.Combobox(root, values=['M', 'F'], state='readonly')
        entry.current(0)
    else:
        entry = tk.Entry(root, width=30)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = entry

# Prediction function
def predict():
    try:
        #values = [float(entries[feat].get()) for feat in feature_names]
        print("Predicting...")
        values = []
        for feat in feature_names:
            if feat == 'Gender':
                entries[feat].get()
                values.append(0 if entries[feat].get() == 'M' else 1)
            else:
                values.append(float(entries[feat].get()))
        print("Values:")
        scaler = load('scaler.joblib')#

        print("Scaler loaded")
        input_array = np.array(values).reshape(1, -1)
        print("Input array:")
        sc_input_array = scaler.transform(input_array) #
        print("Scaled input array:")

        prediction = model.predict(sc_input_array)[0]
        print("Prediction:")

        # Map numeric prediction to class name
        class_map = {
            0: "Non-Diabetic",
            1: "Pre-Diabetic",
            2: "Diabetic"
        }
        print("Class map:")
        result = class_map.get(prediction, "Unknown")
        messagebox.showinfo("Prediction Result", f"The model predicts: {result}")
        print("Prediction result shown in messagebox")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers in all fields.")



# Predict Button
btn = tk.Button(root, text="Predict", command=predict, width=20, background="blue", foreground="white", activebackground="darkblue", activeforeground="white")
#btn.configure(bg="blue", fg="white", activebackground="darkblue", activeforeground="white")
btn.grid(row=len(feature_names), column=0, columnspan=2, pady=20)

# Run the GUI loop
root.mainloop()

    







