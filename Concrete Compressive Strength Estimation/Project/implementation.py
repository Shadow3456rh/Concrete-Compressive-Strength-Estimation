import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import joblib
import numpy as np

loaded_model = joblib.load("concrete_strength_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

def calculate_strength():
    try:
        cement = float(entry_cement.get())
        slag = float(entry_slag.get())
        fly_ash = float(entry_fly_ash.get())
        water = float(entry_water.get())
        superplasticizer = float(entry_superplasticizer.get())
        coarse_aggregate = float(entry_coarse_aggregate.get())
        fine_aggregate = float(entry_fine_aggregate.get())
        age = float(entry_age.get())

        input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
        input_data_scaled = loaded_scaler.transform(input_data)

        strength = loaded_model.predict(input_data_scaled)[0]

        result_label.config(text=f"Concrete compressive strength: {strength:.2f} MPa", bg="lightblue")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for all fields.")


root = tk.Tk()
root.title("Concrete Compressive Strength Calculator")
root.geometry("1200x600")
root.configure(bg="white")

bg_image = Image.open("image.png")
bg_image = bg_image.resize((1200, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg="white", bd=5, relief="groove")
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.7, anchor='n')
frame.config(bg="#f0f0f0") 

fields = ['Cement (kg)', 'Blast Furnace Slag (kg)', 'Fly Ash (kg)', 'Water (kg)', 
          'Superplasticizer (kg)', 'Coarse Aggregate (kg)', 'Fine Aggregate (kg)', 'Age (days)']
entries = {}


for i, field in enumerate(fields):
    label = tk.Label(frame, text=field, font=("Arial", 14), bg="#f0f0f0", fg="black")
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
    
    entry = tk.Entry(frame, font=("Arial", 14), bg="#ffffff", fg="black")
    entry.grid(row=i, column=1, padx=10, pady=5)
    
    entries[field] = entry

entry_cement = entries['Cement (kg)']
entry_slag = entries['Blast Furnace Slag (kg)']
entry_fly_ash = entries['Fly Ash (kg)']
entry_water = entries['Water (kg)']
entry_superplasticizer = entries['Superplasticizer (kg)']
entry_coarse_aggregate = entries['Coarse Aggregate (kg)']
entry_fine_aggregate = entries['Fine Aggregate (kg)']
entry_age = entries['Age (days)']

submit_button = tk.Button(root, text="Submit", font=("Arial", 16), command=calculate_strength, bg="blue", fg="white")
submit_button.place(relx=0.5, rely=0.85, anchor="center")

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="lightblue", fg="black")
result_label.place(relx=0.5, rely=0.9, anchor="center")


root.mainloop()
