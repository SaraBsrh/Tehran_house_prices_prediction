import joblib
import numpy as np

model = joblib.load("Codes/Tehran_house_price_model.pkl")
address = joblib.load("Codes/address_encoder.pkl")

# Check if the model has a predict method
if not hasattr(model, "predict"):
    raise ValueError("Loaded model does not have a 'predict' method. Make sure you saved a trained scikit-learn model.")

def get_user_input():
    print("Please enter the details of the house:")

    Area = float(input("Enter area (sqft): "))
    Room = int(input("Enter number of bedrooms: "))
    Warehouse = int(input("Does it have Warehouse? (1 for Yes, 0 for No): "))
    Elevator = int(input("Does it have Elevator? (1 for Yes, 0 for No): "))
    Parking = int(input("Does it have parking? (1 for Yes, 0 for No): "))
    Address = (input("Enter the name of area(e.g. 'Shahran', 'Narmak', ect.): "))

    if Address in address:
        Address_num = address[Address]
    else:
        print("Warning: Address not found in dataset.(PAY ATTENTION the first letter must be CAPITAL.)")
    
    # Convert to numpy array for prediction
    user_data = np.array([[Address_num,Area, Room, Parking,Warehouse, Elevator]])

    return user_data

# Get input from user
user_input = get_user_input()

# Predict price
predicted_price = model.predict(user_input)

# Show result
print(f"Estimated House Price: {predicted_price[0]:,.2f}")