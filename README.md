Tehran House Price Prediction

Project Overview

This machine learning project aims to predict house prices in Tehran, Iran, using Polynomial Regression (88% accuracy) and Linear Regression (82% accuracy). Since house pricing is often unreliable and inconsistent, this model provides a data-driven approach to estimate house prices based on key features. Additionally, it analyzes which regions in Tehran are the most expensive.

Dataset

The dataset contains 3,479 house records with 8 columns:
	•	Address – Region in Tehran
	•	Area – Size of the house (sqft)
	•	Room – Number of bedrooms
	•	Parking – Availability of parking (Yes → 1, No → 0)
	•	Warehouse – Availability of a warehouse (Yes → 1, No → 0)
	•	Elevator – Availability of an elevator (Yes → 1, No → 0)
	•	Price – House price in Iranian Rial
	•	Price (USD) – House price converted to USD

Data Preprocessing
	•	Handled missing values
	•	Converted categorical features (e.g., Parking, Warehouse, Elevator → Binary: 1/0)
	•	Numerized addresses based on mean price ranking of each region
	•	Removed outliers in the Area column
	•	Cleaned numerical columns (e.g., removing spaces or commas)

Key Insights & Visualizations
	•	Ranking of Tehran’s areas from most to least expensive
	•	Feature importance analysis (identifying the most influential factors in pricing)
	•	Histograms of key features
	•	Scatter plot of Price vs. Area (with size variations based on Address)

Model Performance

Model: Polynomial Regression
MSE: 0.02
R² Score:  0.88

Model: Linear Regression	 82%
MSE: 0.04
R² Score: 0.83


Evaluation Metrics:
	•	Mean Squared Error (MSE)
	•	R² Score

Usage

1. Install Dependencies

Ensure you have all required libraries installed:

pip install -r requirements.txt

2. Running the Project

View Step-by-Step Analysis & Visualizations

Open and run the Jupyter Notebook:

jupyter notebook TehranHousePrices.ipynb

Predict House Prices
	1.	Train the model (run once before prediction):

python house_price_model.py


	2.	Predict a specific house price:

python house_predict.py

	•	Enter house details when prompted.
	•	The model will return the estimated price.

Project Structure

📂 Tehran-House-Price-Prediction
│── 📄 TehranHousePrices.ipynb       # Full analysis and visualizations
│── 📄 house_price_model.py          # Model training & saving
│── 📄 house_predict.py              # User input & price prediction
│── 📄 requirements.txt              # Dependencies
│── 📂 Codes/  
│   ├── Tehran_house_price_model.pkl # Saved model  
│   ├── address_encoder.pkl          # Address mapping  
│── 📂 Data/  
│   ├── Tehran_house_prices.csv      # Dataset  

Conclusion
	•	Polynomial Regression performed better than Linear Regression in this dataset.
	•	Area significantly influences house prices in Tehran.
	•	The model provides a practical tool for estimating house prices based on key features.
