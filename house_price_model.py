import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load dataset
df = pd.read_csv("/Applications/ML/Codes/Tehran-housing-project/house.csv")

'''Data Preprocessing'''
dfc = df.dropna() # removing NAs
dfc = dfc.replace({True:1, False:0}) # replacing True/False with binary numbers

# Numerize the Addresses and sort them by the mean price of each Address
meanprice = dfc.groupby('Address')['Price'].mean().sort_values(ascending=True)

# Addresses with more expensive prices have bigger prices
Address_mapping = {Address: label for label, Address in enumerate(meanprice.index, start=1)}
dfc['Address'] = dfc['Address'].map(Address_mapping)

joblib.dump(Address_mapping, "Codes/address_encoder.pkl")
print("address saved")

# Cleaning Area columns and removing outlier items
dfc['Area'] = dfc['Area'].str.strip().str.replace(',', '').astype(float)
dfc = dfc[dfc['Area'] <= 500.0] 

'''Handeling Outliers'''
# Calculate the 10th and 90th percentiles of the price column to deine the upper and lower limit
lower_limit = np.percentile(dfc['Price'], 10)
upper_limit = np.percentile(dfc['Price'], 90)
dfc['Price'] = np.clip(dfc['Price'], lower_limit, upper_limit)

features = ['Address','Area', 'Room', 'Parking','Warehouse', 'Elevator']
X = dfc[features].values
y = dfc['Price'].values

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1 , random_state=2)

# Create polynomial regression pipeline
degree = 2
poly_model = make_pipeline(
    PolynomialFeatures(degree),
    StandardScaler(),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
y_test_pred = poly_model.predict(X_test[:, :6])
y_test = y_test

rss = np.mean(y_test - y_test_pred) ** 2
r2 = r2_score(y_test, y_test_pred)

print("Polynomial Regression Evaluation:")
print("Residual sum of squares: %.2f" % rss)
print('Variance score: %.2f' % r2)

joblib.dump(poly_model, "Codes/Tehran_house_price_model.pkl")
print("saved")