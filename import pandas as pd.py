import pandas as pd

# Load the dataset
file_path = 'MY.DENGUE.csv'
dengue_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
dengue_data.head()

# Check for missing values in the dataset
missing_values = dengue_data.isnull().sum()

# Feature Engineering: Convert dates to a more usable format
# We'll convert 'PeriodStartDate' to a datetime object and extract the year and month
# This assumes that the year and month are more relevant than the exact date for the prediction

dengue_data['PeriodStartDate'] = pd.to_datetime(dengue_data['PeriodStartDate'])
dengue_data['Year'] = dengue_data['PeriodStartDate'].dt.year
dengue_data['Month'] = dengue_data['PeriodStartDate'].dt.month

# Drop columns that may not be relevant or are redundant
# 'ConditionName', 'ConditionSNOMED', 'PathogenName', 'PathogenTaxonID', 'CountryName', 'CountryISO', 'PeriodEndDate'
# are dropped as they do not vary or are not relevant for the prediction

dengue_data_cleaned = dengue_data.drop(['ConditionName', 'ConditionSNOMED', 'PathogenName', 'PathogenTaxonID', 
                                        'CountryName', 'CountryISO', 'PeriodEndDate'], axis=1)

# Display the missing values and the first few rows of the updated dataset
missing_values, dengue_data_cleaned.head()

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# One-Hot Encoding for 'Admin1Name'
one_hot_encoder = OneHotEncoder(sparse=False)
admin1_encoded = one_hot_encoder.fit_transform(dengue_data_cleaned[['Admin1Name']])
admin1_encoded_df = pd.DataFrame(admin1_encoded, columns=one_hot_encoder.get_feature_names_out(['Admin1Name']))

# Normalization for numerical features
scaler = MinMaxScaler()
numerical_features = ['Weeks', 'Year', 'Month', 'Fatalities']
dengue_data_cleaned[numerical_features] = scaler.fit_transform(dengue_data_cleaned[numerical_features])

# Combine the one-hot encoded and normalized features
processed_data = pd.concat([dengue_data_cleaned.drop(['Admin1Name'], axis=1), admin1_encoded_df], axis=1)

# Display the first few rows of the processed dataset
processed_data.head()

from sklearn.model_selection import train_test_split

# Separate the features and the target variable
X = processed_data.drop(['CountValue', 'PeriodStartDate'], axis=1)  # Dropping 'PeriodStartDate' as we have extracted Year and Month
y = processed_data['CountValue']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the sizes of the training and testing sets
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the ANN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(64, activation='relu'))  # Hidden layer
model.add(Dense(32, activation='relu'))  # Another hidden layer
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Model Loss on Test Data: {loss}')
