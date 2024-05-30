import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

def readDf(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['FIRST NAME', 'LAST NAME'])
    df = df.dropna()
    return df

def typeCast(df):
    df['DOJ'] = pd.to_datetime(df['DOJ'])
    df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
    return df

def featTar(df):
    X = df.drop(columns=['SALARY'])  # Features
    y = df['SALARY']  # Target variable
    return X, y

def split_(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def featEng(X_train, X_test):
    X_train['SERVICE'] = (X_train['CURRENT DATE'] - X_train['DOJ']).dt.days
    X_test['SERVICE'] = (X_test['CURRENT DATE'] - X_test['DOJ']).dt.days
    X_train = X_train.drop(columns=['CURRENT DATE', 'DOJ'])
    X_test = X_test.drop(columns=['CURRENT DATE', 'DOJ'])
    return X_train, X_test

def encode(X_train, X_test):
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_test.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_test = X_test.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_test = pd.concat([num_X_test, OH_cols_valid], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_test.columns = OH_X_test.columns.astype(str)

    return OH_X_train, OH_X_test, OH_encoder

def train_model(OH_X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(OH_X_train, y_train)
    return model

# Function to make predictions
def make_predictions(model, OH_X_test):
    preds = model.predict(OH_X_test)
    return preds

# Function to evaluate the predictions
def evaluate_predictions(preds, y_test):
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return mape, r2, mae

if __name__ == "__main__":
    path = 'Salary Prediction of Data Professions.csv'
    X, y = featTar(typeCast(readDf(path)))
    X_train, X_test, y_train, y_test = split_(X, y)
    X_train, X_test = featEng(X_train, X_test)
    OH_X_train, OH_X_test, encoder = encode(X_train, X_test)

    # Save the encoder and model
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    model = train_model(OH_X_train, y_train)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    preds = make_predictions(model, OH_X_test)
    mape, r2, mae = evaluate_predictions(preds, y_test)
    print(f"MAE: {mae}")
    print(f"MAPE: {mape} %")
    print(f"R2: {r2*100} %")
