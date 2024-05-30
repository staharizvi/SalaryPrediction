# Salary Prediction Web App

This project is a simple web application that predicts salaries based on input features such as color, size, price, date of joining, and current date. It utilizes a RandomForestRegressor model trained on a dataset of salary information.

## Features

- **Input Form**: Users can input color, size, price, date of joining, and current date to get a salary prediction.
- **Prediction**: The app predicts the salary using a trained RandomForestRegressor model.
- **Model Training**: The model is trained on a dataset of salary information, performing feature engineering and one-hot encoding on categorical features.

## How to Use

1. Clone the repository to your local machine.
2. Install the necessary dependencies by running `pip install -r requirements.txt`.
3. Run the Streamlit app by executing `streamlit run app.py` in your terminal.
4. Fill out the input form with the required information and click the "Predict" button to see the salary prediction.

## File Structure

- `app.py`: Contains the Streamlit web application code.
- `model.pkl`: Pickled file containing the trained RandomForestRegressor model.
- `encoder.pkl`: Pickled file containing the trained OneHotEncoder object.
- `Salary Prediction of Data Professions.csv`: Dataset used for model training.
- `README.md`: This file.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- streamlit

## Credits

This project was created by Syed Taha Rizvi . Feel free to contribute or suggest improvements by submitting a pull request.
