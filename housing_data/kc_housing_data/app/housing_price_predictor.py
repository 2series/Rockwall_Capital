import pandas as pd
import streamlit as st
import plotly.express as px
from pycaret.regression import *

# Config
DIR_DATA = "../data/"
DIR_MODELS = "../pickled-models/"
MODEL = "cboost"

# Load dataset and model
df = pd.read_csv(DIR_DATA + "kc_house_data.csv")
model = load_model(DIR_MODELS + MODEL)
X_cols = model.named_steps["dtypes"].numerical_features

# Streamlit app title
st.title("House Price Predictor (King County, CA, US)")

# Since all the features are ints or floats we will just loop
# through them and add them as sliders
input_vars = []
for col in X_cols:
    if col != "trx_age":
        input_vars.append(
            st.slider(
                label=col,
                min_value=min(df[col]),
                max_value=max(df[col]),
                # Initiation with last instance from the DataFrame to demonstrate non-monotonic
                # behavior of cboost model where price decreases with grade for that this house
                value=int(df.loc[len(df) - 1, col]) if df.dtypes[col].name in (
                    "int32", "int64") else float(df.loc[len(df) - 1, col])
            )
        )
input_vars.append(0)

# Convert input_vars to dataframe to allow for PyCaret prediction
X = pd.DataFrame(input_vars).transpose()
X.columns = X_cols

# Model prediction
submit = st.button("Predict House Price")
if submit:
    prediction = model.predict(X)[0]
    # Format predicted price by converting to thousands or millions
    if prediction < 1e6:
        prediction_text = str(round(prediction / 1e3)) + "K"
    elif prediction >= 1e6:
        prediction_text = str(round(prediction / 1e6, 1)) + "M"
    st.write("USD    " + prediction_text)
