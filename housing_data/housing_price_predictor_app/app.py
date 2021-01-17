import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(
page_title='Supervised Ensemble Learning Model',
layout='wide'
)

#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    y = df.iloc[:,-1] # Selecting the last column as y

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write('Test set')
    st.info(y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('y variable')
    st.info(y.name)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=split_size
    )

    rf = RandomForestRegressor(
    n_estimators=parameter_n_estimators,
    max_features=parameter_max_features,
    random_state=parameter_random_state,
    criterion=parameter_criterion,
    min_samples_split=parameter_min_samples_split,
    min_samples_leaf=parameter_min_samples_leaf,
    bootstrap=parameter_bootstrap,
    oob_score=parameter_oob_score,
    n_jobs=parameter_n_jobs
    )

    rf.fit(X_train, y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_train, y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(y_train, y_pred_train))

    st.markdown('**2.2. Test set**')
    y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_test, y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(y_test, y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

#---------------------------------#
st.write("""
## Random Forest

The *RandomForestRegressor()* function is used to build a Multiple Linear Regression using the **Random Forest** algorithm

Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction

Try adjusting the hyperparameters!

""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload CSV file'):
    uploaded_file = st.sidebar.file_uploader("Upload input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/2series/Project-Apps/master/income_growth_App/data/income_growth_1980-2014.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Preview dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting CSV file to be uploaded')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #y = pd.Series(diabetes.target, name='response')
        #df = pd.concat([X, y], axis=1)

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name='response')
        df = pd.concat([X, y], axis=1)

        st.markdown('The Boston housing dataset is used as the example')
        st.write(df.head())

        build_model(df)
