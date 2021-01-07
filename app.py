import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston



#--------- Model -----------#
# Model building
def build_model(df):
    st.markdown('----')
    st.write('Here are the details for building model...')
    
    # Seperate features and label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Data Splitting
    print(split_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100)
    
    # Show info to user
    st.markdown('**1.2. Data Splits**')
    st.write('Original Data Set')
    st.write(df.shape)
    st.write('Training Set')
    st.info(X_train.shape)
    st.write('Test Set')
    st.info(X_test.shape)
    
    st.markdown('**1.3. Variable Details**:')
    st.write('X variable (Feature)')
    st.info(list(X.columns))
    st.write('y variable (Label / Target)')
    st.info(y.name)
    
    # Model Creation and Training
    rf = RandomForestRegressor(n_estimators=parameter_n_estimator,
                               random_state=parameter_random_state,
                               max_features=parameter_max_features,
                               min_samples_split=parameter_min_samples_split,
                               min_samples_leaf=parameter_min_samples_leaf,
                               bootstrap=parameter_bootstrap,
                               oob_score=parameter_oob_score,
                               n_jobs=parameter_n_jobs)
    
    rf.fit(X_train, y_train)
    
    # Evaluating Model Performance
    st.subheader('2. Model Performance')
    
    # with Training Set
    st.markdown('**2.1. Training Set**')
    y_pred_train = rf.predict(X_train)
    
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_train, y_pred_train))
    
    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(y_train, y_pred_train))
    
    # with Test Set
    st.markdown('**2.2. Test Set**')
    y_pred_test = rf.predict(X_test)
    
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(y_test, y_pred_test))
    
    st.write('Error (MSE or MAE)')
    st.info(mean_squared_error(y_test, y_pred_test))
    
    # show using Model parameters
    st.subheader('3. Model Parameters')
    st.write(rf.get_params())
    
    # visualize the tree
    st.subheader('4. Tree Visualization')
    st.write('take a few seconds to load...')
    fig = plt.figure(figsize=(12,8), dpi=200)
    tree.plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)
    st.write(fig)
    
#------------------------------------------------------------------------------


#---------- Initial Page Setup ----------#
# Page layout with full width
st.set_page_config(page_title='Machine Learning App by PTY', layout='wide')


#------- Body Page -------------#
st.write("""
# Machine Learning App 

In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.

**Step 1**: Open sidebar to upload the file (or) Press to use Example Dataset

**Step 2**: Try adjusting the hyperparameters

*Note: Last column of csv will be used as Target/Label.*
""")


#--------- Side Bar ---------#
# Sidebar - collect user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])
    st.sidebar.markdown("""
                        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
                        """)

# Sidebar - Specify parameter settings
# slider (min, max, default value)
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split into (% for Training Set)', 10, 90, 80)

with st.sidebar.header('2.1. Learning Parameters'):
    parameter_n_estimator = st.sidebar.slider('Number of estimator (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max feature (max_feature)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.header('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
    

#--------- Main Body Panel -----------#
# Main Panel

# Display the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV to be uploaded...')
    
    example_datset = st.radio('Example Dataset to try:', ['Diabetes', 'Boston Housing'])
    
    if st.button('Press to use Example Dataset'):
        if example_datset == 'Diabetes':
            # Diabetes Dataset
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            y = pd.Series(diabetes.target, name='response')
            df = pd.concat([X, y], axis=1)
            
            st.markdown('Diabetes dataset is used as the example.')
            st.write(df.head())
            
            build_model(df)
        else:
            # Boston Housing Dataset
            boston = load_boston()
            X = pd.DataFrame(boston.data, columns=boston.feature_names)
            y = pd.Series(boston.target, name='response')
            df = pd.concat([X, y], axis=1)
            
            st.markdown('Boston housing dataset is used as the example.')
            st.write(df.head())
            
            build_model(df)





