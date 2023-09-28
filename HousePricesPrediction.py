import pandas as pd
#import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings 
warnings.filterwarnings('ignore')
import joblib
from sklearn.linear_model import LinearRegression
import pickle
data = pd.read_csv('train.csv')

# load the model 
model = pickle.load(open('house_price_prediction.pkl', 'rb'))

# --------------------- Streamlit Development Starts ---------------------------------
# st.markdown("h1 style = 'text-align: right; color: '#93B1A6'>START UP BUSINESS PREDICTOR</h1>, unsafe_allow_html = True")
# st.markdown("h6 style = 'top margin: 0rem; color: '#B9B4C7'>BUILT BY Gomycode Yellow Orange Beast</h1>, unsafe_allow_html = True")

st.title('HOUSE PRICE PREDICTION')
st.write('Built By Ugochukwu Obuninta')
st.markdown("<br> <br> <br>", unsafe_allow_html=True)

st.write('Please Enter Your Username')
username = st.text_input('Please enter username')
if st.button('Submit Name'):
    st.success(f"Welcome {username}. Pls enjoy your usage")
st.markdown("<br> <br>", unsafe_allow_html=True)

st.markdown("<br> <br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #A2C579'>Project Introduction</h1>", unsafe_allow_html = True)
st.markdown("<p style = 'text-align: justify; color: #AED2FF'>House price prediction is a crucial and highly sought-after task in the real estate industry and financial markets. It involves the use of advanced statistical and machine learning techniques to estimate the market value of residential properties. The goal of house price prediction is to provide valuable insights to homeowners, buyers, sellers, and investors, helping them make informed decisions in the dynamic and competitive housing market.</p>", unsafe_allow_html = True)

# heat_map = plt.figure(figsize = (14, 7))
# correlation_data = data[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']]
# sns.heatmap(correlation_data.corr(), annot = True, cmap ="BuPu")

# st.write(heat_map)

st.write('Unnamed: 0', axis = 1, inplace = True)
st.write(data.sample(10))

st.sidebar.write('Welcome')

# st.sidebar.image('c:\Users\MY LAPTOP\Downloads\pngwing.com (2).png', width = 300, caption= f"welcome {username}", use_column_width= True)

st.markdown("<br>", unsafe_allow_html= True)

st.sidebar.write('Please decide your variable input type')
input_style = st.sidebar.selectbox('Pick Your Preffered Input', ['Slider Input', 'Nunber Input'])

if input_style:
    fullsquare = st.sidebar.slider('full_sq', data['full_sq'].min(), data['full_sq'].max() )
    sub_area = st.sidebar.text_input('sub_area')
    num_room = st.sidebar.slider('num_room', data['num_room'].min(), data['num_room'].max() )
    kitch_sq = st.sidebar.slider('kitch_sq', data['kitch_sq'].min(), data['kitch_sq'].max() )
    healthcare_centers_raion = st.sidebar.slider('healthcare_centers_raion', data['healthcare_centers_raion'].min(), data['healthcare_centers_raion'].max() )
    
else:
    fullsquare = st.sidebar.slider('full_sq', data['full_sq'].min(), data['full_sq'].max() )
    sub_area = st.sidebar.text_input('sub_area')
    num_room = st.sidebar.slider('num_room', data['num_room'].min(), data['num_room'].max() )
    kitch_sq = st.sidebar.slider('kitch_sq', data['kitch_sq'].min(), data['kitch_sq'].max() )
    healthcare_centers_raion = st.sidebar.slider('healthcare_centers_raion', data['healthcare_centers_raion'].min(), data['healthcare_centers_raion'].max() )

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'full_sq': fullsquare, 'sub_area': sub_area, "num_room": num_room, "kitch_sq": kitch_sq, "healthcare_centers_raion": healthcare_centers_raion}])
st.write(input_var)

st.markdown("<br>", unsafe_allow_html= True)

from sklearn.preprocessing import StandardScaler, LabelEncoder
def transformer(dataframe):
    lb = LabelEncoder()
    scaler = StandardScaler()

    # dep = dataframe.drop('price_doc' ,axis = 1)
    for i in dataframe:
        if i in dataframe.select_dtypes(include='number').columns:
            # Scale only numerical columns
            dataframe[[i]] = scaler.fit_transform(dataframe[[i]])
        elif i in dataframe.select_dtypes(include=['object', 'category']).columns:
            # Label encode categorical columns
            dataframe[i] = lb.fit_transform(dataframe[i])

    return dataframe

transformer(input_var)

prediction = model.predict(input_var)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with tab2:
    st.subheader('Model Interpretation')
    st.success(f'The price of the house is ${prediction[0]}')