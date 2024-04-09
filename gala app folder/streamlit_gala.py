import streamlit as s
import pandas as pd
import numpy as np
from datetime import date
import warnings
import joblib
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
#scikitlearn-1.4.Post
warnings.filterwarnings("ignore")
import time

# web page image and title
s.set_page_config(page_title='Gala Grocery Store',layout='wide')
s.markdown("<h1 style='text-align: center;'>Gala Grocery Store - Streamlit web App</h1>",unsafe_allow_html=True)
s.image('g4.jpg',caption='Gala Grocery Store')


#features
cats=['baby products', 'baked goods', 'baking' ,'beverages' ,'canned foods',
 'cheese' ,'cleaning products', 'condiments and sauces' ,'dairy', 'frozen',
 'fruit' ,'kitchen' ,'meat', 'medicine' ,'packaged foods' ,'personal care',
 'pets' ,'refrigerated items' ,'seafood', 'snacks' ,'spices and herbs',
 'vegetables']
cus=['standard', 'premium', 'basic', 'non-member', 'gold']
pay=['cash', 'e-wallet', 'credit card', 'debit card']
w=["Monday", "Tuesday", "Wednesday", "Thursday", 'Friday', 'Saturday', 'Sunday']


#labels
le=LabelEncoder()

#category
category = s.selectbox("**Select Category 🍲**", options=cats)
le.fit(cats)
s.write(f"You selected: {category}")
E_category=le.transform([category])[0]

#Customer type
Cus = s.radio('**Customer Type 👤**', options=cus, key="customer_type", horizontal=True)
le.fit(cus)
s.write(f"Customer type 👤 : {Cus}")
E_cus=le.transform([Cus])[0]

#unit price
Unit_price = s.number_input('**Enter the Unit Price 💵**', min_value=1, max_value=21,value=None, placeholder='Type the unit price')
s.write(f"The unit price: 💵 {Unit_price}")
Unit_price=np.sqrt(Unit_price)

#Quantity 
Quantity = s.slider('**Enter Quantity 🛒**', min_value=1, max_value=4, value=1)
s.write(f"Quantity 🛒 {Quantity}")
Quantity=np.sqrt(Quantity)

#Total
Total = s.number_input('**Enter the Total 💰**', min_value=1.0, max_value=60.0,step=0.25)
s.write(f"Total amount 💰 : {Total}")
Total=np.sqrt(Total)

#Payment method
Payment = s.radio('**Payment Method 💳**', options=pay, key='payment_method', horizontal=True)
le.fit(pay)
s.write(f"Payment method 💳 : {Payment}")
E_pay=le.transform([Payment])[0]

#Temperature
Temp = s.number_input('**Enter the Temperature 🌡️**', step=0.04,min_value=-4.0, max_value=4.0, placeholder='Temperature (Celisius)')
s.write(f"Temperature 🌡️ : {Temp}")
Temp=np.sqrt(Temp+31)

# days of week:    Conversion of week to numbers
def w2n(Week):
    ww = ["Monday", "Tuesday", "Wednesday", "Thursday", 'Friday', 'Saturday', 'Sunday']
    return ww.index(Week)

w = ["Monday", "Tuesday", "Wednesday", "Thursday", 'Friday', 'Saturday', 'Sunday']
Week = s.radio('**Day of Week 📅**', options=w, key=None, horizontal=True)
s.write(f"Selected Day 📅: {Week}")
Weeky=w2n(Week)

#days of month
dday=s.number_input('**Enter the day of Month🌞**',max_value=7,min_value=1)
s.write(f'Day 🌞 : {dday}')

#hour of day
Hour = s.slider('**Time in Hours ⏰**', min_value=9, max_value=19, step=1)
s.write(f"Time ⏰ : {Hour} hours")
Hour=np.sqrt(Hour)

#square roots:[Y,'unit_price', 'quantity', 'total', 'temperature','hour']
# x:['category', 'customer_type', 'unit_price', 'quantity', 'total',
#        'payment_type', 'temperature', 'week', 'day', 'hour']
def Model_pred():
    mod1=joblib.load('histgrdbstreg.joblib')
    mod2=joblib.load('adabstreg.joblib')

    f=[E_category,E_cus,Unit_price,Quantity,Total,E_pay,Temp,Weeky,dday,Hour]
    pred1 = mod1.predict([f])
    pred2 = mod2.predict([f])
    return pred1[0], pred2[0]


if s.button('Predict'):
    with s.spinner('Predicting...'):
        time.sleep(2)  # Simulate a computation or processing time
        p1,p2 = Model_pred()
    
    s.markdown(f"**The Estimated Average Stock Percentage in Gala Store:**", unsafe_allow_html=True)
    s.markdown(f"**HistGradientRegressor: {p1*100:.2f}%**", unsafe_allow_html=True)
    s.markdown(f"**BaggingRegressor: {p2*100:.2f}%**", unsafe_allow_html=True)


