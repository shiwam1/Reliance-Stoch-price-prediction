# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:51:40 2023

@author: shiwam
"""

import streamlit as st
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
import numpy as np
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsa_plots
from datetime import datetime


import yfinance as yf
##################   Loading the datasaet #########################
start='2015-01-01'
end='2022-12-31'
st.title('Reliance Stock Forecast App')

ticker="RELIANCE.NS"
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    data["Date"]=data["Date"].dt.date
    return data
	
data_load_state = st.text('Loading data...')
data = load_data(ticker)
data_load_state.text('Loading data... done!')
raw=st.button("First five Raw Data")
#st.subheader('Raw data')
if raw:
    st.write(data.head())
################## plotting #############################
visual=st.button("Visualizing the dataset")
if visual:
    def plot_raw_data():
    	fig = go.Figure()
    	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    	st.plotly_chart(fig)
    	
    plot_raw_data()

a=tsa_plots.plot_acf(data.Close,lags=12)#Upto 12 lag like y_t to y_(t-12)
def pacf():
    b=tsa_plots.plot_pacf(data.Close,lags=12)
    plt.title("PACF plot of Close price")
    st.pyplot(b)

pacf()
######################### Prediction##########################
train_data=st.button("Last 5 data which is mainly 2021 year")
if train_data:
    Train = data.head(1728)
    st.write(Train.tail())

test_data=st.button("first 5 data which is the last year")
if test_data:
    Test = data.tail(248)
    st.write(Test.tail())

n_years = st.slider('Choose dates of prediction as you want to forecsat the "cost" price:', 1, 31)
model_AR=AutoReg(data['Close'],lags=2).fit()

visual1=st.button("Visualizing the forecasted value")
if visual1:
    def fore(n_years):
        prediction_30_AR=model_AR.predict(1976,1976+n_years)
        df_30_pred_AR=pd.DataFrame(prediction_30_AR,columns=["values"])
        
        p=df_30_pred_AR["values"].values
        o=data["Close"]
    
        day_new=np.arange(1,1977)
        day_pred=np.arange(1977,1976+2+n_years)
        def plot_raw_data1():
        	fig1 = go.Figure()
        	fig1.add_trace(go.Scatter(x=day_new, y=o, name="Observed"))
        	fig1.add_trace(go.Scatter(x=day_pred, y=p, name="forecasted_day"))
        	fig1.layout.update(title_text='Time Series predicted data visualization with Rangeslider', xaxis_rangeslider_visible=True)
        	st.plotly_chart(fig1)
        plot_raw_data1()
    
    fore(n_years)


#model_AR=AutoReg(data['Close'],lags=2).fit()
#prediction_30_AR=model_AR.predict(1976,2006)    
#df_30_pred_AR=pd.DataFrame(prediction_30_AR,columns=["values"])
#p=df_30_pred_AR["values"].values
#o=data["Close"]
#day_new=np.arange(1,1977)
#day_pred=np.arange(1977,2008)


#def plot_raw_data1():
#	fig1 = go.Figure()
#	fig1.add_trace(go.Scatter(x=day_new, y=o, name="Observed"))
##	fig1.layout.update(title_text='Time Series predicted data visualization with Rangeslider', xaxis_rangeslider_visible=True)
	#st.plotly_chart(fig1)
	
####################### Next 30 days predicted dataset#####################
#dates_30=pd.date_range(start='1/1/2023', end='31/01/2023')
#predicted_next_30=pd.DataFrame({"Forecasted_30_AR":prediction_30_AR.values},index=dates_30)
#st.subheader("Next 30  days predicted dataset")
#st.write(predicted_next_30)

d=[]
for i in range(1,n_years+2):
    a=datetime(2023,1,i)

    b=datetime.timestamp(a)
    c=datetime.fromtimestamp(b)
    d.append(c)
prediction_30_AR=model_AR.predict(1976,1976+n_years)
z=pd.DataFrame(d,columns=["Next forecasted dates"])

K=pd.DataFrame(prediction_30_AR.values,columns=["Forecasted_close_values"])

dfa=pd.concat([z,K],axis=1)

forecasted=st.button("Forecasted values")
#st.subheader('Raw data')
if forecasted:
    st.write(dfa)

















