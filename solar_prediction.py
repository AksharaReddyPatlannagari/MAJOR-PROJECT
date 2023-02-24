import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xg
import urllib.request
import sys
import pandas as pd
from datetime import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def obj_to_DT(s,x):
    s.loc[:,x] = s[x].astype('string')
    s[x]= pd.to_datetime(s[x],dayfirst=True)
    return s

def dataSetProcessing(date):
    final=pd.read_csv("finalData.csv")
    final.dropna(inplace=True)
    final.drop('Unnamed: 0',axis=1,inplace=True)
    final.rename(columns = {'Date':'no_of_days'}, inplace = True)
    final.drop(['DC total voltage (v)','DC total current(A)','AC total voltage (v)','windspeed','precip','windgust','no_of_days','sealevelpressure'],inplace=True,axis=1)
    res=model_fitting(final,date)
    return(str(res))

def model_fitting(final,date):
    X=final.drop('Daily Generation (Active)(kWh)',axis=1)
    Y=final['Daily Generation (Active)(kWh)']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,shuffle=False)

    """
    kmeans=KMeans(n_clusters=5)
    x_train["Cluster"]=kmeans.fit_predict(x_train[['temp', 'dew', 'humidity', 'winddir',
       'cloudcover', 'visibility']])
    x_test["Cluster"]=kmeans.fit_predict(x_test[['temp', 'dew', 'humidity', 'winddir',
       'cloudcover', 'visibility']])
    x_train=x_train.drop(['temp', 'dew', 'humidity', 'winddir','cloudcover', 'visibility'],axis=1)
    x_test=x_test.drop(['temp', 'dew', 'humidity', 'winddir','cloudcover', 'visibility'],axis=1)
    linearReg=def_linearReg(x_train,y_train,x_test,y_test)
    decisionTreeReg=def_decionTreeReg(x_train,y_train,x_test,y_test)
    gradientBoostReg=def_gradientBoostReg(x_train,y_train,x_test,y_test)
    """

    xgBoostReg=def_xgBoost(x_train,y_train,x_test,y_test)
    get_future_weather()
    pred_result=get_solar_output(date)
    label_encoder = LabelEncoder()
    pred_result['Time']= label_encoder.fit_transform(pred_result['Time'])
    pred_result=pred_result[['Time','temp', 'dew', 'humidity', 'winddir','cloudcover', 'visibility','solarradiation']]
    pred=pred_result

    """
    pred["Cluster"]=kmeans.fit_predict(pred_result[['temp', 'dew', 'humidity', 'winddir','cloudcover', 'visibility']])
    pred=pred.drop(['temp', 'dew', 'humidity', 'winddir','cloudcover', 'visibility'],axis=1)
    """
    print('XGBoost Reg with r2 score: ',xgBoostReg[0])
    sol=xgBoostReg[1].predict(pred)
    print(sol[-1])
    return sol[-1]
        

def def_linearReg(x_train,y_train,x_test,y_test):
    l=LinearRegression()
    l.fit(x_train,y_train)
    y_predict=l.predict(x_test)
    r2_lr=r2_score(y_true=y_test,y_pred=y_predict)
    print('r2 score for linear regression: ',r2_lr)
    return [r2_lr,l]


def def_decionTreeReg(x_train,y_train,x_test,y_test):
    dt=DecisionTreeRegressor(max_depth=3)
    dt.fit(x_train,y_train)
    y_predict1=dt.predict(x_test)
    r2_dt=r2_score(y_true=y_test,y_pred=y_predict1)
    print('r2 score for decision tree regression: ',r2_dt)
    return [r2_dt,dt]

def def_gradientBoostReg(x_train,y_train,x_test,y_test):
   gb= GradientBoostingRegressor(n_estimators = 200, max_depth = 3, random_state = 1)
   gb.fit(x_train,y_train)
   y_predict2=gb.predict(x_test) 
   r2_gb=r2_score(y_true=y_test,y_pred=y_predict2)
   print('r2 score for gradient boost regression: ',r2_gb)
   return [r2_gb,gb]

def def_xgBoost(x_train,y_train,x_test,y_test):
    xgb= xg.XGBRegressor(objective ='reg:linear',n_estimators = 10, seed = 123)
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_test)
    r2_xgb=r2_score(y_true=y_test,y_pred=pred)
    print('r2 score for xgboost regression: ',r2_xgb)
    return [r2_xgb,xgb]

def get_future_weather():        
    try: 
        ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/17.4116%2C%2078.3987/next30days?unitGroup=metric&include=hours&key=768FEXAYRHYQKK5678HWL4LSM&contentType=csv")
        
        """
    
        https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/17.4116%2C%2078.3987/next7days?unitGroup=metric&include=hours&key=HTXNURB3NT3SGJHKHUBK63QGK&contentType=csv
    
        """
        # Parse the results as CSV
        CSVText = pd.read_csv(ResultBytes)
        CSVText.to_csv('predicted_weather.csv')

    except urllib.error.HTTPError  as e:
        ErrorInfo= e.read().decode() 
        print('Error code: ', e.code, ErrorInfo)
        sys.exit()
    except  urllib.error.URLError as e:
        ErrorInfo= e.read().decode() 
        print('Error code: ', e.code,ErrorInfo)

def get_solar_output(date):
    pred_weather=pd.read_csv("predicted_weather.csv")
    pred_weather=pred_weather.rename(columns = {'datetime':'Time'})
    pred_weather=obj_to_DT(pred_weather,'Time')
    pred_weather=pred_weather[pred_weather['Time'].apply(lambda x: x.time()) < time(19,00,00)]
    pred_weather=pred_weather[pred_weather['Time'].apply(lambda x: x.time()) >= time(6,00)]
    pred_result=[]
    for i in range(len(pred_weather)):
        text= pred_weather.iloc[i]
        txt=str(text['Time'])
        txt=txt.split(' ')
        if(txt[0] == date):
            pred_result.append(pred_weather.iloc[i])
    print(pred_result)
    pred_result=pd.DataFrame(pred_result)
    return(pred_result)
    