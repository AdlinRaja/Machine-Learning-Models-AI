import numpy as np
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,accuracy_score,precision_score,recall_score,f1_score

# to get methods from analysis.py

from analysis import suggest_improvements,generate_summary

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model= genai.GenerativeModel('gemini-2.5-flash-lite')

st.set_page_config('ML-Model',page_icon='🕵🏻',layout='wide')

st.title(' Machine Learning Models ֎🇦🇮')
st.subheader('App that compares the results of ML model using AI and provide suggestions for improvement 🔍')

uploaded_file = st.sidebar.file_uploader("Upload your CSV file 📤", type=["csv"])

if uploaded_file is not None:
    st.markdown('### Preview 📊')
    
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    target=st.sidebar.selectbox(':blue[Select target column]',df.columns)
    
    st.sidebar.write(f':red[Target Variable:]{target}')
    
    if target:
        x=df.drop(columns=[target]).copy()
        y=df[target].copy()
        
        num=x.select_dtypes(include='number').columns.to_list()
        cat=x.select_dtypes(exclude='number').columns.to_list()
        
        # st.write(f':green[Numeric Variable:]{num}')
        # st.write(f':green[Categoric Variable:]{cat}')
        
        # Missing value treatment
        
        x[num]=x[num].fillna(x[num].median())
        x[cat]=x[cat].fillna('Missing data')
        
        # Encoding
        
        x=pd.get_dummies(data=x,columns=cat,drop_first=True,dtype=int)
        
        # for categoric target
        
        if y.dtype=='object':
            label=LabelEncoder()
            y=label.fit_transform(y)
            
        # Detect the problem type
            
        if ((df[target].dtype=='object') or len(np.unique(y))<=20):
            prob='Classification'
        else:
            prob='Regression'
        
        st.sidebar.write(f':green[Problem Type:]{prob}')
        
        # Train Test Split
        
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
        
        # scaling
        # fit transform on train data
        # transform in test data
        
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)
            
        # Models
        # ***************
        results=[]
        if prob=='Regression':
            models={'Linear Regression':LinearRegression(),
                    'Random Forest':RandomForestRegressor(random_state=42),
                    'Gradien Boosting':GradientBoostingRegressor(random_state=42)}
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model Name':name,
                               'R2 score':round(r2_score(ytest,ypred),2),
                               'MSE':round(mean_squared_error(ytest,ypred),2),
                               'RMSE':round(root_mean_squared_error(ytest,ypred),2)})
        
        else:
            models={'Logistic Regression':LogisticRegression(),
                    'Random Forest':RandomForestClassifier(random_state=42),
                    'Gradien Boosting':GradientBoostingClassifier(random_state=42)}
            for name,model in models.items():
                model.fit(xtrain,ytrain)
                ypred=model.predict(xtest)
                
                results.append({'Model Name':name,
                               'Accuracy score':round(accuracy_score(ytest,ypred),2),
                               'Precision':round(precision_score(ytest,ypred,average='weighted'),2),
                               'Recall':round(recall_score(ytest,ypred,average='weighted'),2),
                               'f1-score':round(f1_score(ytest,ypred,average='weighted'),2)})
        
        results_df=pd.DataFrame(results)
        st.write('## :green[Results:]')
        st.dataframe(results_df)
        
        if prob=='Regression':
            st.bar_chart(results_df.set_index('Model Name')['R2 score'])
            st.bar_chart(results_df.set_index('Model Name')['RMSE'])
        else:
            st.bar_chart(results_df.set_index('Model Name')['Accuracy score'])
            st.bar_chart(results_df.set_index('Model Name')['f1-score'])
            
        if st.button('Generate Summary'):
            summary=generate_summary(results_df)
            st.write(summary)
            
        if st.button('Suggest improvements'):
            improvements=suggest_improvements(results_df)
            st.write(improvements)
        






