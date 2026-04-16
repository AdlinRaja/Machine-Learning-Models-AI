import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model= genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_summary(result_df):
    
    prompt=f''' you are a data scientist expert,
    Here are the model results:
    {result_df.to_string()}
    1. Identify the best model
    2. Explain why it is best
    3. Summarize the performance of the models'''
    
    response = model.generate_content(prompt)
    return response.text

def suggest_improvements(results_df):
    prompt = f''' You are a data scientist expert.
    Here are the model results :
    {results_df.to_string()}
    
    Suggest:
    - Ways to improve the model performance
    - Hyperparameter tuning and give range od values in each parameter
    - Better suitable algorithms for the given data
    - Data preprocessing improvements'''
    
    response = model.generate_content(prompt)
    return response.text