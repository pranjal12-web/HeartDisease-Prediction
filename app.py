from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import cherrypy
# from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('./heartDisease.pkl')

def convert_to_numeric(df, columns):
    for col_name in columns:
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

def fill_based_cat(data, columns, based_cat, metric):
    data = data.copy()
    for col in columns:
        data[col] = data[col].fillna(data.groupby(based_cat)[col].transform(metric))
    return data

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_outliers(dataframe, num_list):
    for col in num_list:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
        dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

def one_hot_encode(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def scale_numeric_features(dataframe, num_cols, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe, scaler


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    features = [float(x) for x in request.form.values()]
    
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([features], columns=['Age','Sex', 'cp','trestbps', 'chol',  'fbs','restecg','thalch', 'exang','oldpeak',  'slope', 'ca', 'thal'])
    
    columns_to_convert = ['trestbps', 'chol',  'fbs','restecg','thalch', 'exang','oldpeak',  'slope', 'ca', 'thal']
    convert_to_numeric(input_data, columns_to_convert)

    # Create 'Age_Group' column
    input_data['Age_Group'] = pd.cut(input_data['Age'], bins=[-1, 30, 50, float('inf')], labels=['Young Adults', 'Middle-Aged', 'Seniors'])
    
    missing_ones = input_data.isnull().sum()[input_data.isnull().sum() > 0].index
    input_data = fill_based_cat(input_data, missing_ones, based_cat="Age_Group", metric="median")

    num_list = ['Age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'Sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    cat_list = ['Age_Group']  
    
    # Replace outliers
    replace_outliers(input_data, num_list)
    
    # One-hot encode categorical features
    input_data = one_hot_encode(input_data, categorical_cols=['Age_Group'])
    
    # Scale numeric features
    input_data,scaler = scale_numeric_features(input_data, num_cols=num_list, scaler=None)
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Return the prediction
    return render_template('index.html', prediction_text=f'Predicted Outcome: {prediction[0]}')

if __name__ == '__main__':
    # app.run(debug=True)

     # Run the app using Gunicorn
    # import os
    # host = '0.0.0.0'
    # port = int(os.environ.get('PORT', 5000))
    # workers = int(os.environ.get('WEB_CONCURRENCY', 1))
    # command = f'gunicorn -w {workers} -b {host}:{port} app:app'
    # os.system(command)

    # host = '0.0.0.0'
    # port = int(os.environ.get('PORT', 5000))
    # serve(app, host=host, port=port)

    host = '127.0.0.1'
    port = int(os.environ.get('PORT', 5000))
    cherrypy.config.update({'server.socket_host': host, 'server.socket_port': port})
    cherrypy.tree.graft(app, '/')
    cherrypy.engine.start()
    cherrypy.engine.block()

    
