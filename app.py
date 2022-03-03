from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('delhi_cleaned.csv')
pipe = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def index():
    locality = sorted(data['locality'].unique())
    seller_type = sorted(data['seller_type'].unique())
    layout_type = sorted(data['layout_type'].unique())
    property_type = sorted(data['property_type'].unique())
    furnish_type = sorted(data['furnish_type'].unique())
    return render_template('index.html', locality=locality, seller_type=seller_type, layout_type=layout_type, property_type=property_type, furnish_type=furnish_type)


@app.route('/predict', methods=['POST'])
def predict():
    locality = request.form.get('locality')
    seller_type = request.form.get('seller_type')
    layout_type = request.form.get('layout_type')
    property_type = request.form.get('property_type')
    furnish_type = request.form.get('furnish_type')
    bedroom = request.form.get('bedroom')
    area = request.form.get('area')
    # print(locality,seller_type,layout_type,property_type,furnish_type,bedroom,area)
    input = pd.DataFrame([[locality,seller_type,layout_type,property_type, furnish_type,float(bedroom), float(area)]],columns=['locality','seller_type','layout_type','property_type', 'furnish_type','bedroom', 'area'])
    # print(input)
    prediction = pipe.predict(input)[0]
    print(prediction)
    return str(np.round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True)