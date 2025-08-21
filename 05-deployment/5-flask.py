import pickle
from flask import Flask
from flask import request
from flask import jsonify

with open('model1.bin', 'rb') as f_in: 
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in2:
    dict_vectorizer = pickle.load(f_in2)   
    
app = Flask('q4')
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dict_vectorizer.transform([client])
    y_pred = round(model.predict_proba(X)[0,1],3)
    score = {
        'probability' : float(y_pred)
    }
    return jsonify(score)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696