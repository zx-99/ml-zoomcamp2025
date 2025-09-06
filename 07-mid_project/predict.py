import pickle
import pandas as pd   
from flask import Flask
from flask import request
from flask import jsonify 

with open('./models/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask('project')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        patient = request.get_json()
        if not patient:
            return jsonify({'error': 'No input data provided'}), 400
        X = pd.DataFrame([patient])
        X.columns = X.columns.str.lower().str.replace(' ', '_')    
        y_pred = model.predict_proba(X)[0,1]
        score = {'probability': float(y_pred)}
        return jsonify(score)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# def predict():
#     patient = request.get_json()
#     y_pred = model.predict_proba(pd.DataFrame([patient]))[0,1]
#     score = {'probability': float(y_pred)}
#     return jsonify(score)

# def load_data(data_path: str) -> pd.DataFrame:
#     df = pd.read_csv(data_path)
#     df.columns = df.columns.str.lower().str.replace(' ', '_')
#     df = df.drop(columns='id')
#     return df

# def main():

#     df_test = load_data('./data/test.csv')
#     X_test = df_test
    
#     sample_submission = pd.read_csv('./data/sample_submission.csv')
    
#     y_pred_test = model.predict_proba(X_test)[:,1]
#     sample_submission['smoking'] = y_pred_test
#     sample_submission.to_csv(f'submission.csv', index=False)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)