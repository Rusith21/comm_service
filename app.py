from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
artifacts = joblib.load('comm_score_model.pkl')
pipe, label_enc = artifacts['pipeline'], artifacts['label_encoder']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data] if isinstance(data, dict) else data)
    pred = pipe.predict(df)
    probs = pipe.predict_proba(df)
    return jsonify([
      {
        'prediction': label_enc.inverse_transform([p])[0],
        'probabilities': dict(zip(label_enc.classes_, prob))
      }
      for p, prob in zip(pred, probs)
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
