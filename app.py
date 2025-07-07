import os
   from flask import Flask, request, jsonify
   import joblib, pandas as pd

   app = Flask(__name__)
   artifacts = joblib.load('comm_score_model.pkl')
   pipe, label_enc = artifacts['pipeline'], artifacts['label_encoder']

   # Health check endpoint
   @app.route('/', methods=['GET'])
   def health():
       return 'OK', 200

   # Prediction endpoint
   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       df = pd.DataFrame([data])
       pred = pipe.predict(df)
       probs = pipe.predict_proba(df)
       return jsonify([
           {
             'prediction': label_enc.inverse_transform([pred[0]])[0],
             'probabilities': dict(zip(label_enc.classes_, probs[0]))
           }
       ])

   if __name__ == '__main__':
       # Bind to the PORT environment variable provided by Railway
       port = int(os.environ.get('PORT', 5000))
       app.run(host='0.0.0.0', port=port)
