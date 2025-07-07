import os
import traceback
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Factory to create Flask app
def create_app():
    app = Flask(__name__)

    # Load ML model with error logging
    try:
        model_path = os.path.join(os.getcwd(), 'comm_score_model.pkl')
        artifacts = joblib.load(model_path)
        pipe, label_enc = artifacts['pipeline'], artifacts['label_encoder']
        app.logger.info(f'Model loaded from {model_path}')
    except Exception:
        app.logger.error('Model load failed:\n' + traceback.format_exc())
        raise

    # Health check endpoint
    @app.route('/', methods=['GET'])
    def health():
        return 'OK', 200

    # Prediction endpoint
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json(force=True)
            df = pd.DataFrame([data])
            pred = pipe.predict(df)
            probs = pipe.predict_proba(df)
            return jsonify([
                {
                    'prediction': label_enc.inverse_transform([pred[0]])[0],
                    'probabilities': dict(zip(label_enc.classes_, probs[0]))
                }
            ])
        except Exception:
            app.logger.error('Prediction failed:\n' + traceback.format_exc())
            return jsonify({'error': 'Internal server error'}), 500

    return app

# Entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app = create_app()
    app.logger.info(f'Starting server on port {port}')
    app.run(host='0.0.0.0', port=port)
