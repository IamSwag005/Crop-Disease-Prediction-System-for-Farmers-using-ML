import os
import cv2
import numpy as np
import pickle
import base64
from flask import (Flask, request, render_template, jsonify, flash,
                   redirect, url_for)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store the model and preprocessing objects
model = None
scaler = None
pca = None
label_encoder = None


def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, pca, label_encoder
    
    try:
        # Load the trained model
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing objects
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('models/pca.pkl', 'rb') as f:
            pca = pickle.load(f)
            
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        print("✅ Model and preprocessing objects loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please run train_model.py first to train and save the model.")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def preprocess_image(image_path):
    """Preprocess the uploaded image for prediction"""
    try:
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Resize to 64x64 (same as training)
        image = cv2.resize(image, (64, 64))
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Flatten the image
        image_flattened = image.flatten().reshape(1, -1)
        
        # Apply same preprocessing as training
        image_scaled = scaler.transform(image_flattened)
        image_pca = pca.transform(image_scaled)
        
        return image_pca
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def predict_disease(image_path):
    """Predict crop disease from image"""
    if model is None:
        return None, "Model not loaded"
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, "Error processing image"
    
    try:
        # Make prediction
        prediction = model.predict(processed_image)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_image)[0]
            confidence = np.max(probabilities) * 100
        else:
            confidence = 0
        
        # Decode the prediction
        disease_name = label_encoder.inverse_transform([prediction])[0]
        
        return {
            'disease': disease_name,
            'confidence': confidence,
            'prediction_code': int(prediction)
        }, None
        
    except Exception as e:
        return None, f"Error making prediction: {e}"


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, error = predict_disease(filepath)
        
        if error:
            flash(f'Error: {error}')
            os.remove(filepath)  # Clean up
            return redirect(request.url)
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html',
                               result=result,
                               image_data=img_data)
    
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(request.url)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Make prediction
    result, error = predict_disease(filepath)
    
    # Clean up uploaded file
    os.remove(filepath)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify(result)


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('predict'))


if __name__ == '__main__':
    # Load the model when starting the app
    if load_model():
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start app without model. "
              "Please run train_model.py first.")
