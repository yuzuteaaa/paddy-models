from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi folder untuk unggahan
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model_path = "model/paddy_model.h5"
model = load_model(model_path)

# Label untuk prediksi
class_labels = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)  # Konversi ke float
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        predicted_class, confidence = process_image(file_path)
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(confidence)  # Konversi ke float
        })
    return jsonify({'error': 'Invalid file'}), 400




# Fungsi untuk memproses gambar
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Prediksi gambar
            predicted_class, confidence = process_image(file_path)
            return render_template(
                'index.html',
                uploaded_image=file_path,
                prediction=predicted_class,
                confidence=confidence
            )
    return render_template('index.html')

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
