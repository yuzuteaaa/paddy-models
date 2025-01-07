from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import joblib
from flask_sqlalchemy import SQLAlchemy
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Setup koneksi ke MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/api_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Folder upload
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan kategori
model_path = "model/model_normal.pkl"
model = joblib.load(model_path)

Categories = ['blast', 'blight', 'tungro', 'normal']

class APIKey(db.Model):
    __tablename__ = 'api_key'
    id = db.Column(db.Integer, primary_key=True)
    key_name = db.Column(db.String(100), unique=True, nullable=False)
    key_value = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f"APIKey('{self.key_name}', '{self.key_value}')"

# Buat database
with app.app_context():
    db.create_all()

# Fungsi utilitas
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(img_path):
    img = imread(img_path)
    img_resized = resize(img, (150, 150, 3))
    img_flattened = img_resized.flatten().reshape(1, -1)
    predicted_class = model.predict(img_flattened)[0]
    probabilities = model.predict_proba(img_flattened)[0]

    predicted_label = Categories[predicted_class]
    confidence = float(max(probabilities) * 100)
    return predicted_label, confidence

# Routes Flask
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class, confidence = process_image(file_path)
            return render_template(
                'index.html',
                uploaded_image=file_path,
                prediction=predicted_class,
                confidence=confidence
            )
    return render_template('index.html')

@app.route('/get_api_keys', methods=['GET'])
def get_api_keys():
    api_keys = APIKey.query.all()
    keys = {key.key_name: key.key_value for key in api_keys}
    return jsonify(keys)

@app.route('/update_api_key', methods=['GET', 'POST'])
def update_api_key_form():
    if request.method == 'POST':
        key_name = request.form['key_name']
        new_key_value = request.form['key_value']

        if not key_name or not new_key_value:
            return jsonify({'error': 'Key name and new key value are required'}), 400

        api_key = APIKey.query.filter_by(key_name=key_name).first()

        if api_key:
            api_key.key_value = new_key_value
            db.session.commit()
            return jsonify({'message': f'API key {key_name} updated successfully'}), 200
        else:
            return jsonify({'error': 'API key not found'}), 404

    return render_template('update_api_key.html')

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
            'confidence': float(confidence)
        })
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
