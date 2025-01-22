from flask import Flask, request, jsonify, render_template
import os
import torch
from werkzeug.utils import secure_filename
from imple_sys import LCNN, process_audio_file, preprocess_audio, evaluate_single_audio_sample

app = Flask(__name__)

UPLOAD_FOLDER = 'data_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inisialisasi model
model = LCNN()
model.load_state_dict(torch.load('LCNN CQT.pth'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Proses file audio menggunakan model
        audio_tensor = preprocess_audio(filepath)
        audio_tensor = audio_tensor.unsqueeze(0)  # Tambahkan batch dimension
        
        model.eval()
        with torch.no_grad():
            output = model(audio_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

        # Hasil: 1 untuk AI, 0 untuk manusia
        result = "AI" if predicted_label == 1 else "Manusia"
        return jsonify({"result": result})
    return jsonify({"error": "File processing error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
