from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from detection_engine import analyze_media

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return "Deepfake Detection API Running ✅"

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files["file"]
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only JPG, PNG, WEBP, BMP, TIFF are supported."}), 400

        # Secure the filename and save it temporarily for the detection engine
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Send the file path and file type/mimetype to the new engine
        result = analyze_media(filepath, file.mimetype)
        
        if "error" in result:
             return jsonify(result), 400

        # Clean up the file to save disk space
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as cleanup_err:
            print(f"Cleanup error: {cleanup_err}")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)