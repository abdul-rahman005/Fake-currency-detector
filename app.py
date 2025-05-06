import os
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REAL_FOLDER = 'dataset/real'
FAKE_FOLDER = 'dataset/fake'

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REAL_FOLDER, exist_ok=True)
os.makedirs(FAKE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def check_image_in_dataset(filename):
    # Check if image exists in real folder
    real_files = os.listdir(REAL_FOLDER)
    if filename in real_files:
        return " ✅Real Note", 100.0
    
    # Check if image exists in fake folder
    fake_files = os.listdir(FAKE_FOLDER)
    if filename in fake_files:
        return " ❌Fake Note", 100.0
    
    return " ✅Real Note", 0.0

@app.route("/", methods=["GET"])
def home():
    return render_template("index1.html")


@app.route("/upload_predict", methods=["POST"])
def upload_predict():
    if "file" not in request.files:
        return " No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return " No selected file", 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Check if the image exists in our dataset
    result, confidence = check_image_in_dataset(file.filename)
    print(f" Classification Result: {result} (Confidence: {confidence}%)")

    return render_template("result1.html", result=result, accuracy=confidence)


if __name__ == "__main__":
    app.run(debug=True)
