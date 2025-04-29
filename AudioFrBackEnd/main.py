import os
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Force Matplotlib to use a non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
CORS(app)
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 64
HOP_LENGTH = 512
MAX_TIME_STEPS = 109

# Class names
class_names = ["real", "fake"]

# Load the trained model
model_path = 'TESTSETAaudio_classifier.keras'
loaded_model = load_model(model_path)

def preprocess_audio(audio_file):
    """Loads and preprocesses an audio file into a mel spectrogram."""
    try:
        audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
    except Exception as e:
        return None, f"Error loading audio file: {e}"

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure fixed size
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    return mel_spectrogram, None

def plot_mel_spectrogram(mel_spectrogram):
    """Creates a mel spectrogram plot and encodes it as a base64 string."""
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def make_prediction(audio_file):
    """Processes the audio file, makes a prediction, and returns results."""
    mel_spectrogram, error = preprocess_audio(audio_file)
    if error:
        return None, None, error

    # Prepare input for model
    model_input = mel_spectrogram.reshape(1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)
    input_tensor = tf.convert_to_tensor(model_input, dtype=tf.float32)

    # Make prediction
    predictions = loaded_model(input_tensor)
    predicted_class = np.argmax(predictions.numpy(), axis=1)[0]
    predicted_label = class_names[predicted_class]

    # Generate spectrogram image
    spectrogram_image = plot_mel_spectrogram(mel_spectrogram)

    return predicted_label, spectrogram_image, None

@app.route("/predict", methods=["POST"])
def predict():
    """Handles file upload(s), processes them, and returns predictions."""
    files = request.files
    file1 = files.get("file1")
    file2 = files.get("file2")

    # Ensure at least one file is uploaded
    if not file1 and not file2:
        return jsonify({"error": "No files provided"}), 400

    results = {}

    # Process first file if provided
    if file1 and not file2:
        # If only one file, return "prediction" instead of "prediction1"
        prediction, spectrogram, error = make_prediction(file1)
        if error:
            return jsonify({"error": error}), 400
        return jsonify({"prediction": prediction, "spectrogram": spectrogram})

    # Process two files
    if file1:
        prediction1, spectrogram1, error1 = make_prediction(file1)
        if error1:
            return jsonify({"error": error1}), 400
        results["prediction1"] = prediction1
        results["spectrogram1"] = spectrogram1

    if file2:
        prediction2, spectrogram2, error2 = make_prediction(file2)
        if error2:
            return jsonify({"error": error2}), 400
        results["prediction2"] = prediction2
        results["spectrogram2"] = spectrogram2

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)