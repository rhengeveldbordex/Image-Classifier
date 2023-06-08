import os
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained InceptionV3 model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Create Flask application
app = Flask(__name__)

# Function to classify images
def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Return the top predictions as a list of dictionaries
    results = []
    for _, label, probability in decoded_predictions:
        results.append({'label': label, 'probability': probability * 100})

    return results

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No image file uploaded')

        image_file = request.files['image']

        # Check if a file was selected
        if image_file.filename == '':
            return render_template('index.html', error='No file selected')

        # Check if the file is valid
        if image_file and allowed_file(image_file.filename):
            # Save the uploaded file to a temporary location
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Classify the image
            results = classify_image(image_path)

            # Return the results to the user
            return render_template('index.html', results=results, image_file=image_file.filename)

    return render_template('index.html')

# Function to check if the file extension is allowed
def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Define the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
