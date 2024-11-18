# app.py

# Import necessary libraries
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Set image dimensions
IMG_HEIGHT, IMG_WIDTH = 227, 227

# Load the trained model
model = tf.keras.models.load_model('best_model')

# Ensure the 'uploads' folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")


# Function to prepare an uploaded image for prediction
def prepare_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Define the main route for uploading and predicting
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file:
            try:
                # Save the uploaded image temporarily
                image_path = os.path.join("uploads", image_file.filename)
                image_file.save(image_path)
                print(f"File saved to {image_path}")

                # Prepare and predict the image
                image = prepare_image(image_path)
                prediction = model.predict(image)
                os.remove(image_path)  # Clean up the uploaded file after prediction

                # Interpret the prediction
                label = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
                return jsonify({"result": label})

            except Exception as e:
                print("Error during processing:", e)
                return jsonify({"result": "Error processing image."}), 500
        else:
            print("No file received in the POST request.")
            return jsonify({"result": "Error: No file uploaded."}), 400

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
