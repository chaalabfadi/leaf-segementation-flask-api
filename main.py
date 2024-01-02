from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
from flask_cors import CORS
from PIL import Image
import numpy as np
from skimage.transform import resize
from tensorflow.keras.models import load_model
import os
import logging

# Configure the logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
api = Api(app)
app.config['OUTPUT_FOLDER'] = 'outputs'

model_path = "models/unet_leaf_segmentation_final.h5"
model = load_model(model_path)


class LeafSegmentation(Resource):

    def post(self):
        # Handle image upload
        if 'image' not in request.files:
            logging.error("No file part")
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        image_path = "uploads/image.jpg"

        # Save the uploaded image
        logging.info('Image being uploaded: ' + image_path)
        image_file.save(image_path)
        logging.info('Image uploaded')
        # Perform leaf segmentation
        logging.info('Image is being processed.')
        segmented_mask = predict_leaf_disease(image_path)
        logging.info('Segmented mask is being processed.')

        segmented_mask_image_filename = "segmented_mask.png"
        segmented_mask_image_path = os.path.join(app.config['OUTPUT_FOLDER'], segmented_mask_image_filename)
        logging.info('Segmented mask')
        Image.fromarray(segmented_mask * 255).save(segmented_mask_image_path)

        # Get the full URL of the segmented mask image
        segmented_mask_image_url = request.url_root + 'outputs/' + segmented_mask_image_filename
        logging.info('Saved segmented mask')

        return jsonify({'segmented_mask_image_url': segmented_mask_image_url})


def predict_leaf_disease(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = resize(img, (128, 128), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    segmented_mask = (prediction > 0.5).astype(np.uint8)

    return segmented_mask.squeeze()


api.add_resource(LeafSegmentation, '/segment')


# Serve static files from the 'outputs' directory
@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
