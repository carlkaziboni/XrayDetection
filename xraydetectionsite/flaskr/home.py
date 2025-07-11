from flask import Blueprint, render_template, request, redirect, url_for, flash
from flaskr.db import get_db
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import cv2

bp = Blueprint('home', __name__, url_prefix='/')

@bp.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page."""
    return render_template('home/index.html')

@bp.route('/upload', methods=['POST'])
def upload():
    """Handle file upload."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.getcwd(),'flaskr','static', 'uploads', filename)
        file.save(file_path)
        
        # Here you would typically process the uploaded file
        # For example, save the path to the database
        db = get_db()
        db.execute('INSERT INTO xray_detections (image_path, findings) VALUES (?, ?)', (file_path, ''))
        db.commit()
        
        flash('File successfully uploaded')

        # Predict findings using the detection model
        import flaskr.detection as detection
        findings = detection.get_predictions(file_path)
        print(f"Predicted findings for {filename}: {findings}")

        # Get image array for Grad CAM
        img = cv2.imread(file_path)
        img_array = cv2.resize(img, (224, 224))
        img_array = img_array / 255.0
        img_array = img_array.reshape((1, 224, 224, 3))

        # Produce Grad CAM heatmap
        heatmap = detection.get_grad_cam_heatmap_transfer(img_array)
        
        # overlay heatmap on the original image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlayed_image = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)

        # Save the overlayed image
        overlayed_image_path = os.path.join(os.getcwd(), 'flaskr','static', 'uploads', f'overlayed_{filename}')
        cv2.imwrite(overlayed_image_path, overlayed_image)

        return render_template('home/result.html', filename=filename, findings=findings, overlayed_image_path=overlayed_image_path)
    
    flash('File upload failed')
    return redirect(request.url)