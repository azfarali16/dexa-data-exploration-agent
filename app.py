from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import pandas as pd
import uuid
import json
import google.generativeai as genai
from utils.data_processor import DataProcessor
import datetime
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

active_datasets = {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/analyze',methods=['POST'])
def test():
    print(request.files)
    print('HEAVY')

    return redirect(url_for('analytics')) 


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        # Generate unique ID for this dataset
        dataset_id = str(uuid.uuid4())
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}_{file.filename}")
        file.save(file_path)
        
        # Process the dataset
        try:
            processor = DataProcessor(file_path)
            data_summary = processor.get_summary()
            
            # Store for later use
            active_datasets[dataset_id] = {
                'file_path': file_path,
                'processor': processor,
                'filename': file.filename
            }
            
            return jsonify({
                'success': True,
                'dataset_id': dataset_id,
                'summary': data_summary
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload CSV or Excel files.'}), 400

@app.route('/dashboard/<dataset_id>')
def dashboard(dataset_id):
    if dataset_id not in active_datasets:
        return render_template('index.html', error="Dataset not found")
    
    dataset = active_datasets[dataset_id]
    return render_template('dashboard.html', 
                          dataset_id=dataset_id, 
                          filename=dataset['filename'])

@app.route('/api/data-summary/<dataset_id>')
def get_data_summary(dataset_id):
    if dataset_id not in active_datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    processor = active_datasets[dataset_id]['processor']
    summary = processor.get_summary()
    
    # Generate initial visualizations
    
    return jsonify({
        'summary': summary,
    })


if __name__ == '__main__':
    app.run(debug=True) 