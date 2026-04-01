from flask import Flask, render_template, request, jsonify
import os
from pathlib import Path
from Arrangr import audio_to_chords_and_melody, arrange
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processing = False
status_message = "Ready"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global processing, status_message
    
    if processing:
        return jsonify({'error': 'Already processing a file. Please wait.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        return jsonify({'error': 'File must be MP3 or WAV format'}), 400
    
    try:
        processing = True
        status_message = "Uploading..."
        
        # Save uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        status_message = "Analyzing audio..."
        
        # Process in background thread
        def process():
            global status_message
            try:
                status_message = "Extracting chords and melody..."
                chord_prog, melody = audio_to_chords_and_melody(filepath)
                
                if not chord_prog or not melody:
                    raise ValueError("Could not extract musical information")
                
                status_message = "Creating SATB arrangement..."
                score = arrange(chord_prog, melody)
                
                output_filename = os.path.splitext(file.filename)[0] + "_arranged.musicxml"
                output_path = os.path.join(UPLOAD_FOLDER, output_filename)
                score.write("musicxml", fp=output_path)
                
                status_message = f"✓ Success! File: {output_filename}"
                return True, output_filename
            except Exception as e:
                status_message = f"✗ Error: {str(e)}"
                return False, str(e)
        
        success, result = process()
        processing = False
        
        if success:
            return jsonify({'success': True, 'filename': result, 'message': status_message})
        else:
            return jsonify({'error': result}), 500
    
    except Exception as e:
        processing = False
        status_message = f"✗ Error: {str(e)}"
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    global status_message
    return jsonify({'status': status_message, 'processing': processing})

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return open(filepath, 'rb'), 200, {'Content-Disposition': f'attachment; filename={filename}'}
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("Starting Arrangr Web GUI...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=False, host='localhost', port=5000)
