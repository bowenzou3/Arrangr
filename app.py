from flask import Flask, render_template, request, jsonify
import os
import traceback
import uuid
from pathlib import Path
from Arrangr import audio_to_chords_and_melody, arrange
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

jobs = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.lower().endswith('.mp3'):
        return jsonify({'error': 'File must be MP3 format'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'Queued',
        'processing': True,
        'message': 'Uploading...',
        'filename': None,
        'error': None,
    }

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    def process_job(job_id, filepath, original_name):
        try:
            jobs[job_id]['status'] = 'Extracting chords and melody'
            jobs[job_id]['message'] = 'Extracting chords and melody...'

            chord_prog, melody, tempo_bpm, key_sig = audio_to_chords_and_melody(filepath)
            if not chord_prog or not melody:
                raise ValueError('Could not extract musical information from audio')

            jobs[job_id]['status'] = 'Creating SATB arrangement'
            jobs[job_id]['message'] = 'Creating SATB arrangement...'

            # Extract song title from filename
            title = os.path.splitext(original_name)[0]
            
            score = arrange(chord_prog, melody, title=title, key_signature=key_sig, tempo_bpm=tempo_bpm)
            output_filename = os.path.splitext(original_name)[0] + '_arranged.musicxml'
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            score.write('musicxml', fp=output_path)

            jobs[job_id].update({
                'status': 'Completed',
                'processing': False,
                'message': f'✓ Success! File: {output_filename}',
                'filename': output_filename,
            })
        except Exception as e:
            err_text = f'{type(e).__name__}: {e}\n' + traceback.format_exc()
            print('[ERROR] upload processing failed:\n' + err_text)
            jobs[job_id].update({
                'status': 'Failed',
                'processing': False,
                'message': f'✗ Error: {str(e)}',
                'error': err_text,
            })

    thread = threading.Thread(target=process_job, args=(job_id, filepath, file.filename), daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'Processing', 'message': 'File accepted and is being processed.'}), 202

@app.route('/status')
def get_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'error': 'Missing job_id parameter'}), 400

    job = jobs.get(job_id)
    if job is None:
        return jsonify({'error': 'Invalid job_id'}), 404

    return jsonify(job)

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return open(filepath, 'rb'), 200, {'Content-Disposition': f'attachment; filename={filename}'}
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════╗")
    print("║      Arrangr — SATB + Soloist A Cappella         ║")
    print("║   MP3 → Melody · Harmony · Chords · MusicXML    ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Upload an MP3 and get a full SATB arrangement   ║")
    print("║  Solo · Soprano · Alto · Tenor · Bass            ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Running at → http://localhost:5000              ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("Starting Arrangr Web GUI...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=False, host='localhost', port=5000)
