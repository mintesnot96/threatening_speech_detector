
# Multimodal Threatening Speech Detector

## Overview
The Multimodal Threatening Speech Detector is a web application designed to identify threatening speech within text, audio, and video inputs. Utilizing machine learning and the pretrained OpenAI Whisper model, this project provides a comprehensive analysis tool for detecting harmful content. Developed with Flask for deployment on the Render platform and Jupyter Notebook for preprocessing and model training, this application integrates two main models: a text-based threat detector and an audio/video processor that transcribes inputs using Whisper before analysis.

## Features
- Text input analysis for threatening speech detection.
- Audio and video input analysis via transcription with OpenAI's Whisper model.
- Results displayed on a user-friendly web interface.
- Deployment on Render for easy access and usage.

## Installation

### Prerequisites
- Python 3.10.9
- Flask
- Keras
- TensorFlow
- Whisper
- Other dependencies listed in `requirements.txt`.

### Clone the Repository
```bash
git clone https://github.com/mintesnot96/threatening_speech_detector.git
cd multimodal-threat-detector
```

### Setup Virtual Environment
```bash
python -m venv venv
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### Install Requirements
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
python app.py
```
This will start the Flask server, making the web application accessible on `http://127.0.0.1:5000/`.

### Using the Application
1. Navigate to `http://127.0.0.1:5000/` in your web browser.
2. Input threatening text or upload an audio/video file to be analyzed.
3. Submit the input for processing. The application will display whether threatening speech was detected.

## Interface Preview
![Screenshot 1](screanshots/Code%20Screenshot%202024-02-25%20162143%20.png)
![Screenshot 2](screanshots/UI1%20Screenshot%202024-02-25%20162355.png)
![Screenshot 1](screanshots/UI%202%20Screenshot%202024-02-25%20162503.png)
![Screenshot 2](screanshots/UI%203%20Screenshot%202024-02-25%20162645.png)

## Development and Contributions
- The project was developed using Flask for the web application framework and Jupyter Notebook for data preprocessing and model development.
- Contributions to the project are welcome. Please refer to the contribution guidelines for more information.

## Acknowledgments
- This project utilizes the OpenAI Whisper model for audio and video transcription.
- The threat detection model was trained with a custom dataset and developed using Keras and TensorFlow.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
