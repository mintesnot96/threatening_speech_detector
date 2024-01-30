from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pandas as pd
import keras
import pickle
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import re
# from nltk.corpus import stopwords
import string
import nltk
stemmer = nltk.SnowballStemmer("english")
# stopword=set(stopwords.words('english'))
import whisper
whisper_model = whisper.load_model("base")
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import os
import moviepy.editor as mp
from moviepy.editor import VideoFileClip

stopword={'on', 'we', 'ours', "won't", 'shouldn', 'does', 'an', "it's", 'whom', 'under', 'his', 'ain', "aren't", "weren't", 'during', "mustn't", 'should', 'won', 'some', 'further', "doesn't", "hasn't", "don't", 'shan', 'she', 'which', 'ourselves', 'of', 'our', 'all', 'their', 'each', 'weren', 'just', 'were', 'm', 'for', "needn't", "you're", 'd', 'mustn', 'as', 'her', 'over', 'very', 'theirs', 'in', 'me', 'himself', 'so', 'up', "haven't", 'with', 'above', 'mightn', 'about', 'having', 'most', 'doing', 'been', 'itself', 'isn', "mightn't", 'at', 'against', "wouldn't", 'this', 'into', 'until', 'such', 'am', 'needn', 'haven', 'ma', "shouldn't", "wasn't", 'a', 't', "couldn't", 'between', 'o', "hadn't", 'they', 'no', 'him', 'when', 'll', 'why', 'after', 're', 'wouldn', 'any', 'didn', 'y', 'can', 'too', 'from', 'that', 'own', 'both', 'other', "you'd", 'it', 'yourself', 'these', 'herself', 'if', 'through', 'where', "shan't", 'do', 'the', 'are', 'couldn', 'my', 'now', 'few', 'wasn', 'doesn', 'before', 'down', 'he', 'while', 'being', 'have', 'more', "should've", "that'll", "you've", 'again', 'did', 'hers', 'yours', 'below', 'them', 'by', 'hadn', 'i', 'themselves', 'same', 'its', 'to', 'off', 'than', 'you', 'was', 'don', "isn't", 'or', 'once', 'your', 'is', "you'll", 've', 'aren', 'who', 'hasn', 'will', 'what', 'be', 'here', 'how', 'because', 'out', 'there', 'myself', 'and', 'nor', 's', 'not', 'those', 'had', 'only', 'but', 'yourselves', "didn't", "she's", 'then', 'has'}

def transcribe_audio_video(file_path):
    """
    Transcribes audio or video file using Whisper model.
    """
    if file_path.endswith('.m4a') or file_path.endswith('.mp4'):
        # Process .m4a or .mp4 files
        audio_filename = file_path + '.wav'  # Convert to WAV

        if file_path.endswith('.m4a'):
            audio = AudioSegment.from_file(file_path, "m4a")
            audio.export(audio_filename, format="wav")
        elif file_path.endswith('.mp4'):
            video = VideoFileClip(file_path)
            video.audio.write_audiofile(audio_filename)

        file_path = audio_filename

    # Transcribe
    result = whisper_model.transcribe(file_path, fp16=False)
    return result["text"]

def clean_text(text):

    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

app = Flask(__name__)
model = keras.models.load_model("./threat_detector_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()

def predict():
    if request.method == "POST":
        file = request.files.get('file')
        if file and file.filename != '':
            if file.filename != '':
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                    tmp.write(file.read())
                    tmp_filename = tmp.name

                # Transcribe the audio/video file
                transcription = transcribe_audio_video(tmp_filename)

                threatspeech = [clean_text(transcription)]
                seq = load_tokenizer.texts_to_sequences(threatspeech)
                padded = pad_sequences(seq, maxlen=300)
                pred = model.predict(padded)
                if pred<0.2:
                    
                    return render_template('home.html',prediction_text="There is no Threatening speech found!!{}".format(pred))
                
                else:

                    return render_template('home.html',prediction_text=" Threatening speech found!! {}".format(pred))
 
        elif 'threat' in request.form:
            threatspeech = request.form["threat"]
            threatspeech = [clean_text(threatspeech)]
            seq = load_tokenizer.texts_to_sequences(threatspeech)
            

            paddedh = pad_sequences(seq, maxlen=300)
            predh = model.predict(paddedh)
            if predh<0.2:
                return render_template('home.html',prediction_text="There is no Threatening speech found!! {}".format(predh))
            else:
                return render_template('home.html',prediction_text=" Threatening speech found!!{}".format(predh))
    return render_template("home.html")
if __name__ == "__main__":
    app.run(debug=True)


        




