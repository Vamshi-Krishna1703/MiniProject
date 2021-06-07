from flask import Flask, render_template, request
import pickle
import numpy as np
import librosa
import soundfile
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/' , methods = ['GET'])
def helloworld():
    return render_template("index.html")


@app.route('/', methods = ['POST'])
def predict():
    audioFile = request.files['AudioFile']
    audio_path = "./audio/"+audioFile.filename
    audioFile.save(audio_path)

    fs=extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
    example_x = []
    example_x.append(fs)

    op = model.predict(example_x)[0]
    return render_template("index.html", pred = op)
    

#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

if __name__ == '__main__':
    app.run(port = 3000, debug = True)