from fastapi import FastAPI, File
from feature_extractor import get_features_df
import pandas as pd
import librosa
import io
import numpy as np
import joblib

app = FastAPI(
    title='Audio Autentification'
)

model_person = joblib.load('./person_clf.joblib')
model_phrase = joblib.load('./phrase_clf1.joblib')

threshold = 0.5


@app.post("/file/upload-file")
def upload_file(audio_1: bytes = File(), audio_2: bytes = File()):
    '''Функция принимате на вход два аудиофайла и сравнивает их между собой'''
    y_1, sr_1 = librosa.load(io.BytesIO(audio_1), mono=True, duration=1)
    y_2, sr_2 = librosa.load(io.BytesIO(audio_2), mono=True, duration=1)

    features_1 = get_features_df(y_1, sr_1)
    features_2 = get_features_df(y_2, sr_2)

    person_probas = model_person.predict_proba(
        np.array(pd.concat([features_1, features_2], axis=1)))
    phrase_probas = model_phrase.predict_proba(
        np.array(pd.concat([features_1, features_2], axis=1)))

    person = 1 if person_probas[0][1] > threshold else 0
    phrase = 1 if phrase_probas[0][1] > threshold else 0

    flag = False
    similarity = 0
    if person:
        if phrase:
            flag = True
            similarity = round(
                (person_probas[0][1]*100 + phrase_probas[0][1]*100)/2, 2)

    return {'Access': flag, 'Similarity': str(similarity) + '%'}
