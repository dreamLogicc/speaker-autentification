from fastapi import FastAPI, File
from feature_extractor import get_features_df
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
from augmentation import augmentation
import librosa
import io
import numpy as np

app = FastAPI(
    title='Audio Autentification'
)

@app.post("/file/upload-file")
def upload_file(audio_1: bytes = File(), audio_2: bytes = File()):
  '''Функция принимате на вход два аудиофайла и сравнивает их между собой'''
  y_1, sr_1 = librosa.load(io.BytesIO(audio_1), mono=True, duration=1)
  y_2, sr_2 = librosa.load(io.BytesIO(audio_2), mono=True, duration=1)

  data = pd.read_csv('/app/data_for_api.csv')

  audio_1_features = get_features_df(y_1, sr_1)
  audio_2_features = get_features_df(y_2, sr_2)

  similarity = round(100 - mean_absolute_percentage_error(audio_1_features, audio_2_features), 2)

  audio_1_features = audio_1_features.drop(0) 
  for augmented_audio in augmentation(y_1, sr_1, data.shape[0]):
    audio_1_features = pd.concat([audio_1_features, get_features_df(augmented_audio[0], augmented_audio[1])])

  labels = [1] * data.shape[0]
  labels.extend([0] * audio_1_features.shape[0])

  model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)

  model.fit(pd.concat([data, audio_1_features], axis=0, ignore_index=True), labels)

  pred = model.predict(np.array(audio_2_features))
  flag = False
  if pred[0] == 0:
    flag = True
  else: similarity = 0

  return {'Access': flag, 'Similarity': str(similarity) + '%'}