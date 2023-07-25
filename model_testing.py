from feature_extractor import get_features_df
import librosa
import pandas as pd
import joblib
import numpy as np

model_person = joblib.load('./person_clf.joblib')
model_phrase = joblib.load('./phrase_clf1.joblib')

y_1, sr_1 = librosa.load('./voice_data/Степа/Гайволя Степан Вход (2).wav', mono=True, duration=1)
y_2, sr_2 = librosa.load('./voice_data/Алексей/Вход - не распознано Алексей.wav', mono=True, duration=1)

features_1 = get_features_df(y_1, sr_1)
features_2 = get_features_df(y_2, sr_2)

print(model_person.predict_proba(np.array(pd.concat([features_1, features_2], axis=1))))
print(model_phrase.predict_proba(np.array(pd.concat([features_1, features_2], axis=1))))