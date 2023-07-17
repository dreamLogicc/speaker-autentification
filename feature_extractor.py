import librosa
import pandas as pd
import numpy as np 
import os
import scipy

def get_files():
    files = []
    for dir in os.listdir('./fastapi_app/voice_data'):
        for file in os.listdir('./fastapi_app/voice_data' + '/' + dir):

            if 'Вход' in file:
                phrase = 'Вход'
            if 'Пицца' in file:
                phrase = 'Пицца'
            if 'Привет' in file:
                phrase = 'Привет'
            if 'Собака' in file:
                phrase = 'Собака'
            if 'Шкаф' in file:
                phrase = 'Шкаф'

            files.append({
                    'path': './fastapi_app/voice_data' + '/' + dir + '/' + file,
                    'name': dir,
                    'phrase': phrase
                })

    return pd.DataFrame(files)

def feature_extractor(y, sr):
    features = [] 
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr, 
                                                              n_mfcc=20)])  # mfcc_mean<0..20>
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                             n_mfcc=20)])   # mfcc_std
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                            axis = 0)[0])     # cent_mean
    features.append(np.std(librosa.feature.spectral_centroid(y=y,sr=sr).T,
                           axis = 0)[0])       # cent_std
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y,sr=sr).T,
                                     axis = 0)[0])    # cent_skew
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, 
                            axis = 0)[0])      # rolloff_mean
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T, 
                           axis = 0)[0])       # rolloff_std
    
    return features

def get_features_df(y, sr):
    all_features = []
    features = feature_extractor(y, sr)
    temp = {}
    for i in range(20):
        temp[f'mfcc_mean{i}'] = round(features[i], 3)
    for i in range(20, 40):
        temp[f'mfcc_std{i}'] = round(features[i], 3)
    temp['cent_mean'] = round(features[40], 3)
    temp['cent_std'] = round(features[41], 3)
    temp['cent_skew'] = round(features[42], 3)
    temp['rolloff_mean'] = round(features[43], 3)
    temp['rolloff_std'] = round(features[44], 3)
    all_features.append(temp)
    return pd.DataFrame(all_features)

def get_full_data():
    data = get_files()
    print(data)
    features = pd.DataFrame()
    for path in data['path']:
        y, sr = librosa.load(path, mono=True, duration=1)
        temp = get_features_df(y, sr)
        features = pd.concat([features, temp])
    features = features.reset_index()
    return pd.concat([data, features], axis=1)


if __name__ == '__main__':
    data = get_full_data().drop(['index'], axis = 1)
    data.to_csv('./fastapi_app/data/full_data.csv', index=False)
    data = data.drop(data[data['name'] == 'Степа'].index).drop(['name', 'phrase', 'path'], axis = 1)
    data.to_csv('./fastapi_app/data/data_for_api.csv', index=False)