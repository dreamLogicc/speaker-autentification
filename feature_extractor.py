import librosa
import pandas as pd
import numpy as np
import os
import scipy
from itertools import combinations


def get_files():
    '''
        Функция сообирает файлы из дериктории в датафрейм
    '''
    files = []
    for dir in os.listdir('./voice_data'):
        for file in os.listdir('./voice_data' + '/' + dir):

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
                'path': './voice_data' + '/' + dir + '/' + file,
                'name': dir,
                'phrase': phrase
            })

    return pd.DataFrame(files)


def feature_extractor(y, sr):
    '''
        Функция для выделения фичей
        y: временной ряд аудиофайла
        sr: частота дискретизации
    '''

    features = []
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                              n_mfcc=20)])  # mfcc_mean<0..20>
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr,
                                                             n_mfcc=20)])   # mfcc_std
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                            axis=0)[0])     # cent_mean
    features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                           axis=0)[0])       # cent_std
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,
                                     axis=0)[0])    # cent_skew
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                            axis=0)[0])      # rolloff_mean
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,
                           axis=0)[0])       # rolloff_std

    return features


def get_features_df(y, sr):
    '''
        Функция для создания датафрейма из фичей
        y: временной ряд аудиофайла
        sr: частота дискретизации
    '''

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
    '''
        Функция для получения полного датасета
    '''

    data = get_files()
    features = pd.DataFrame()
    for path in data['path']:
        y, sr = librosa.load(path, mono=True, duration=1)
        temp = get_features_df(y, sr)
        features = pd.concat([features, temp])
    features = features.reset_index()
    return pd.concat([data, features], axis=1)


if __name__ == '__main__':
    data = get_full_data().drop(['index'], axis=1)
    data = data.drop(data[data['name'] == 'Степа'].index)
    # data['name_phrase'] = data[['name', 'phrase']].apply(
    #     lambda x: '_'.join(x), axis=1)
    # data = data.drop(['name', 'phrase'], axis=1)
    a, b = map(list, zip(*combinations(data.index, 2)))
    combined = pd.concat(
        [data.loc[a].reset_index().rename(columns={'name': 'name_comb', 'phrase':'phrase_comb'}), data.loc[b].reset_index()], axis=1)
    combined = combined.drop(['path', 'index'], axis=1)
    labels_names = pd.concat([combined['name'], combined['name_comb']], axis=1)
    labels_phrases = pd.concat([combined['phrase_comb'], combined['phrase']], axis=1)
    y_names = []
    y_phrases = []
    for (i, j) in zip(labels_names['name'], labels_names['name_comb']):
        y_names.append(1 if i==j else 0)
    for (i, j) in zip(labels_phrases['phrase'], labels_phrases['phrase_comb']):
        y_phrases.append(1 if i==j else 0)
    combined = combined.drop(['name', 'name_comb', 'phrase_comb',  'phrase'], axis=1)
    combined['name_labels'] = y_names
    combined['phrase_labels'] = y_phrases
    combined.to_csv('./data/train.csv', index=False)
