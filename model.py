from feature_extractor import get_features_df, get_full_data
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV
import pandas as pd
from augmentation import augmentation
import librosa
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
import numpy as np

data = get_full_data()

to_test = data[data['name'] == 'Степа'].reset_index().drop(['index'], axis=1)

data = data.drop(data[data['name'] == 'Степа'].index)

test, test_sr =  librosa.load(to_test['path'][2], mono=True, duration=1)

test_features = get_features_df(test, test_sr).drop(0)

data = data.drop(['path', 'name','phrase'], axis=1)

for augmented_audio in augmentation(test, test_sr, data.shape[0]):
    test_features = pd.concat([test_features, get_features_df(augmented_audio[0], augmented_audio[1])])

test_features = test_features.reset_index()
train = pd.concat([data, test_features], axis=0, ignore_index=True).drop(['index'], axis=1)
labels = [1] * data.shape[0]
labels.extend([0] * test_features.shape[0])

param_grid = dict(n_neighbors = list(range(1, 30)))
model = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
model.fit(train, labels)
model_params = model.get_params()

y_pred = []
y_probas = []
for path in to_test['path']:
    y, sr = librosa.load(path, mono=True, duration=1)
    pred = model.best_estimator_.predict(get_features_df(y, sr))
    y_probas.append(model.best_estimator_.predict_proba(get_features_df(y, sr))[0])
    y_pred.append(pred[0])
    print(pred, path.split('/')[-1])

wandb.init(project='speaker-recognition', config=model_params)

wandb.config.update({"test_size" : 0.095,
                    "train_len" : len(train),
                    "test_len" : len(to_test)})

y_test  = [0]*5 + [1]*(to_test.shape[0] - 5)
print(y_test)
plot_class_proportions(labels, y_test)
plot_learning_curve(model, train, labels)
plot_roc(y_test, y_probas)
plot_precision_recall(y_test, y_probas)
plot_feature_importances(model)

wandb.finish()


