from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc
import numpy as np
from imblearn.over_sampling import ADASYN
import joblib

data = pd.read_csv('./data/train.csv')

labels = data['name_labels']

data = data.drop(['name_labels', 'phrase_labels'], axis=1)
print(labels.value_counts())
print(data)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)

ros = ADASYN()
ros_x_train, ros_y_train = ros.fit_resample(x_train, y_train)

ros_x_test, ros_y_test = ros.fit_resample(x_test, y_test)

print(Counter(ros_y_train))
print(Counter(ros_y_test))

param_grid = { 
    'n_estimators': [100, 150, 200],
    'max_depth' : [4,5,6],
    'criterion' :['gini', 'entropy'],
    'verbose': [1],
    'n_jobs': [-1]
}

model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)

model.fit(ros_x_train, ros_y_train)

print(classification_report(model.best_estimator_.predict(ros_x_test), ros_y_test))
print(roc_auc_score(model.best_estimator_.predict(ros_x_test), ros_y_test))

joblib.dump(model.best_estimator_, './person_clf.joblib')

# model_params = model.best_estimator_.get_params()

# wandb.init(project='speaker-recognition', config=model_params)

# wandb.config.update({"test_size" : 0.2,
#                     "train_len" : len(x_train),
#                     "test_len" : len(x_test)})

# print(y_test)
# plot_class_proportions([0,1], ros_y_test)
# plot_learning_curve(model.best_estimator_, ros_x_train, [0,1])
# plot_roc(ros_y_test, model.best_estimator_.predict_proba(ros_x_test))
# plot_precision_recall(y_test, model.best_estimator_.predict_proba(ros_x_test))
# plot_feature_importances(model.best_estimator_)

# wandb.finish()





