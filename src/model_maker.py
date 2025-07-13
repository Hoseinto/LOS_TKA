import os
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class ModelSelector:
    def __init__(self, X_train, y_train, save_dir="saved_models"):
        self.X_train = X_train
        self.y_train = y_train
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_model = None
        self.best_score = -np.inf
        self.best_name = None

    def _tune_and_save_model(self, model, params, name):
        print(f"🔍 Tuning {name}...")
        grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)

        model_file = os.path.join(self.save_dir, f"{name}_best.joblib")
        joblib.dump(grid.best_estimator_, model_file)
        print(f"✅ {name} saved to {model_file} with F1 score: {grid.best_score_:.4f}")

        if grid.best_score_ > self.best_score:
            self.best_score = grid.best_score_
            self.best_model = grid.best_estimator_
            self.best_name = name

    def run_all(self):
        self._tune_and_save_model(LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10]
        }, 'LogisticRegression')

        self._tune_and_save_model(RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }, 'RandomForest')

        self._tune_and_save_model(KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7]
        }, 'KNN')

        self._tune_and_save_model(SVC(probability=True), {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }, 'SVM')

        self._tune_and_save_model(XGBClassifier(eval_metric='logloss'), {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }, 'XGBoost')
