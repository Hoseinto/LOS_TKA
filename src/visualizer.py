import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

class ModelVisualizer:
    def __init__(self, models_dir, X_test, y_test, output_dir="results/figures"):
        self.models_dir = models_dir
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = self._load_models()

    def _load_models(self):
        models = {}
        for file in os.listdir(self.models_dir):
            if file.endswith("_best.joblib"):
                name = file.replace("_best.joblib", "")
                models[name] = joblib.load(os.path.join(self.models_dir, file))
        return models

    def plot_roc_curves(self):
        plt.figure(figsize=(8, 6))
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(self.X_test)[:, 1]
            else:  # e.g., SVM without probability=True
                y_scores = model.decision_function(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()

    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"{name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")
        # Hide unused subplot if fewer than 6
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrices.png"))
        plt.close()

    def plot_shap_values(self):
        for name, model in self.models.items():
            try:
                explainer = shap.Explainer(model, self.X_test)
                shap_values = explainer(self.X_test)
                shap.summary_plot(shap_values, self.X_test, show=False)
                plt.title(f"SHAP Summary - {name}")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"shap_{name}.png"))
                plt.close()
            except Exception as e:
                print(f"⚠️ SHAP failed for {name}: {e}")
