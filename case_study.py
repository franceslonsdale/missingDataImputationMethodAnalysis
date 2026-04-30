import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

# import the models
from gain import GAIN
from midas_imputer import MIDAS
from config import GAIN_CONFIG, MIDAS_CONFIG

warnings.filterwarnings("ignore")

def loadAndCorruptData(miss_rate=0.30, random_state=42):
    """Loads Breast Cancer dataset and injects MCAR missingness."""
    data = load_breast_cancer()
    X_true = data.data
    y = data.target
    
    rng = np.random.default_rng(random_state)
    mask = rng.random(X_true.shape) < miss_rate
    
    X_miss = X_true.copy()
    X_miss[mask] = np.nan
    
    return X_true, X_miss, y

def evalImp(X_imp, y):
    """Splits data, trains Random Forest, and returns metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return acc, fpr, tpr, roc_auc

def run_case_study():
    print("="*50)
    print("Starting Breast Cancer Case Study (30% Missingness)")
    print("="*50)
    
    X_true, X_miss, y = loadAndCorruptData(miss_rate=0.30)
    results = {}
    
    plt.figure(figsize=(10, 8))
    
    # Baseline - Complete Data
    print("Running Complete Data Baseline...")
    acc_true, fpr_true, tpr_true, auc_true = evalImp(X_true, y)
    plt.plot(fpr_true, tpr_true, label=f'Complete Data (AUC = {auc_true:.3f})', 
             linestyle='--', color='black', linewidth=2)
    
    # MICE (via Python's IterativeImputer for continuous data)
    print("Running MICE...")
    mice = IterativeImputer(max_iter=10, random_state=42)
    X_mice = mice.fit_transform(X_miss)
    acc_m, fpr_m, tpr_m, auc_m = evalImp(X_mice, y)
    results['MICE'] = {'Accuracy': acc_m, 'AUC': auc_m}
    plt.plot(fpr_m, tpr_m, label=f'MICE (AUC = {auc_m:.3f})', linewidth=2)
    
    # GAIN
    print("Running GAIN...")
    gain = GAIN(**GAIN_CONFIG, seed=42)
    XGain = gain.fit_transform(X_miss, n_imputations=1)
    if isinstance(XGain, list): XGain = XGain[0] 
    acc_g, fpr_g, tpr_g, auc_g = evalImp(XGain, y)
    results['GAIN'] = {'Accuracy': acc_g, 'AUC': auc_g}
    plt.plot(fpr_g, tpr_g, label=f'GAIN (AUC = {auc_g:.3f})', linewidth=2)
    
    # 4. MIDAS
    print("Running MIDAS...")
    midas = MIDAS(**MIDAS_CONFIG, seed=42)
    XMidas = midas.fit_transform(X_miss, n_imputations=1)
    if isinstance(XMidas, list): XMidas = XMidas[0]
    acc_md, fpr_md, tpr_md, auc_md = evalImp(XMidas, y)
    results['MIDAS'] = {'Accuracy': acc_md, 'AUC': auc_md}
    plt.plot(fpr_md, tpr_md, label=f'MIDAS (AUC = {auc_md:.3f})', linewidth=2)
    
    # Finalize Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves for Breast Cancer Classification\n(Trained on 30% Missing Data)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.savefig("figures/case_study_roc.png", dpi=300, bbox_inches='tight')
    print("\nSaved ROC curve to figures/case_study_roc.png")
    
    # Print Results Table
    print("\nFinal Test Set Results! :")
    print(f"{'Method':<15} | {'Accuracy':<10} | {'AUC':<10}")
    print("-" * 40)
    for method, metrics in results.items():
        print(f"{method:<15} | {metrics['Accuracy']:.4f}     | {metrics['AUC']:.4f}")

if __name__ == "__main__":
    run_case_study()