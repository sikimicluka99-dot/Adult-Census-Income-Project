import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_all_models(processed_dir, models_dir, results_dir):
    """Evaluacija svih modela sa optimalnim threshold-om za Gradient Boosting."""
    
    fig_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Učitavanje test skupa
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    X_test = test_df.drop(columns=['income'])
    y_test = test_df['income']
    
    print(f"Test skup: {X_test.shape}")
    
    # Top 5 atributa za redukovani model
    top_5_features = joblib.load(os.path.join(models_dir, 'top_5_features.pkl'))
    
    # Učitaj optimalni threshold za Gradient Boosting
    BEST_THRESHOLD = joblib.load(os.path.join(models_dir, 'best_threshold.pkl'))
    print(f"\n🔧 Optimalni threshold za Gradient Boosting: {BEST_THRESHOLD:.2f}")
    
    # SVI modeli za evaluaciju
    models_to_eval = {
        'Logistic Regression (Baseline)': (
            joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
            X_test,
            False
        ),
        'SVM (Support Vector Machine)': (
            joblib.load(os.path.join(models_dir, 'svm.pkl')),
            X_test,
            False
        ),
        'Random Forest (Optimized)': (
            joblib.load(os.path.join(models_dir, 'random_forest.pkl')),
            X_test,
            False
        ),
        f'Gradient Boosting (Optimized) (threshold={BEST_THRESHOLD:.2f})': (
            joblib.load(os.path.join(models_dir, 'gradient_boosting.pkl')),
            X_test,
            True,
            BEST_THRESHOLD
        ),
        'XGBoost (Optimized)': (
            joblib.load(os.path.join(models_dir, 'xgboost.pkl')),
            X_test,
            False
        ),
        'XGBoost (Reduced - Top 5)': (
            joblib.load(os.path.join(models_dir, 'xgboost_reduced.pkl')),
            X_test[top_5_features],
            False
        )
    }
    
    results = []
    
    print("\n" + "="*80)
    print(" EVALUACIJA MODELA - ADULT CENSUS INCOME")
    print("="*80)
    
    for name, model_info in models_to_eval.items():
        if len(model_info) == 3:
            model, X_data, use_threshold = model_info
            threshold = 0.5
        else:
            model, X_data, use_threshold, threshold = model_info
        
        if use_threshold and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_data)[:, 1]
            y_pred = (y_prob > threshold).astype(int)
            y_prob_for_roc = y_prob
            display_name = name
        else:
            y_pred = model.predict(X_data)
            display_name = name
            
            try:
                y_prob_for_roc = model.predict_proba(X_data)[:, 1]
            except AttributeError:
                try:
                    y_prob_for_roc = model.decision_function(X_data)
                    y_prob_for_roc = (y_prob_for_roc - y_prob_for_roc.min()) / (y_prob_for_roc.max() - y_prob_for_roc.min())
                except:
                    y_prob_for_roc = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob_for_roc)
        
        results.append({
            'Model': display_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        # Matrica konfuzije
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                    xticklabels=['<=50K', '>50K'],
                    yticklabels=['<=50K', '>50K'])
        plt.title(f'Matrica konfuzije: {display_name}\nF1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}')
        plt.ylabel('Stvarna klasa')
        plt.xlabel('Predviđena klasa')
        plt.tight_layout()
        
        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').replace('=', '_')
        plt.savefig(os.path.join(fig_dir, f'{filename}_confusion_matrix.png'), dpi=150)
        plt.close()
        
        print(f"\n{display_name}:")
        print(f"  → Accuracy:  {acc:.4f}")
        print(f"  → Precision: {prec:.4f}")
        print(f"  → Recall:    {rec:.4f}")
        print(f"  → F1-Score:  {f1:.4f}")
        print(f"  → ROC-AUC:   {roc_auc:.4f}")
    
    # Čuvanje CSV izveštaja
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    results_df.to_csv(os.path.join(metrics_dir, 'model_comparison.csv'), index=False)
    
    # Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(results_df.set_index('Model')[['F1-Score', 'ROC-AUC', 'Precision', 'Recall', 'Accuracy']],
                annot=True, cmap='RdYlGn', center=0.7, fmt='.3f')
    plt.title('Toplotna mapa performansi modela - Adult dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'models_heatmap.png'), dpi=150)
    plt.close()
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    bars = plt.barh(results_df['Model'], results_df['F1-Score'], color=colors[:len(results_df)])
    plt.xlabel('F1-Score')
    plt.title('Poređenje modela po F1-Score')
    plt.xlim(0, 1)
    
    for bar, value in zip(bars, results_df['F1-Score']):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'models_comparison_bar.png'), dpi=150)
    plt.close()
    
    print("\n" + "="*80)
    print(" ZAVRŠNI IZVEŠTAJ")
    print("="*80)
    print(results_df[['Model', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    print("="*80)
    print(f"\n✅ Svi rezultati sačuvani u: {results_dir}")

if __name__ == "__main__":
    evaluate_all_models(
        processed_dir='data/processed',
        models_dir='models',
        results_dir='results'
    )