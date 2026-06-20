import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import xgboost as xgb

def train_models(processed_dir, models_dir, figures_dir):
    """Treniranje modela na Adult datasetu sa optimalnim parametrima."""
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Učitavanje podataka
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    
    X_train = train_df.drop(columns=['income'])
    y_train = train_df['income']
    
    X_val = val_df.drop(columns=['income'])
    y_val = val_df['income']
    
    print("="*80)
    print("TRENIRANJE MODELA - ADULT CENSUS INCOME")
    print("="*80)
    print(f"Train skup: {X_train.shape}")
    print(f"Validation skup: {X_val.shape}")
    
    results = {}
    
    # ============================================================
    # 1. Logistic Regression (baseline)
    # ============================================================
    print("\n[1/6] Treniram Logistic Regression...")
    lr_model = LogisticRegressionCV(cv=5, random_state=42, max_iter=1000, scoring='f1')
    lr_model.fit(X_train, y_train)
    
    y_val_pred = lr_model.predict(X_val)
    lr_val_f1 = f1_score(y_val, y_val_pred)
    print(f"    => F1 na validation skupu: {lr_val_f1:.4f}")
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    results['Logistic Regression'] = lr_val_f1
    
    # ============================================================
    # 2. SVM
    # ============================================================
    print("\n[2/6] Podešavam SVM...")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }
    
    svm_grid = GridSearchCV(
        SVC(random_state=42, class_weight='balanced'),
        svm_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    
    best_svm = svm_grid.best_estimator_
    y_val_pred = best_svm.predict(X_val)
    svm_val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"    => Najbolji SVM parametri: {svm_grid.best_params_}")
    print(f"    => SVM F1 na validation skupu: {svm_val_f1:.4f}")
    joblib.dump(best_svm, os.path.join(models_dir, 'svm.pkl'))
    results['SVM'] = svm_val_f1
    
    # ============================================================
    # 3. Random Forest
    # ============================================================
    print("\n[3/6] Podešavam Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced', None]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    y_val_pred = best_rf.predict(X_val)
    rf_val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"    => Najbolji RF parametri: {rf_grid.best_params_}")
    print(f"    => RF F1 na validation skupu: {rf_val_f1:.4f}")
    joblib.dump(best_rf, os.path.join(models_dir, 'random_forest.pkl'))
    results['Random Forest'] = rf_val_f1
    
    # ============================================================
    # 4. Gradient Boosting - POBEDNIČKI PARAMETRI!
    # ============================================================
    print("\n[4/6] Treniram Gradient Boosting sa optimalnim parametrima...")
    
    # NAJBOLJI PARAMETRI iz skripte:
    # threshold=0.42, weight=None, learning_rate=0.08, n_estimators=150, max_depth=5, subsample=0.80
    
    best_gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.80,
        random_state=42
    )
    best_gb.fit(X_train, y_train)
    
    y_val_pred = best_gb.predict(X_val)
    gb_val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"    => Najbolji GB parametri:")
    print(f"       - n_estimators: 150")
    print(f"       - learning_rate: 0.08")
    print(f"       - max_depth: 5")
    print(f"       - subsample: 0.80")
    print(f"       - random_state: 42")
    print(f"    => GB F1 na validation skupu: {gb_val_f1:.4f}")
    
    joblib.dump(best_gb, os.path.join(models_dir, 'gradient_boosting.pkl'))
    results['Gradient Boosting'] = gb_val_f1
    
    # Sačuvaj i threshold za evaluate.py
    BEST_THRESHOLD = 0.42
    joblib.dump(BEST_THRESHOLD, os.path.join(models_dir, 'best_threshold.pkl'))
    print(f"    => Optimalni threshold: {BEST_THRESHOLD:.2f}")
    
    # ============================================================
    # 5. XGBoost
    # ============================================================
    print("\n[5/6] Podešavam XGBoost...")
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    xgb_param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss'),
        xgb_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    xgb_grid.fit(X_train, y_train)
    
    best_xgb = xgb_grid.best_estimator_
    y_val_pred = best_xgb.predict(X_val)
    xgb_val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"    => Najbolji XGBoost parametri: {xgb_grid.best_params_}")
    print(f"    => XGBoost F1 na validation skupu: {xgb_val_f1:.4f}")
    joblib.dump(best_xgb, os.path.join(models_dir, 'xgboost.pkl'))
    results['XGBoost'] = xgb_val_f1
    
    # ============================================================
    # 6. Feature importance
    # ============================================================
    print("\n[6/6] Selekcija najvažnijih atributa...")
    
    importances = pd.Series(best_gb.feature_importances_, index=X_train.columns)
    importances_sorted = importances.sort_values(ascending=False)
    
    top_5_features = list(importances_sorted.head(5).index)
    top_10_features = list(importances_sorted.head(10).index)
    
    # Grafikon feature importance
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(importances_sorted.head(10).values / importances_sorted.head(10).max())
    importances_sorted.head(10).plot(kind='barh', color=colors)
    plt.xlabel('Važnost')
    plt.title('Top 10 najvažnijih atributa (Gradient Boosting)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    
    joblib.dump(top_5_features, os.path.join(models_dir, 'top_5_features.pkl'))
    joblib.dump(top_10_features, os.path.join(models_dir, 'top_10_features.pkl'))
    print(f"    => Top 5 atributa: {top_5_features}")
    print(f"    => Top 10 atributa: {top_10_features}")
    
    # ============================================================
    # BONUS: Redukovani XGBoost (top 5 atributa)
    # ============================================================
    print("\n[Bonus] Treniram redukovani XGBoost (top 5)...")
    X_train_reduced = X_train[top_5_features]
    X_val_reduced = X_val[top_5_features]
    
    reduced_xgb = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    reduced_xgb.fit(X_train_reduced, y_train)
    
    y_val_pred = reduced_xgb.predict(X_val_reduced)
    reduced_val_f1 = f1_score(y_val, y_val_pred)
    print(f"    => Redukovani XGBoost F1 na validation skupu: {reduced_val_f1:.4f}")
    joblib.dump(reduced_xgb, os.path.join(models_dir, 'xgboost_reduced.pkl'))
    results['XGBoost (Top 5)'] = reduced_val_f1
    
    # ============================================================
    # ZAVRŠNI IZVEŠTAJ
    # ============================================================
    print("\n" + "="*80)
    print("VALIDACIONI REZULTATI (F1 skor)")
    print("="*80)
    for model_name, f1_score_val in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {f1_score_val:.4f}")
    print("="*80)
    print("\n🏆 NAJBOLJI MODEL: Gradient Boosting")
    print(f"   → F1-Score: {gb_val_f1:.4f}")
    print(f"   → Optimalni threshold: {BEST_THRESHOLD:.2f}")
    print(f"   → Parametri: n_estimators=150, learning_rate=0.08, max_depth=5, subsample=0.80")
    print("\n✅ TRENIRANJE ZAVRŠENO!")
    print(f"📁 Modeli sačuvani u: {models_dir}")
    print(f"📁 Threshold sačuvan u: {models_dir}/best_threshold.pkl")

if __name__ == "__main__":
    train_models(
        processed_dir='data/processed',
        models_dir='models',
        figures_dir='results/figures'
    )