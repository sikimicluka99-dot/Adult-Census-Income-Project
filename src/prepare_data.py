import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# UTF-8 podrska za Windows terminal
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


def convert_to_numeric(df):
    """Konvertuje sve numeričke kolone u odgovarajući tip."""
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def check_anomalies(df, name="Dataset"):
    """Proverava anomalije u dataset-u."""
    print(f"\n{'='*60}")
    print(f" PROVERA ANOMALIJA - {name}")
    print(f"{'='*60}")
    
    # 1. Age
    print(f"\n📊 Age:")
    print(f"   Min: {df['age'].min()}")
    print(f"   Max: {df['age'].max()}")
    print(f"   Mean: {df['age'].mean():.2f}")
    print(f"   Std: {df['age'].std():.2f}")
    
    if (df['age'] < 18).any():
        print(f"   ⚠️ Ima {(df['age'] < 18).sum()} osoba mladjih od 18 godina")
    if (df['age'] > 90).any():
        print(f"   ⚠️ Ima {(df['age'] > 90).sum()} osoba starijih od 90 godina")
    
    # 2. Hours per week
    print(f"\n📊 Hours per week:")
    print(f"   Min: {df['hours_per_week'].min()}")
    print(f"   Max: {df['hours_per_week'].max()}")
    print(f"   Mean: {df['hours_per_week'].mean():.2f}")
    print(f"   Std: {df['hours_per_week'].std():.2f}")
    
    if (df['hours_per_week'] == 0).any():
        print(f"   ⚠️ Ima {(df['hours_per_week'] == 0).sum()} osoba sa 0 radnih sati")
    if (df['hours_per_week'] > 80).any():
        print(f"   ⚠️ Ima {(df['hours_per_week'] > 80).sum()} osoba sa preko 80 radnih sati")
    
    # 3. Capital gain
    print(f"\n📊 Capital gain:")
    print(f"   Min: {df['capital_gain'].min()}")
    print(f"   Max: {df['capital_gain'].max()}")
    print(f"   Mean: {df['capital_gain'].mean():.2f}")
    print(f"   Median: {df['capital_gain'].median():.2f}")
    
    gain_percent = (df['capital_gain'] > 0).sum() / len(df) * 100
    print(f"   Ima capital_gain: {gain_percent:.1f}%")
    
    # 4. Capital loss
    print(f"\n📊 Capital loss:")
    print(f"   Min: {df['capital_loss'].min()}")
    print(f"   Max: {df['capital_loss'].max()}")
    print(f"   Mean: {df['capital_loss'].mean():.2f}")
    print(f"   Median: {df['capital_loss'].median():.2f}")
    
    loss_percent = (df['capital_loss'] > 0).sum() / len(df) * 100
    print(f"   Ima capital_loss: {loss_percent:.1f}%")
    
    # 5. Logičke greške
    print(f"\n📊 Logičke greške:")
    both_gain_loss = ((df['capital_gain'] > 0) & (df['capital_loss'] > 0)).sum()
    print(f"   Ima i capital_gain i capital_loss: {both_gain_loss}")
    
    high_income_zero_hours = ((df['income'] == '>50K') & (df['hours_per_week'] == 0)).sum()
    if high_income_zero_hours > 0:
        print(f"   ⚠️ Ima {high_income_zero_hours} osoba sa >50K i 0 radnih sati")
    
    # 6. Outlier-i (IQR metod)
    print(f"\n📊 Outlier-i (IQR metod):")
    for col in ['age', 'hours_per_week', 'capital_gain', 'capital_loss']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"   {col}: {outliers} outlier-a (donja: {lower_bound:.2f}, gornja: {upper_bound:.2f})")
    
    print(f"\n{'='*60}")
    print(f"✅ Provera anomalija zavrsena!\n")


def load_train_data(filepath):
    """Učitava adult.train fajl, popunjava nedostajuće vrednosti."""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
    
    # Zameni '?' sa NaN
    df = df.replace('?', np.nan)
    
    # Konvertuj numeričke kolone (pre provere anomalija)
    df = convert_to_numeric(df)
    
    # Prvo obriši redove gde income nedostaje (target ne možemo popuniti)
    before = len(df)
    df = df.dropna(subset=['income'])
    after = len(df)
    if before > after:
        print(f"   → Obrisano {before - after} redova sa nedostajućim income")
    
    # Broj nedostajućih vrednosti pre popunjavanja
    missing_before = df.isnull().sum().sum()
    print(f"   → Nedostajuće vrednosti pre popunjavanja: {missing_before}")
    
    # ========== POPUNJAVANJE (IMPUTACIJA) ==========
    
    # 1. Numeričke kolone: popuni sa MEDIJANOM
    num_cols = ['capital_gain', 'capital_loss']
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      → {col}: popunjeno sa medijanom = {median_val:.2f}")
    
    # 2. Kategoričke kolone: popuni sa NAJČEŠĆOM vrednošću (mod)
    cat_cols = ['workclass', 'occupation', 'native_country']
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"      → {col}: popunjeno sa '{mode_val}'")
    
    # 3. Ostale kategoričke kolone
    other_cat_cols = ['marital_status', 'relationship', 'race', 'sex']
    for col in other_cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    
    # Provera da li ima još NaN
    missing_after = df.isnull().sum().sum()
    print(f"   → Nedostajuće vrednosti posle popunjavanja: {missing_after}")
    
    # Dropovanje irrelevantnih atributa
    df = df.drop(columns=['fnlwgt', 'education'])
    
    # Provera anomalija PRE brisanja duplikata
    check_anomalies(df, "Train skup (pre brisanja duplikata)")
    
    # Provera duplikata
    cols_for_duplicates = [col for col in df.columns if col != 'income']
    dup_count = df[cols_for_duplicates].duplicated().sum()
    
    if dup_count > 0:
        print(f"⚠️ U train skupu pronađeno {dup_count} duplikata ({dup_count/len(df)*100:.2f}%). Brisanje...")
        df = df.drop_duplicates(subset=cols_for_duplicates, keep='first')
        print(f"   → Train skup nakon brisanja: {len(df)} redova")
    
    # Provera anomalija POSLE brisanja duplikata
    check_anomalies(df, "Train skup (posle brisanja duplikata)")
    
    return df


def load_test_data(filepath):
    """Učitava adult.test fajl, popunjava nedostajuće vrednosti."""
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
    
    # Zameni '?' sa NaN
    df = df.replace('?', np.nan)
    df['income'] = df['income'].str.rstrip('.')
    
    # Konvertuj numeričke kolone (pre provere anomalija)
    df = convert_to_numeric(df)
    
    # Prvo obriši redove gde income nedostaje
    before = len(df)
    df = df.dropna(subset=['income'])
    after = len(df)
    if before > after:
        print(f"   → Obrisano {before - after} redova sa nedostajućim income")
    
    # Broj nedostajućih vrednosti pre popunjavanja
    missing_before = df.isnull().sum().sum()
    print(f"   → Nedostajuće vrednosti pre popunjavanja: {missing_before}")
    
    # ========== POPUNJAVANJE (IMPUTACIJA) ==========
    
    # 1. Numeričke kolone: popuni sa MEDIJANOM
    num_cols = ['capital_gain', 'capital_loss']
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"      → {col}: popunjeno sa medijanom = {median_val:.2f}")
    
    # 2. Kategoričke kolone: popuni sa NAJČEŠĆOM vrednošću
    cat_cols = ['workclass', 'occupation', 'native_country']
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"      → {col}: popunjeno sa '{mode_val}'")
    
    # 3. Ostale kategoričke kolone
    other_cat_cols = ['marital_status', 'relationship', 'race', 'sex']
    for col in other_cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    
    # Provera
    missing_after = df.isnull().sum().sum()
    print(f"   → Nedostajuće vrednosti posle popunjavanja: {missing_after}")
    
    # Dropovanje irrelevantnih atributa
    df = df.drop(columns=['fnlwgt', 'education'])
    
    # Provera anomalija PRE brisanja duplikata
    check_anomalies(df, "Test skup (pre brisanja duplikata)")
    
    # Provera duplikata
    cols_for_duplicates = [col for col in df.columns if col != 'income']
    dup_count = df[cols_for_duplicates].duplicated().sum()
    
    if dup_count > 0:
        print(f"⚠️ U test skupu pronađeno {dup_count} duplikata ({dup_count/len(df)*100:.2f}%). Brisanje...")
        df = df.drop_duplicates(subset=cols_for_duplicates, keep='first')
        print(f"   → Test skup nakon brisanja: {len(df)} redova")
    
    # Provera anomalija POSLE brisanja duplikata
    check_anomalies(df, "Test skup (posle brisanja duplikata)")
    
    return df


def perform_eda(df, fig_dir):
    """Eksplorativna analiza podataka."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Distribucija ciljne varijable (pie chart)
    plt.figure(figsize=(6, 4))
    income_counts = df['income'].value_counts()
    plt.pie(income_counts, labels=['<=50K', '>50K'], autopct='%1.1f%%',
            colors=['lightblue', 'coral'], explode=(0, 0.05))
    plt.title('Distribucija prihoda - Train skup')
    plt.savefig(os.path.join(fig_dir, 'eda_income_distribution.png'))
    plt.close()
    
    # 2. Prihod po polu
    plt.figure(figsize=(8, 5))
    pd.crosstab(df['sex'], df['income']).plot(kind='bar', color=['lightblue', 'coral'])
    plt.title('Prihod po polu')
    plt.xlabel('Pol')
    plt.ylabel('Broj osoba')
    plt.legend(title='Prihod', labels=['<=50K', '>50K'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_income_by_sex.png'))
    plt.close()
    
    # 3. Korelaciona matrica
    plt.figure(figsize=(8, 6))
    num_cols_eda = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    corr_matrix = df[num_cols_eda].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Korelaciona matrica numeričkih atributa')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_correlation_matrix.png'))
    plt.close()
    
    # 4. Boxplot starosti i radnih sati
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(data=df, x='income', y='age', ax=axes[0])
    axes[0].set_title('Starost: >50K vs <=50K')
    
    sns.boxplot(data=df, x='income', y='hours_per_week', ax=axes[1])
    axes[1].set_title('Radni sati nedeljno: >50K vs <=50K')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_age_hours_boxplot.png'))
    plt.close()
    
    print(f"EDA grafikoni sačuvani u: {fig_dir}")


def preprocess_data(df_train, df_test):
    """Priprema podataka za modeliranje (encoding)."""
    
    # Label encoding za target
    le = LabelEncoder()
    df_train['income'] = le.fit_transform(df_train['income'])
    df_test['income'] = le.transform(df_test['income'])
    
    # Numeričke kolone (uključujući originalne capital_gain i capital_loss)
    num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    # Nema binarnih kolona
    binary_cols = []
    
    # Kategoričke kolone (one-hot encoding)
    cat_cols = ['workclass', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']
    
    # One-hot encoding
    df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
    
    # Poravnavanje kolona
    for col in df_train.columns:
        if col not in df_test.columns and col != 'income':
            df_test[col] = 0
    
    df_test = df_test[df_train.columns]
    
    return df_train, df_test, num_cols, binary_cols


def save_processed_data(train_path, test_path, processed_dir, fig_dir):
    """Glavna funkcija – priprema i čuva podatke."""
    
    print("="*60)
    print("PRIpreMA PODATAKA - ADULT CENSUS INCOME")
    print("="*60)
    
    print("\n[1/5] Učitavam podatke...")
    df_train = load_train_data(train_path)
    df_test = load_test_data(test_path)
    print(f"      Train skup: {df_train.shape}")
    print(f"      Test skup:  {df_test.shape}")
    
    print("\n[2/5] Radim eksplorativnu analizu...")
    perform_eda(df_train, fig_dir)
    
    print("\n[3/5] Encoding podataka...")
    df_train_enc, df_test_enc, num_cols, binary_cols = preprocess_data(df_train, df_test)
    
    print("\n[4/5] Delim podatke na train/val/test...")
    X = df_train_enc.drop(columns=['income'])
    y = df_train_enc['income']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_test = df_test_enc.drop(columns=['income'])
    y_test = df_test_enc['income']
    
    print(f"      → Train:      {X_train.shape}")
    print(f"      → Validation: {X_val.shape}")
    print(f"      → Test:       {X_test.shape}")
    print(f"      → Numeričke kolone (skaliraju se): {num_cols}")
    
    print("\n[5/5] Skaliram podatke i čuvam...")
    
    scaler = StandardScaler()
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    os.makedirs(processed_dir, exist_ok=True)
    
    import joblib
    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
    
    X_train.join(y_train).to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    X_val.join(y_val).to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    X_test.join(y_test).to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    print(f"\n" + "="*60)
    print("PRIpreMA PODATAKA ZAVRŠENA!")
    print("="*60)
    print(f"Podaci sačuvani u: {processed_dir}")
    print(f"Grafikoni sačuvani u: {fig_dir}")
    print(f"Broj atributa (features): {X_train.shape[1]}")


if __name__ == "__main__":
    save_processed_data(
        train_path='data/raw/adult_train.csv',
        test_path='data/raw/adult_test.csv',
        processed_dir='data/processed',
        fig_dir='results/figures'
    )