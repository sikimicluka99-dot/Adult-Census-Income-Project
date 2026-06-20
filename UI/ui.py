import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Konfiguracija stranice
st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="wide"
)

# Tema
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
    }
    .prediction-high {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .prediction-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Naslov
st.markdown('<div class="main-header">💰 Adult Census Income Predictor</div>', unsafe_allow_html=True)
st.markdown("### Predviđanje da li osoba zarađuje **više od 50.000$** godišnje")
st.markdown("---")

# Učitavanje modela
@st.cache_resource
def load_model():
    """Učitava najbolji model i scaler."""
    models_dir = 'models'
    
    # Učitaj model
    model_path = os.path.join(models_dir, 'gradient_boosting.pkl')
    if not os.path.exists(model_path):
        st.error(f"❌ Model nije pronađen na: {model_path}")
        st.info("Pokrenite prvo `python src/train.py` da trenirate model.")
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join('data/processed', 'scaler.pkl'))
    threshold = joblib.load(os.path.join(models_dir, 'best_threshold.pkl'))
    
    return model, scaler, threshold

# Učitaj model
model, scaler, threshold = load_model()

if model is not None:
    st.success(f"✅ Model učitan: Gradient Boosting (threshold={threshold:.2f})")
    
    # Dve kolone za unos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Osnovne informacije")
        
        age = st.slider(
            "Godine starosti",
            min_value=17, max_value=90, value=35,
            help="Starost osobe u godinama"
        )
        
        sex = st.selectbox(
            "Pol",
            options=["Male", "Female"]
        )
        
        education_num = st.selectbox(
            "Nivo obrazovanja (1-16)",
            options=list(range(1, 17)),
            index=12,  # Bachelors = 13
            help="1=Preschool, 9=HS-grad, 10=Some-college, 13=Bachelors, 14=Masters, 16=Doctorate"
        )
        
        marital_status = st.selectbox(
            "Bračno stanje",
            options=[
                "Never-married", "Married-civ-spouse", "Married-spouse-absent",
                "Married-AF-spouse", "Divorced", "Separated", "Widowed"
            ]
        )
        
        relationship = st.selectbox(
            "Porodični odnos",
            options=[
                "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
            ]
        )
        
        race = st.selectbox(
            "Rasa",
            options=["White", "Black", "Asian-Pac-Islander", "Other", "Amer-Indian-Eskimo"]
        )
    
    with col2:
        st.subheader("💼 Radne informacije")
        
        workclass = st.selectbox(
            "Klasa posla",
            options=[
                "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                "Local-gov", "State-gov", "Without-pay", "Never-worked"
            ]
        )
        
        occupation = st.selectbox(
            "Zanimanje",
            options=[
                "Tech-support", "Craft-repair", "Other-service", "Sales",
                "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
            ]
        )
        
        hours_per_week = st.slider(
            "Radni sati nedeljno",
            min_value=1, max_value=99, value=40
        )
        
        capital_gain = st.number_input(
            "Kapitalni dobitak (capital gain)",
            min_value=0, max_value=99999, value=0
        )
        
        capital_loss = st.number_input(
            "Kapitalni gubitak (capital loss)",
            min_value=0, max_value=99999, value=0
        )
        
        native_country = st.selectbox(
            "Zemlja porekla",
            options=[
                "United-States", "Mexico", "Philippines", "Germany", "Canada",
                "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
                "Jamaica", "South", "China", "Italy", "Dominican-Republic",
                "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
                "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua",
                "Peru", "France", "Greece", "Ecuador", "Ireland",
                "Hong", "Trinadad&Tobago", "Cambodia", "Laos", "Thailand",
                "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary", "Scotland",
                "Holand-Netherlands"
            ]
        )
    
    # Dugme za predikciju
    st.markdown("---")
    predict_button = st.button("🔮 PREDVIĐAJ PRIHOD", use_container_width=True)
    
    if predict_button:
        with st.spinner("Analiziram podatke..."):
            try:
                # Pravljenje DataFrame-a sa unosima
                input_data = pd.DataFrame({
                    'age': [age],
                    'workclass': [workclass],
                    'education_num': [education_num],
                    'marital_status': [marital_status],
                    'occupation': [occupation],
                    'relationship': [relationship],
                    'race': [race],
                    'sex': [sex],
                    'capital_gain': [capital_gain],
                    'capital_loss': [capital_loss],
                    'hours_per_week': [hours_per_week],
                    'native_country': [native_country]
                })
                
                # One-hot encoding
                cat_cols = ['workclass', 'marital_status', 'occupation',
                            'relationship', 'race', 'sex', 'native_country']
                input_data = pd.get_dummies(input_data, columns=cat_cols, drop_first=True)
                
                # Učitaj trening kolone (da poravnamo)
                train_df = pd.read_csv('data/processed/train.csv')
                train_cols = train_df.drop(columns=['income']).columns
                
                # Dodaj nedostajuće kolone
                for col in train_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0
                
                # Isti redosled kolona
                input_data = input_data[train_cols]
                
                # Skaliranje numeričkih kolona
                num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
                input_data[num_cols] = scaler.transform(input_data[num_cols])
                
                # Predikcija
                y_prob = model.predict_proba(input_data)[0][1]
                y_pred = 1 if y_prob > threshold else 0
                
                # Prikaz rezultata
                if y_pred == 1:
                    st.markdown(f"""
                        <div class="prediction-box prediction-high">
                            🎉 PREDIKCIJA: <span style="font-size: 2rem;">>50K$</span><br>
                            <span style="font-size: 1rem;">Verovatnoća: {y_prob*100:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""
                        <div class="prediction-box prediction-low">
                            📊 PREDIKCIJA: <span style="font-size: 2rem;">≤50K$</span><br>
                            <span style="font-size: 1rem;">Verovatnoća: {(1-y_prob)*100:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Prikaz unetih podataka
                with st.expander("📋 Pregled unetih podataka"):
                    st.write(f"- **Godine:** {age}")
                    st.write(f"- **Pol:** {sex}")
                    st.write(f"- **Obrazovanje:** nivo {education_num}")
                    st.write(f"- **Bračno stanje:** {marital_status}")
                    st.write(f"- **Zanimanje:** {occupation}")
                    st.write(f"- **Klasa posla:** {workclass}")
                    st.write(f"- **Radni sati:** {hours_per_week}h nedeljno")
                    st.write(f"- **Kapitalni dobitak:** ${capital_gain:,}")
                    st.write(f"- **Kapitalni gubitak:** ${capital_loss:,}")
                    st.write(f"- **Zemlja porekla:** {native_country}")
                    
            except Exception as e:
                st.error(f"Greška pri predikciji: {e}")
    
    # Footer
    st.markdown("---")
    st.caption("📊 Model: Gradient Boosting | 🔮 Threshold: " + str(threshold))
    st.caption("ℹ️ Ovo je demonska aplikacija – rezultati nisu 100% tačni za stvarnu upotrebu.")

else:
    st.error("❌ Model nije učitan. Pokrenite `python src/train.py` prvo.")