## 🚀 Uputstvo za pokretanje projekta

#1. Kloniranje ili preuzimanje projekta

```bash
git clone <link-do-repozitorijuma>
cd "Adult census income - project"
# Kreiraj virtuelno okruženje
python -m venv venv

# Aktiviraj (Windows)
venv\Scripts\activate

# Aktiviraj (Mac/Linux)
source venv/bin/activate

#Instalacija potrebnih biblioteka 
pip install -r requirements.txt

#Priprema podataka
python src/prepare_data.py

#Treniranje modela
python src/train.py

#Evaluacija modela
python src/evaluate.py

#Pokretanje korisničkog interfejsa (UI)
streamlit run app.py
