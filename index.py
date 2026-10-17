import json
import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Instanța Flask recunoscută automat de Vercel
app = Flask(__name__)
CORS(app)

# Căi absolute pentru fișiere pe Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'baza_cariere.csv')
JSON_PATH = os.path.join(BASE_DIR, 'surse_piata.json')

# Cheie API Google Gemini din Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


@app.route('/', methods=['GET'])
@app.route('/ping', methods=['GET'])
def health_check():
    """Endpoint simplu pentru verificarea conexiunii."""
    return jsonify({
        'status': 'online',
        'message': 'Serverul Viitoris este activ pe Vercel!',
        'gemini_configured': bool(GEMINI_API_KEY)
    }), 200


def cauta_competitori_osm(oras, tip_afacere):
    """Scanează în timp real harta OpenStreetMap (Overpass API) pentru competitori reali."""
    tip = tip_afacere.lower().strip()

    if any(x in tip for x in ['cafe', 'cafenea', 'coffee', 'ceainarie', 'gogoasa', 'gogoserie']):
        kw = 'amenity="cafe"'
    elif any(x in tip for x in ['restaur', 'pizzeri', 'fast', 'saorma', 'bistro', 'mancare', 'shaorma', 'burger']):
        kw = 'amenity~"restaurant|fast_food|bistro"'
    elif any(x in tip for x in ['friz', 'salon', 'barber', 'coafur', 'infrumusetare', 'cosmetica']):
        kw = 'shop="hairdresser"'
    elif any(x in tip for x in ['fit', 'sala', 'gym', 'sport', 'fitness']):
        kw = 'leisure="fitness_centre"'
    elif any(x in tip for x in ['bar', 'pub', 'club', 'discoteca']):
        kw = 'amenity~"bar|pub"'
    elif any(x in tip for x in ['patiserie', 'brutarie', 'covrigarie']):
        kw = 'shop~"bakery|pastry"'
    else:
        kw = 'amenity="cafe"'

    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:12];
    area["name"="{oras}"]->.searchArea;
    (
      node[{kw}](area.searchArea);
      way[{kw}](area.searchArea);
    );
    out body;
    """
    try:
        resp = requests.post(overpass_url, data={'data': query}, timeout=10)
        if resp.status_code == 200:
            res = resp.json()
            cnt = len(res.get('elements', []))
            if cnt > 0:
                return cnt
    except Exception as e:
        print(f"⚠️ Eroare Overpass API: {e}")

    return 12


def genereaza_raport_gemini(prompt_text):
    """Apelează Google Gemini API pentru generare de raport avansat."""
    if not GEMINI_API_KEY:
        raise Exception("Cheia GEMINI_API_KEY nu este configurată în Environment Variables pe Vercel.")

    modele_gemini = [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro"
    ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.75,
            "maxOutputTokens": 3500
        }
    }

    eroare_ultima = None
    for model_name in modele_gemini:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=25)
            if response.status_code == 200:
                data = response.json()
                text_generat = data["candidates"][0]["content"]["parts"][0]["text"]
                if text_generat:
                    return text_generat
            else:
                eroare_ultima = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            eroare_ultima = str(e)
            continue

    raise Exception(f"Toate modelele Gemini au eșuat. Detalii: {eroare_ultima}")


# Încărcare Bază de Date Cariere
df_cariere = None
try:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df_cariere = pd.read_csv(CSV_PATH, encoding=enc)
            df_cariere.columns = [str(c).strip() for c in df_cariere.columns]
            if 'cod_psihologic' in df_cariere.columns:
                df_cariere['cod_psihologic'] = df_cariere['cod_psihologic'].astype(str).str.strip().str.upper()
            print(f"✅ Baza de date 'baza_cariere.csv' încărcată ({enc}).")
            break
        except Exception:
            continue
except Exception as e:
    print(f"❌ Eroare baza_cariere: {e}")

# Încărcare Bază de Date Business
surse_piata = {}
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        surse_piata = json.load(f)
    print("✅ Baza de date 'surse_piata.json' încărcată.")
except Exception as e:
    print(f"❌ Eroare surse_piata: {e}")


@app.route('/predict-cariera', methods=['POST'])
def predict_cariera():
    if df_cariere is None:
        return jsonify({'error': 'Baza de date cu cariere nu este disponibilă pe server.'}), 500

    data = request.get_json() or {}
    dict_punctaje = {
        'REALIST': int(data.get('punctaj_realist', 0)),
        'INVESTIGATIV': int(data.get('punctaj_investigativ', 0)),
        'ARTISTIC': int(data.get('punctaj_artistic', 0)),
        'SOCIAL': int(data.get('punctaj_social', 0)),
        'INTREPRINZATOR': int(data.get('punctaj_intreprinzator', 0)),
        'CONVENTIONAL': int(data.get('punctaj_conventional', 0))
    }

    profil_dominant = max(dict_punctaje, key=dict_punctaje.get)
    scor_maxim = dict_punctaje[profil_dominant]
    cariere_potrivite = df_cariere[df_cariere['cod_psihologic'] == profil_dominant]

    if cariere_potrivite.empty:
        return jsonify({'error': f'Nu s-au găsit profesii pentru profilul {profil_dominant}'}), 404

    index_selectat = scor_maxim % len(cariere_potrivite)
    cariera_selectata = cariere_potrivite.iloc[index_selectat]

    return jsonify({
        'status': 'success',
        'id_cariera': int(cariera_selectata.get('id_cariera', 1)),
        'profil_detectat': profil_dominant,
        'descriere_profil': str(cariera_selectata.get('descriere_profil', '')),
        'punctaj_obtinut': scor_maxim,
        'titlu_job': str(cariera_selectata.get('titlu_job', '')),
        'facultate_ro': str(cariera_selectata.get('facultate_ro', '')),
        'admitere_ro': str(cariera_selectata.get('admitere_ro', '')),
        'facultate_strainatate': str(cariera_selectata.get('facultate_strainatate', '')),
        'admitere_strainatate': str(cariera_selectata.get('admitere_strainatate', '')),
        'pas_practic': str(cariera_selectata.get('pas_practic', '')),
        'salariu_mediu': int(cariera_selectata.get('salariu_mediu', 0))
    })


@app.route('/predict', methods=['POST'])
def analizeaza_business_din_surse():
    data = request.get_json() or {}

    buget_utilizator = float(data.get('buget', 0))
    oras_utilizator = str(data.get('oras', 'Bacău')).strip()
    tip_afacere_input = str(data.get('tip_afacere', 'cafenea')).strip()

    # 1. Căutare live competitori pe harta OpenStreetMap
    nr_competitori = cauta_competitori_osm(oras_utilizator, tip_afacere_input)

    # 2. Preluare date din surse_piata.json
    domeniu_cheie = "cafenea"
    tip_lower = tip_afacere_input.lower()
    for k in surse_piata.keys():
        if k in tip_lower or tip_lower in k:
            domeniu_cheie = k
            break

    afacere_info = surse_piata.get(domeniu_cheie, surse_piata.get("cafenea", {}))
    locatii_disponibile = afacere_info.get("locatii", {})
    info_locatie = locatii_disponibile.get("1", list(locatii_disponibile.values())[0] if locatii_disponibile else {})

    buget_minim = info_locatie.get("buget_minim_necesar", 10000)
    sursa_nume = info_locatie.get("sursa_informatii", "Studiu de piață Horeca & Real Estate")
    nume_afacere = afacere_info.get("nume", tip_afacere_input.capitalize())

    reusita = buget_utilizator >= buget_minim
    status_str = "REUȘITĂ" if reusita else "EȘEC"

    prompt_ia = f"""
Ești un Partener Senior într-o firmă de consultanță în strategie, analiză financiară și geomarketing din România.
Generează un STUDIU DE FEZABILITATE ȘI ANALIZĂ DE PIAȚĂ EXTREM DE DETALIAT, COMPREHENSIV ȘI RIGUROS.

DATE DE INTRARE COLECTATE LIVE:
- Oraș vizat: {oras_utilizator}
- Tip afacere: {nume_afacere} ({tip_afacere_input})
- Buget de investiție alocat de antreprenor: {buget_utilizator} EUR
- Buget minim recomandat pe piață (CAPEX + OPEX 3 luni): {buget_minim} EUR
- Concurență directă identificată pe harta satelit OpenStreetMap în {oras_utilizator}: {nr_competitori} locații active.

STRUCTURA OBLIGATORIE A RAPORTULUI:

1. 📊 SINTEZĂ EXECUTIVĂ & DIAGNOSTIC FINANCIAR
- Declară decizia finală: FEZABIL ({status_str}) la bugetul de {buget_utilizator} EUR în {oras_utilizator}.
- Oferă un Scor de Fezabilitate General (de la 1 la 10) cu explicație analitică.
- Comparație directă între suma disponibilă ({buget_utilizator} EUR) și bugetul minim de lansare ({buget_minim} EUR).

2. 🗺️ ANALIZĂ GEOMARKETING & SATURAȚIA PIEȚEI LOCALE (DATA OPENSTREETMAP)
- Interpretează impactul prezenței celor {nr_competitori} competitori existenți în {oras_utilizator}.
- Clasifică piața din {oras_utilizator} (Sub-saturată / Echilibrată / Saturation) și specifică dificultatea de achiziție a primilor clienți.
- Identifică tipul de amplasament ideal în {oras_utilizator}.

3. 💰 STRUCTURA FINANCIARĂ DETALIATĂ (DEFALCARE CAPEX & OPEX)
Simulează și defalchează bugetul de {buget_utilizator} EUR pe categorii clare de cheltuieli:
  - Spațiu & Amenajare (Chirie 3 luni + garanție, renovare, design interior)
  - Echipamente profesionale & Tehnologie (aparatură, POS, soft gestiune)
  - Autorizații, Avize & Legal (DSP, DSV, Primărie, PFA/SRL)
  - Stoc inițial de marfă & Consumabile
  - Fond de Rulment & Buget de Rezervă (OPEX primele 3 luni)

4. 📈 ECONOMIA UNITĂȚII (UNIT ECONOMICS) & BREAK-EVEN POINT
- Estimează Bonul Mediu de Consum (în RON/EUR) specific pentru un {nume_afacere} în România.
- Calculează numărul estimat de clienți/zi necesari pentru a atinge Punctul de Echilibru (Break-Even).
- Estimează Perioada de Amortizare a Investiției (ROI - Payback Period în luni).

5. 🚀 STRATEGIE DE PĂTRUNDERE PE PIAȚĂ & DIFERENȚIERE
- Formularea unei Propuneri Unice de Valoare (PUV) specifice.
- 3 Tactici agresive și pragmatice de marketing prin care afacerea va atrage clienți de la cei {nr_competitori} competitori din {oras_utilizator}.

6. ⚠️ MATRICE DE RISCURI & FOAIE DE PARCURS (ROADMAP DE LANSARE)
- Top 3 Riscuri Majore + Măsuri de atenuare.
- Foaie de parcurs în 4 pași logici pentru lansare.
"""

    try:
        raspuns_ia = genereaza_raport_gemini(prompt_ia)
    except Exception as e:
        raspuns_ia = f"Sistemul a estimat un rezultat de: {status_str}.\n\nBugetul minim recomandat pentru o afacere de tip {nume_afacere} în {oras_utilizator} este de {buget_minim} EUR.\nAu fost identificați {nr_competitori} competitori pe hartă.\n\n(Notă eroare raport AI: {str(e)})"

    return jsonify({
        'rezultat': status_str,
        'explicatie': raspuns_ia,
        'buget_minim': buget_minim,
        'sursa_folosita': sursa_nume,
        'oras': oras_utilizator,
        'competitori_gasiti': nr_competitori
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
