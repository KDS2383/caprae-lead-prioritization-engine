from flask import Flask, render_template, request, jsonify
import os
import random
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from threading import Lock
import re

# New imports for scraping
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import concurrent.futures

app = Flask(__name__)

# --- SECRETS & CONFIG ---
# IMPORTANT: Set these as environment variables for security.
# On Windows: set GOOGLE_API_KEY=your_key
# On Mac/Linux: export GOOGLE_API_KEY=your_key
# Do the same for GOOGLE_CSE_ID

# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyBINwHWXeY7L1l-X4Tk-qoFWe_LPyUbrmg"

# GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_CSE_ID = "55c8c08fe9a5e496c"

# --- GLOBAL MODEL & LOCK ---
XGB_RANKER_MODEL = None
model_lock = Lock()

# --- SHARED CONFIG & HELPERS ---
ICP_VECTOR_RAW = np.array([[10.0, 150, 4.0]])
BBB_RATING_MAP = {'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'N/A': 0}

# --- LTR MODEL TRAINING (remains the same as before) ---
def train_ltr_model():
    # This function is unchanged from the previous version.
    global XGB_RANKER_MODEL
    if XGB_RANKER_MODEL is not None: return
    print("--- Training Learning-to-Rank (LTR) Model ---")
    mock_training_leads = [{"name": f"TrainCo-{i}", "revenue_mil": round(random.uniform(0.1, 50.0), 1), "employees": random.randint(5, 1000), "bbb_rating": random.choice(list(BBB_RATING_MAP.keys()))} for i in range(100)]
    X_train = _create_feature_vectors(mock_training_leads)
    y_train = _generate_relevance_labels(X_train, ICP_VECTOR_RAW)
    group_train = [len(X_train)]
    model = xgb.XGBRanker(objective='rank:pairwise', learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train, group=group_train, verbose=False)
    XGB_RANKER_MODEL = model
    print("--- LTR Model Training Complete ---")

# (Helper functions _create_feature_vectors and _generate_relevance_labels are also unchanged)
def _create_feature_vectors(leads):
    return np.array([ [lead['revenue_mil'], lead['employees'], BBB_RATING_MAP.get(lead['bbb_rating'], 0)] for lead in leads ])
def _generate_relevance_labels(lead_vectors, icp_vector):
    from sklearn.metrics.pairwise import cosine_similarity
    scaler = StandardScaler().fit(np.vstack([lead_vectors, icp_vector]))
    scaled_leads = scaler.transform(lead_vectors)
    scaled_icp = scaler.transform(icp_vector)
    similarities = cosine_similarity(scaled_leads, scaled_icp).flatten()
    labels = np.zeros(len(similarities))
    labels[similarities > 0.85] = 2
    labels[similarities > 0.5] = 1
    return labels

@app.before_request
def initialize_model():
    with model_lock:
        if XGB_RANKER_MODEL is None:
            train_ltr_model()

# --- REAL-TIME SCRAPING FUNCTIONS ---

def get_google_search_results(query):
    """Queries the Google Custom Search API to get a list of company URLs."""
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=10).execute() # Get top 10 results
        items = res.get('items', [])
        urls = [item['link'] for item in items if 'link' in item]
        return urls
    except Exception as e:
        print(f"Error fetching Google results: {e}")
        return []

def scrape_website_data(url):
    """Scrapes a single website to extract features for our model."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text().lower()

        # --- Creative Feature Extraction from Raw Text ---
        # Revenue (highly heuristic, looks for patterns like "$10M", "10 million")
        revenue_mil = 0
        revenue_match = re.search(r'\$?(\d+\.?\d*)\s*m(illion)?', text)
        if revenue_match:
            revenue_mil = float(revenue_match.group(1))

        # Employees (looks for "employees" or "team members")
        employees = 50 # Default assumption
        employees_match = re.search(r'(\d+)\s*(employees|team)', text)
        if employees_match:
            employees = int(employees_match.group(1))

        # For this demo, we'll assign a default BBB rating. A real system could use another API for this.
        bbb_rating = random.choice(['A', 'B', 'C'])

        # Get company name from title tag
        title_tag = soup.find('title')
        company_name = title_tag.string.split('|')[0].strip() if title_tag else url

        return {
            "name": company_name,
            "url": url,
            "revenue_mil": revenue_mil,
            "employees": employees,
            "bbb_rating": bbb_rating
        }
    except Exception as e:
        print(f"Could not scrape {url}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_leads():
    data = request.get_json()
    industry = data.get('industry')
    location = data.get('location')

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return jsonify({"error": "Server is not configured with Google API credentials."}), 500

    # 1. Get URLs from Google Search
    query = f'"{industry}" companies in "{location}"'
    urls = get_google_search_results(query)

    if not urls:
        return jsonify([])

    # 2. Scrape websites concurrently for speed
    scraped_leads = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_website_data, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            if result:
                scraped_leads.append(result)

    if not scraped_leads:
        return jsonify([])

    # 3. Use the LTR model to rank the REAL, scraped leads
    X_live = _create_feature_vectors(scraped_leads)
    predicted_scores = XGB_RANKER_MODEL.predict(X_live)
    
    for lead, score in zip(scraped_leads, predicted_scores):
        lead['lead_score'] = round(float(score), 4)
    
    prioritized_leads = sorted(scraped_leads, key=lambda x: x['lead_score'], reverse=True)
    
    return jsonify(prioritized_leads)

if __name__ == '__main__':
    # The model training will happen before the first request now
    app.run(debug=True)