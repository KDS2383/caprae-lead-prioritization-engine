# AI-Powered Lead Prioritization Engine

### A Submission for the Caprae Capital Intern Interview Pre-Work

This project is a fully functional web application designed to solve a critical business problem: **lead prioritization**. Instead of just scraping a list of companies, this tool uses a Learning-to-Rank (LTR) machine learning model to analyze and rank real-world leads in real-time, enabling sales teams to focus their efforts on the most promising opportunities first.

This solution directly addresses the core challenge of moving beyond simple data collection to providing actionable, intelligent insights, embodying the Caprae philosophy of using technology to drive operational efficiency and growth.

---

### Live Demo

https://caprae-lead-ranker.onrender.com

---

### Key Features & Technical Architecture

The architecture is designed to be both powerful and scalable, moving from simple rules to a data-driven system.

1.  **Real-Time Data Acquisition:**
    *   The user provides an `Industry` and `Location`.
    *   The backend queries the **Google Custom Search API** to retrieve a list of relevant company websites.

2.  **Concurrent Web Scraping:**
    *   A multi-threaded scraper visits each URL simultaneously to download and parse the website's HTML using **BeautifulSoup**.
    *   It heuristically extracts key firmographic features (e.g., revenue figures, employee counts) from unstructured text.

3.  **Learning-to-Rank (LTR) Inference Engine:**
    *   This is the core of the application. On startup, a **Gradient Boosting model (XGBoost)** is trained on a synthetic dataset to learn the complex, non-linear patterns that define a high-value lead.
    *   The real, scraped data is then fed into this pre-trained `XGBRanker` model, which predicts a relevance score for each lead. This moves beyond simple rule-based scoring to a model that has *learned* what a good lead looks like.

4.  **Prioritized Results:**
    *   The final list of leads is sorted by the model's predicted rank score and displayed in a clean, intuitive frontend built with **Flask, HTML, CSS, and JavaScript**.

---

### Tech Stack

*   **Backend:** Python, Flask
*   **Machine Learning:** Scikit-learn, XGBoost, NumPy
*   **Data Acquisition:** Google Custom Search API, Requests, BeautifulSoup4
*   **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

---

### Setup and Running the Application

Follow these steps to run the project locally.

#### 1. Prerequisites
- Python 3.9+
- Git

#### 2. Clone the Repository
```bash
git clone https://github.com/KDS2383/caprae-lead-prioritization-engine.git
cd caprae-lead-prioritization-engine
