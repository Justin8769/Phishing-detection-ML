from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import re
import json
import socket
import ipaddress
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

try:
    import whois as python_whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'phishing_model.pkl')

model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

FEATURE_NAMES = [
    'Index', 'UsingIP', 'LongURL', 'ShortURL', 'Symbol@',
    'Redirecting//', 'PrefixSuffix-', 'SubDomains', 'HTTPS',
    'DomainRegLen', 'Favicon', 'NonStdPort', 'HTTPSDomainURL',
    'RequestURL', 'AnchorURL', 'LinksInScriptTags', 'ServerFormHandler',
    'InfoEmail', 'AbnormalURL', 'WebsiteForwarding', 'StatusBarCust',
    'DisableRightClick', 'UsingPopupWindow', 'IframeRedirection',
    'AgeofDomain', 'DNSRecording', 'WebsiteTraffic', 'PageRank',
    'GoogleIndex', 'LinksPointingToPage', 'StatsReport'
]

FETCH_TIMEOUT = 6  # seconds


LOGIN_PATHS = re.compile(
    r'/(login|signin|sign-in|account|verify|secure|update|confirm|banking|wp-admin)',
    re.IGNORECASE
)

SUSPICIOUS_SCORE_RANGE = (0.35, 0.65)  # ambiguous zone


def is_private_ip(host):
    """Return True if host is a private/loopback/link-local IP address."""
    try:
        addr = ipaddress.ip_address(host.split(':')[0])
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False


def get_domain_age_feature(domain):
    """Return 1 (>= 6 months old), -1 (< 6 months), or 0 (unknown) via WHOIS."""
    if not WHOIS_AVAILABLE:
        return 0
    try:
        w = python_whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation and isinstance(creation, datetime):
            return 1 if (datetime.now() - creation) > timedelta(days=180) else -1
    except Exception:
        pass
    return 0


def count_red_flags(url, features_list):
    """Count obvious phishing red flags for the rule-based override."""
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]
    idx = dict(zip(FEATURE_NAMES, features_list))
    flags = 0
    if idx['UsingIP'] == -1:
        flags += 1
    if idx['HTTPS'] == -1:
        flags += 1
    if idx['NonStdPort'] == -1:
        flags += 1
    if idx['ShortURL'] == -1:
        flags += 1
    if idx['Symbol@'] == -1:
        flags += 1
    if idx['PrefixSuffix-'] == -1:
        flags += 1
    if LOGIN_PATHS.search(parsed.path):
        flags += 1
    if is_private_ip(domain):
        flags += 1
    return flags


def fetch_html(url):
    """Fetch page HTML. Returns (soup, redirect_count) or (None, 0) on failure."""
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        resp = requests.get(url, timeout=FETCH_TIMEOUT, headers=headers,
                            allow_redirects=True, verify=False)
        redirect_count = len(resp.history)
        soup = BeautifulSoup(resp.text, 'html.parser')
        return soup, redirect_count
    except Exception:
        return None, 0


def check_dns(domain):
    """Return 1 if domain resolves via DNS, -1 if not or if private IP."""
    host = domain.split(':')[0]
    if is_private_ip(host):
        return -1  # private IPs are never legitimate public sites
    try:
        socket.gethostbyname(host)
        return 1
    except socket.gaierror:
        return -1


def get_favicon_feature(soup, domain):
    """
    1  = favicon loaded from same domain (legit)
    -1 = favicon loaded from external domain (phishing)
    """
    if soup is None:
        return 0
    icon = soup.find('link', rel=lambda r: r and 'icon' in ' '.join(r).lower())
    if icon and icon.get('href'):
        href = icon['href']
        if href.startswith('http'):
            parsed = urlparse(href)
            return 1 if domain in parsed.netloc else -1
    return 1  # relative path = same domain


def get_request_url_feature(soup, domain):
    """
    Ratio of external resource URLs (img, script, link).
    > 61% external  → -1
    22–61% external → 0
    < 22% external  → 1
    """
    if soup is None:
        return 0
    tags = (
        [(t.get('src') or '') for t in soup.find_all(['img', 'script'])] +
        [(t.get('href') or '') for t in soup.find_all('link')]
    )
    urls = [u for u in tags if u.startswith('http')]
    if not urls:
        return 1
    external = sum(1 for u in urls if domain not in urlparse(u).netloc)
    ratio = external / len(urls) * 100
    if ratio > 61:
        return -1
    if ratio >= 22:
        return 0
    return 1


def get_anchor_url_feature(soup, domain):
    """
    Ratio of <a href> pointing to different domain or '#' / 'javascript:'.
    > 67% suspicious → -1
    31–67%           → 0
    < 31%            → 1
    """
    if soup is None:
        return 0
    anchors = [a.get('href', '') for a in soup.find_all('a', href=True)]
    if not anchors:
        return 1
    suspicious = 0
    for href in anchors:
        if href.startswith('#') or href.lower().startswith('javascript'):
            suspicious += 1
        elif href.startswith('http') and domain not in urlparse(href).netloc:
            suspicious += 1
    ratio = suspicious / len(anchors) * 100
    if ratio > 67:
        return -1
    if ratio >= 31:
        return 0
    return 1


def get_links_in_script_tags_feature(soup, domain):
    """
    Ratio of external links in <script src> and <meta content>.
    > 81% external  → -1
    17–81% external → 0
    < 17% external  → 1
    """
    if soup is None:
        return 0
    srcs = (
        [s.get('src', '') for s in soup.find_all('script') if s.get('src')] +
        [m.get('content', '') for m in soup.find_all('meta') if m.get('content', '').startswith('http')]
    )
    if not srcs:
        return 1
    external = sum(1 for s in srcs if s.startswith('http') and domain not in urlparse(s).netloc)
    ratio = external / len(srcs) * 100
    if ratio > 81:
        return -1
    if ratio >= 17:
        return 0
    return 1


def get_server_form_handler_feature(soup, domain):
    """
    1  = all forms post to same domain
    0  = form posts to external domain
    -1 = form action is empty / about:blank
    """
    if soup is None:
        return 0
    forms = soup.find_all('form')
    if not forms:
        return 1
    for form in forms:
        action = form.get('action', '').strip()
        if not action or action.lower() in ('about:blank', '#', 'javascript:void(0)'):
            return -1
        if action.startswith('http') and domain not in urlparse(action).netloc:
            return 0
    return 1


def get_website_forwarding_feature(redirect_count):
    """
    <= 1 redirects → 1 (legit)
    2–3 redirects  → 0 (suspicious)
    >= 4 redirects → -1 (phishing)
    """
    if redirect_count <= 1:
        return 1
    if redirect_count <= 3:
        return 0
    return -1


def get_status_bar_cust_feature(soup):
    """
    -1 if page uses onmouseover to customise status bar
    1  otherwise
    """
    if soup is None:
        return 0
    html = str(soup)
    if 'onmouseover' in html.lower() and 'window.status' in html:
        return -1
    return 1


def get_disable_right_click_feature(soup):
    """
    -1 if page disables right-click
    1  otherwise
    """
    if soup is None:
        return 0
    html = str(soup)
    patterns = ['event.button==2', 'event.button == 2',
                'oncontextmenu', 'contextmenu']
    if any(p in html for p in patterns):
        return -1
    return 1


def get_popup_window_feature(soup):
    """
    -1 if page uses popup windows with input fields
    1  otherwise
    """
    if soup is None:
        return 0
    html = str(soup)
    if 'window.open' in html:
        return -1
    return 1


def get_iframe_feature(soup):
    """
    -1 if page contains <iframe>
    1  otherwise
    """
    if soup is None:
        return 0
    return -1 if soup.find('iframe') else 1


def extract_features(url):
    """Extract all 31 features from a URL, fetching page content for accuracy."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        # --- Fetch page once, reuse for all HTML-based features ---
        soup, redirect_count = fetch_html(url)

        features = {}

        # 1. Index
        features['Index'] = 0

        # 2. UsingIP
        ip_pattern = re.compile(
            r'(\d{1,3}\.){3}\d{1,3}|0x[0-9a-fA-F]{8}|0[0-7]{10}'
        )
        features['UsingIP'] = -1 if ip_pattern.search(domain) else 1

        # 3. LongURL
        url_len = len(url)
        if url_len < 54:
            features['LongURL'] = 1
        elif url_len <= 75:
            features['LongURL'] = 0
        else:
            features['LongURL'] = -1

        # 4. ShortURL
        shorteners = r'bit\.ly|goo\.gl|tinyurl|ow\.ly|t\.co|is\.gd|buff\.ly|adf\.ly|tiny\.cc'
        features['ShortURL'] = -1 if re.search(shorteners, url) else 1

        # 5. Symbol@
        features['Symbol@'] = -1 if '@' in url else 1

        # 6. Redirecting//
        features['Redirecting//'] = -1 if '//' in url[7:] else 1

        # 7. PrefixSuffix-
        features['PrefixSuffix-'] = -1 if '-' in domain else 1

        # 8. SubDomains
        dot_count = domain.count('.')
        if dot_count == 1:
            features['SubDomains'] = 1
        elif dot_count == 2:
            features['SubDomains'] = 0
        else:
            features['SubDomains'] = -1

        # 9. HTTPS
        features['HTTPS'] = 1 if parsed.scheme == 'https' else -1

        # 10. DomainRegLen
        features['DomainRegLen'] = 1 if len(domain) < 20 else -1

        # 11. Favicon — fetched from HTML
        features['Favicon'] = get_favicon_feature(soup, domain)

        # 12. NonStdPort
        port = parsed.port
        features['NonStdPort'] = -1 if port and port not in [80, 443] else 1

        # 13. HTTPSDomainURL
        features['HTTPSDomainURL'] = -1 if 'https' in domain.lower() else 1

        # 14. RequestURL — fetched from HTML
        features['RequestURL'] = get_request_url_feature(soup, domain)

        # 15. AnchorURL — fetched from HTML (23% importance!)
        features['AnchorURL'] = get_anchor_url_feature(soup, domain)

        # 16. LinksInScriptTags — fetched from HTML
        features['LinksInScriptTags'] = get_links_in_script_tags_feature(soup, domain)

        # 17. ServerFormHandler — fetched from HTML
        features['ServerFormHandler'] = get_server_form_handler_feature(soup, domain)

        # 18. InfoEmail
        features['InfoEmail'] = -1 if 'mailto:' in url else 1

        # 19. AbnormalURL
        features['AbnormalURL'] = -1 if domain not in url else 1

        # 20. WebsiteForwarding — based on HTTP redirect chain
        features['WebsiteForwarding'] = get_website_forwarding_feature(redirect_count)

        # 21. StatusBarCust — fetched from HTML
        features['StatusBarCust'] = get_status_bar_cust_feature(soup)

        # 22. DisableRightClick — fetched from HTML
        features['DisableRightClick'] = get_disable_right_click_feature(soup)

        # 23. UsingPopupWindow — fetched from HTML
        features['UsingPopupWindow'] = get_popup_window_feature(soup)

        # 24. IframeRedirection — fetched from HTML
        features['IframeRedirection'] = get_iframe_feature(soup)

        # 25. AgeofDomain — free WHOIS lookup
        host_only = domain.split(':')[0]
        features['AgeofDomain'] = (
            -1 if is_private_ip(host_only)
            else get_domain_age_feature(host_only)
        )

        # 26. DNSRecording — real DNS lookup (private IPs → -1)
        features['DNSRecording'] = check_dns(domain)

        # 27-31: No external API — use context-aware default.
        # If core signals are suspicious, lean -1; otherwise stay 0.
        suspicious_context = (
            features['UsingIP'] == -1 or
            features['HTTPS'] == -1 or
            features['DNSRecording'] == -1 or
            bool(LOGIN_PATHS.search(path))
        )
        default_unknown = -1 if suspicious_context else 0
        features['WebsiteTraffic'] = default_unknown
        features['PageRank'] = default_unknown
        features['GoogleIndex'] = default_unknown
        features['LinksPointingToPage'] = default_unknown
        features['StatsReport'] = default_unknown

        return [features[f] for f in FEATURE_NAMES]

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return [0] * len(FEATURE_NAMES)


def get_feature_analysis(url, features_list):
    """Build the feature analysis cards shown in the UI."""
    parsed = urlparse(url)
    domain = parsed.netloc
    idx = {name: val for name, val in zip(FEATURE_NAMES, features_list)}

    def label(val):
        if val == 1:
            return 'Safe'
        if val == -1:
            return 'Suspicious'
        return 'Uncertain'

    return [
        {"name": "URL Length",         "value": len(url),                                        "suspicious": len(url) > 75},
        {"name": "HTTPS",              "value": "Yes" if parsed.scheme == 'https' else "No",     "suspicious": parsed.scheme != 'https'},
        {"name": "Has @ Symbol",       "value": "Yes" if '@' in url else "No",                   "suspicious": '@' in url},
        {"name": "Has IP Address",     "value": "Yes" if idx['UsingIP'] == -1 else "No",         "suspicious": idx['UsingIP'] == -1},
        {"name": "Has Hyphen",         "value": "Yes" if '-' in domain else "No",                "suspicious": '-' in domain},
        {"name": "Sub Domains",        "value": domain.count('.'),                               "suspicious": domain.count('.') > 2},
        {"name": "URL Shortener",      "value": "Yes" if idx['ShortURL'] == -1 else "No",        "suspicious": idx['ShortURL'] == -1},
        {"name": "Double Slash",       "value": "Yes" if idx['Redirecting//'] == -1 else "No",   "suspicious": idx['Redirecting//'] == -1},
        {"name": "Anchor URLs",        "value": label(idx['AnchorURL']),                         "suspicious": idx['AnchorURL'] != 1},
        {"name": "Ext. Resources",     "value": label(idx['RequestURL']),                        "suspicious": idx['RequestURL'] != 1},
        {"name": "Form Handler",       "value": label(idx['ServerFormHandler']),                 "suspicious": idx['ServerFormHandler'] != 1},
        {"name": "DNS Record",         "value": "Found" if idx['DNSRecording'] == 1 else "Missing", "suspicious": idx['DNSRecording'] != 1},
        {"name": "Redirects",          "value": label(idx['WebsiteForwarding']),                 "suspicious": idx['WebsiteForwarding'] != 1},
        {"name": "iFrame",             "value": "Detected" if idx['IframeRedirection'] == -1 else "None", "suspicious": idx['IframeRedirection'] == -1},
        {"name": "Popup Window",       "value": "Yes" if idx['UsingPopupWindow'] == -1 else "No","suspicious": idx['UsingPopupWindow'] == -1},
        {"name": "Right-Click Block",  "value": "Yes" if idx['DisableRightClick'] == -1 else "No","suspicious": idx['DisableRightClick'] == -1},
    ]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not url.startswith('http'):
            url = 'http://' + url

        features = extract_features(url)
        features_array = pd.DataFrame([features], columns=FEATURE_NAMES)

        prediction = model.predict(features_array)[0]
        score = float(model.predict_proba(features_array)[0][1])
        is_phishing = bool(prediction == 1)

        # Rule-based override: when ML is ambiguous, count hard red flags
        red_flags = count_red_flags(url, features)
        low, high = SUSPICIOUS_SCORE_RANGE
        if low <= score <= high and red_flags >= 2:
            is_phishing = True
            score = max(score, 0.70)  # push confidence up to reflect the flags

        return jsonify({
            'url': url,
            'is_phishing': is_phishing,
            'confidence': round(score * 100, 2),
            'label': 'Phishing' if is_phishing else 'Legitimate',
            'red_flags': red_flags,
            'features': get_feature_analysis(url, features)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
