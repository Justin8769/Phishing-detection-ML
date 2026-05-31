# Phishing Website Detection using Machine Learning

## Overview
A machine learning system that detects phishing websites by analysing 
URL and domain-based features. Built as part of an MSc Cybersecurity 
project at the University of Hertfordshire.

The system achieves **97.1% accuracy** using a Random Forest classifier 
trained on 11,055 URL samples with 30 features including HTTPS usage, 
URL length, anchor tags, and domain age indicators.

## Tech Stack
- **Python** — core programming language
- **Scikit-learn** — model training and evaluation
- **Random Forest** — primary deployed model (97.1% accuracy)
- **Logistic Regression** — comparison model (92.3% accuracy)
- **Pandas & NumPy** — data processing and feature engineering
- **Joblib** — model serialisation and loading
- **Flask** — web framework and REST API (`/predict` endpoint)
- **Chart.js** — client-side performance visualisation

## Dataset
UCI Machine Learning Repository — *Phishing Websites Dataset*  
Mohammad, Thabtah & McCluskey (2012)  
11,055 samples · 30 features · Labels encoded as -1 / 0 / 1

## Project Structure
- `phishing_detection.ipynb` — ML model training and evaluation
- `app.py` — Flask web application with prediction API
- `phishing.csv` — dataset
- `phishing_model.pkl` — trained Random Forest model

## MSc Cybersecurity | University of Hertfordshire | 2025–2026
