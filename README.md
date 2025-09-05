Dynamic Pricing Engine 💰

📌 Overview

This project implements a Dynamic Pricing Engine that simulates real-world retail market scenarios, predicts demand, and optimizes product pricing strategies using Machine Learning and Reinforcement Learning.
It consists of:
Synthetic dataset generation (products, cities, seasons, weather).
Demand prediction models (Ridge, Lasso, Random Forest).
Q-learning agent for reinforcement learning–based pricing.
Market Simulator to test price-demand-profit relationships.
Streamlit Dashboard for interactive analytics & monitoring.

🚀 Features

Generates 1M+ synthetic transactions with realistic pricing and demand patterns.
Models demand using Random Forest Regression with feature engineering.
Hyperparameter tuning with GridSearchCV.
Q-learning agent learns optimal markup strategies.
Visualization of Price vs Demand & Profit.
Streamlit UI for dashboards, simulation, live monitoring.
Saves models (tuned_demand_model.pkl, simulator_engine.pkl) and Q-table (q_learning_strategy.csv).

📂 Project Structure
├── synthetic_pricing_dataset_v4_10lac.csv   # Generated dataset
├── simulator_engine.pkl                     # Trained Random Forest simulator
├── model_columns.json                       # Feature column mapping
├── tuned_demand_model.pkl                   # Tuned Random Forest model
├── q_learning_strategy.csv                  # Learned Q-table
├── apps8.py                                 # Streamlit dashboard
├── 7df9a8fa-eb59-43e7-8543-12df47d926f1.py  # Main Python script
└── README.md                                # Project documentation

⚙️ Installation

Clone the repository and install dependencies:
git clone https://github.com/your-username/dynamic-pricing-engine.git
cd dynamic-pricing-engine
pip install -r requirements.txt
Dependencies
Python 3.8+
pandas, numpy
scikit-learn
matplotlib, seaborn, plotly
streamlit
joblib
pyngrok
Install them manually if needed:
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib pyngrok

▶️ Usage

1. Generate Dataset
python 7df9a8fa-eb59-43e7-8543-12df47d926f1.py
2. Train Model & Q-learning Agent
Model training and tuning runs automatically.
The trained models and Q-learning strategy are saved.
3. Run Streamlit Dashboard
streamlit run apps8.py

📊 Dashboard Feature
Executive Overview → Revenue, Profit, Sales, Trends.
AI-Powered Simulator → Predict demand & profit under new pricing.
Q-Learning Strategy → Visualize learned strategies.
Performance Analytics → Seasonal, weather & city-based analysis.
Live Monitoring → Real-time market simulation.
<img width="1870" height="912" alt="image" src="https://github.com/user-attachments/assets/118f2196-be88-4881-90ef-9b6823e0b817" />


🤖 Q-Learning Agent

Actions: markup_10, markup_20, markup_30, markup_40, markup_50.
Reward: Profit from simulated market response.
Learns an optimal pricing policy for different products & contexts.

📌 Example Output

Price vs Demand & Profit curves.
<img width="962" height="636" alt="graph3" src="https://github.com/user-attachments/assets/22907a82-a6f7-4645-a511-ab442643fac1" />

Q-table heatmap showing best markup strategies.
Streamlit live dashboard with interactive visualizations.
