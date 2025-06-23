from flask import Flask, jsonify
import requests
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PPOAgent(nn.Module):
    def __init__(self):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.stop_loss = 0.05
        self.last_prices = {'SPK.NZ': None, 'BTC': None, 'ETH': None, 'XRP': None, 'NZDUSD=X': None}

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        action = np.random.choice(['buy', 'sell', 'hold'], p=probs[0].numpy())
        return action

    def update_last_price(self, asset, price):
        self.last_prices[asset] = price

agent = PPOAgent()
predictor = LSTMPredictor()

def calculate_rsi(prices):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1e-10
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_sentiment(asset):
    # Mock Grok 3 sentiment; replace with actual API call
    return np.random.uniform(-1, 1)

def fetch_historical_data(asset):
    ticker = {'SPK.NZ': 'SPK.NZ', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'XRP': 'XRP-USD', 'NZDUSD=X': 'NZDUSD=X'}[asset]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1h')
    return data['Close'].values

def predict_price(history):
    history = (history - np.mean(history)) / np.std(history)
    history_tensor = torch.FloatTensor(history).reshape(1, -1, 1)
    with torch.no_grad():
        return predictor(history_tensor).item()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        response = requests.get('https://api.supergrok.nz/api/market-data', headers={'Authorization': 'Bearer YOUR_TOKEN'})
        data = response.json()
        prices = data['prices']
        
        actions = {}
        for asset in ['SPK.NZ', 'BTC', 'ETH', 'XRP', 'NZDUSD=X']:
            history = fetch_historical_data(asset)
            rsi = calculate_rsi(history)
            sentiment = fetch_sentiment(asset)
            forecast = predict_price(history)
            state = [prices[asset], rsi, 1000, sentiment, forecast, 0]
            action = agent.act(state)
            if agent.last_prices[asset] and prices[asset] < agent.last_prices[asset] * (1 - agent.stop_loss):
                action = 'sell'
            agent.update_last_price(asset, prices[asset])
            amount = {'SPK.NZ': 1, 'BTC': 0.001, 'ETH': 0.01, 'XRP': 10, 'NZDUSD=X': 1000}[asset]
            actions[asset] = {'action': action, 'amount': amount}
        
        chosen_asset = np.random.choice(list(actions.keys()))
        return jsonify({'action': actions[chosen_asset]['action'], 'asset': chosen_asset, 'amount': actions[chosen_asset]['amount']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)