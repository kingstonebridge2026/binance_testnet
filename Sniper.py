import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import ccxt.async_support as ccxt
import json
import websockets
import logging
import ta
import aiohttp
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "0hb4IO19WSbyO6VlM8S0Aa8tWwHSYhtQhDRoOG70iu912J95qm7HhtRspAoykSml"
    BINANCE_SECRET = "RE8tftdsuG4MzcMfR4VNy6yvkho27qDMGiLZ6yR4cHXRWmCq1sV5AfBmgIIH06dK"
    TELEGRAM_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
    TELEGRAM_CHAT_ID = "5665906172"
    
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
        "MATIC/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",
        "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "SUI/USDT"
    ]
    
    BASE_POSITION_USD = 50.0  
    MAX_SLOTS = 15            
    SEQUENCE_LENGTH = 60 
    INPUT_SIZE = 7 
    TARGET_PROFIT_NET = 0.006 
    STOP_LOSS_PCT = 0.012    
    AI_THRESHOLD = 0.52      # Adjusted slightly for initial testing

# ==================== AI MODEL ====================
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size*2, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_size*2, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(torch.relu(self.input_layer(x)))
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.gate(attn_out[:, -1, :])

# ==================== TRADING CORE ====================
class DeepAlphaBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        # Set to False if using Live Keys, True if using Testnet Keys
        self.exchange.set_sandbox_mode(True) 
        
        self.model = TemporalFusionTransformer(Config.INPUT_SIZE)
        self.scaler = StandardScaler()
        self.positions = []
        self.banked_pnl = 0.0
        self.is_running = True
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("AlphaBot")

    def engineer_features(self, df):
        df = df.copy()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
        df['ema_25'] = ta.trend.ema_indicator(df['close'], window=25)
        df['bb_w'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        return df[['close', 'volume', 'rsi', 'ema_7', 'ema_25', 'bb_w', 'zscore']].dropna()

    async def train_on_fly(self):
        """Memory-optimized training to prevent Railway OOM crashes"""
        self.logger.info("📥 Starting LITE Warm-up (Memory-Safe Mode)...")
        all_data = []
        
        # Reduced to only 2 symbols for the 'warm-up' to save RAM
        for symbol in Config.SYMBOLS[:2]: 
            try:
                # Reduced history from 1000 to 300 to keep RAM low
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=300) 
                df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                features = self.engineer_features(df)
                all_data.append(features.values)
            except Exception as e:
                self.logger.error(f"Data fetch error: {e}")
        
        if not all_data:
            self.logger.warning("No data found for training. Skipping warm-up.")
            return

        train_data = np.vstack(all_data)
        self.scaler.fit(train_data)
        
        # Prepare small batches to keep memory usage flat
        scaled_data = self.scaler.transform(train_data)
        X, Y = [], []
        for i in range(len(scaled_data) - Config.SEQUENCE_LENGTH - 1):
            X.append(scaled_data[i : i + Config.SEQUENCE_LENGTH])
            Y.append(1 if train_data[i + Config.SEQUENCE_LENGTH + 1, 0] > train_data[i + Config.SEQUENCE_LENGTH, 0] else 0)
        
        X = torch.FloatTensor(np.array(X))
        Y = torch.LongTensor(np.array(Y))
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.logger.info(f"🧠 Training on {len(X)} samples...")
        self.model.train()
        for epoch in range(5): # Reduced epochs for speed/stability
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            # Clear cache to free RAM
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        self.model.eval()
        self.logger.info("✅ LITE Warm-up complete. RAM usage stabilized.")


    async def send_telegram(self, message):
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Telegram Error: {e}")

    async def manage_risk_multi(self, symbol, current_price):
        for pos in self.positions[:]:
            if pos['symbol'] == symbol:
                pnl_pct = (current_price - pos['entry']) / pos['entry']
                if pnl_pct > Config.TARGET_PROFIT_NET or pnl_pct < -Config.STOP_LOSS_PCT:
                    side = "TP" if pnl_pct > 0 else "SL"
                    try:
                        await self.exchange.create_market_sell_order(symbol, pos['amt'])
                        profit = (current_price - pos['entry']) * pos['amt']
                        self.banked_pnl += profit
                        self.positions.remove(pos)
                        emoji = "✅" if side == "TP" else "⚠️"
                        await self.send_telegram(f"{emoji} <b>{side} HIT: {symbol}</b>\nPnL: ${profit:.2f}")
                    except Exception as e:
                        self.logger.error(f"Exit Error {symbol}: {e}")

    async def inference_loop(self):
        await self.train_on_fly()
        await self.send_telegram("🤖 <b>AI Bot Online</b>\nSignals active for 20 pairs.")
        
        while self.is_running:
            for symbol in Config.SYMBOLS:
                try:
                    if len(self.positions) >= Config.MAX_SLOTS: break
                    if any(p['symbol'] == symbol for p in self.positions): continue

                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    features_df = self.engineer_features(df)
                    
                    # Use transform, NOT fit_transform here
                    scaled = self.scaler.transform(features_df.values)
                    inp = torch.FloatTensor(scaled[-Config.SEQUENCE_LENGTH:]).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = self.model(inp)
                        probs = torch.softmax(logits, dim=1)
                        prediction = probs[0][1].item() # Probability of 'Up'

                    print(f"DEBUG: {symbol} Score: {prediction:.4f}")

                    if prediction > Config.AI_THRESHOLD:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        await self.exchange.create_market_buy_order(symbol, amt)
                        self.positions.append({'symbol': symbol, 'entry': price, 'amt': amt})
                        await self.send_telegram(f"🚀 <b>AI BUY: {symbol}</b>\nConf: {prediction:.2f}")

                    await asyncio.sleep(0.5)
                except Exception as e:
                    self.logger.error(f"Scanner Error ({symbol}): {e}")
            await asyncio.sleep(10)

# ==================== WEBSOCKET REFLEX ====================
async def binance_multi_stream(bot):
    streams = [f"{s.replace('/', '').lower()}@ticker" for s in Config.SYMBOLS]
    url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
    while bot.is_running:
        try:
            async with websockets.connect(url) as ws:
                while bot.is_running:
                    data = json.loads(await ws.recv())
                    symbol_raw = data['s']
                    for original_symbol in Config.SYMBOLS:
                        if original_symbol.replace('/', '') == symbol_raw:
                            await bot.manage_risk_multi(original_symbol, float(data['c']))
                            break
        except: await asyncio.sleep(5)

async def main():
    bot = DeepAlphaBot()
    await asyncio.gather(bot.inference_loop(), binance_multi_stream(bot))

if __name__ == "__main__":
    asyncio.run(main())
