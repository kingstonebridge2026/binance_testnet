import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import asyncio
import ccxt.async_support as ccxt
import json
import websockets
import logging
import ta
import aiohttp # Added for Telegram
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7"
    BINANCE_SECRET = "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH"
    
    # --- TELEGRAM CONFIG ---
    TELEGRAM_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
    TELEGRAM_CHAT_ID = "5665906172"
    
    SYMBOL = "BTC/USDT"
    BASE_POSITION_USD = 100.0  
    MAX_SLOTS = 10
    
    SEQUENCE_LENGTH = 60 
    INPUT_SIZE = 7 
    
    TARGET_PROFIT_NET = 0.005 # 0.5% Net
    STOP_LOSS_PCT = 0.015    # 1.5% SL

# ==================== AI MODEL (TRANSFORMER) ====================
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

# ==================== THE TRADING ENGINE ====================
class DeepAlphaBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True) 
        
        self.model = TemporalFusionTransformer(Config.INPUT_SIZE)
        self.scaler = StandardScaler()
        self.positions = []
        self.last_price = 0.0
        self.banked_pnl = 0.0
        self.is_running = True
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("AlphaBot")

    async def send_telegram(self, message):
        """Sends real-time alerts to your phone"""
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Telegram Error: {e}")

    def engineer_features(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
        df['ema_25'] = ta.trend.ema_indicator(df['close'], window=25)
        df['bb_w'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        return df[['close', 'volume', 'rsi', 'ema_7', 'ema_25', 'bb_w', 'zscore']].dropna()

    async def manage_risk(self, current_price):
        for pos in self.positions[:]:
            pnl_pct = (current_price - pos['entry']) / pos['entry']
            
            if pnl_pct > (Config.TARGET_PROFIT_NET + 0.002):
                await self.exchange.create_market_sell_order(Config.SYMBOL, pos['amt'])
                profit = (current_price - pos['entry']) * pos['amt']
                self.banked_pnl += profit
                self.positions.remove(pos)
                msg = f"✅ <b>TP HIT!</b>\nProfit: +${profit:.2f}\nPNL: {pnl_pct:.2%}\nTotal Banked: ${self.banked_pnl:.2f}"
                await self.send_telegram(msg)
            
            elif pnl_pct < -Config.STOP_LOSS_PCT:
                await self.exchange.create_market_sell_order(Config.SYMBOL, pos['amt'])
                loss = (current_price - pos['entry']) * pos['amt']
                self.banked_pnl += loss
                self.positions.remove(pos)
                msg = f"⚠️ <b>SL HIT!</b>\nLoss: ${loss:.2f}\nPNL: {pnl_pct:.2%}"
                await self.send_telegram(msg)

    async def inference_loop(self):
        await self.send_telegram("🤖 <b>Transformer Bot Active</b>\nSystem initialized on Railway.")
        while self.is_running:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(Config.SYMBOL, '1m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                features_df = self.engineer_features(df)
                
                scaled = self.scaler.fit_transform(features_df.values)
                inp = torch.FloatTensor(scaled[-Config.SEQUENCE_LENGTH:]).unsqueeze(0)
                
                with torch.no_grad():
                    prediction, volatility = self.model(inp)[0].tolist()

                if prediction > 0.6 and len(self.positions) < Config.MAX_SLOTS:
                    price = self.last_price if self.last_price > 0 else df['close'].iloc[-1]
                    amt = Config.BASE_POSITION_USD / price
                    await self.exchange.create_market_buy_order(Config.SYMBOL, amt)
                    self.positions.append({'entry': price, 'amt': amt})
                    await self.send_telegram(f"🚀 <b>AI LONG ENTRY</b>\nPrice: {price}\nConfidence: {prediction:.2f}")

                await asyncio.sleep(10) 
            except Exception as e:
                self.logger.error(f"Inference Error: {e}")
                await asyncio.sleep(10)

# ==================== WEBSOCKET REFLEX ====================
async def binance_stream(symbol, bot):
    stream_symbol = symbol.replace('/', '').lower()
    url = f"wss://stream.binance.com:9443/ws/{stream_symbol}@ticker"
    
    while bot.is_running:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                while bot.is_running:
                    try:
                        data = json.loads(await ws.recv())
                        current_price = float(data['c'])
                        bot.last_price = current_price
                        if bot.positions:
                            await bot.manage_risk(current_price)
                    except websockets.ConnectionClosed:
                        break 
        except Exception as e:
            await asyncio.sleep(5)

async def main():
    bot = DeepAlphaBot()
    await asyncio.gather(bot.inference_loop(), binance_stream(Config.SYMBOL, bot))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
