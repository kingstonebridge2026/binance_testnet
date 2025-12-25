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
import aiohttp
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "0hb4IO19WSbyO6VlM8S0Aa8tWwHSYhtQhDRoOG70iu912J95qm7HhtRspAoykSml"
    BINANCE_SECRET = "RE8tftdsuG4MzcMfR4VNy6yvkho27qDMGiLZ6yR4cHXRWmCq1sV5AfBmgIIH06dK"
    TELEGRAM_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
    TELEGRAM_CHAT_ID = "5665906172"
    
    # Expanded to 20 High-Volume Pairs
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
    
    TARGET_PROFIT_NET = 0.006 # 0.6%
    STOP_LOSS_PCT = 0.012    # 1.2%
    AI_THRESHOLD = 0.55      # Slightly more aggressive for multi-coin

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
        self.exchange.set_sandbox_mode(True) 
        
        self.model = TemporalFusionTransformer(Config.INPUT_SIZE)
        self.scaler = StandardScaler()
        self.positions = []
        self.banked_pnl = 0.0
        self.is_running = True
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("AlphaBot")

    async def send_telegram(self, message):
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

    async def manage_risk_multi(self, symbol, current_price):
        """Modified to handle specific symbol risks"""
        for pos in self.positions[:]:
            if pos['symbol'] == symbol:
                pnl_pct = (current_price - pos['entry']) / pos['entry']
                
                if pnl_pct > Config.TARGET_PROFIT_NET:
                    await self.exchange.create_market_sell_order(symbol, pos['amt'])
                    profit = (current_price - pos['entry']) * pos['amt']
                    self.banked_pnl += profit
                    self.positions.remove(pos)
                    await self.send_telegram(f"✅ <b>TP HIT: {symbol}</b>\nProfit: +${profit:.2f}\nTotal: ${self.banked_pnl:.2f}")
                
                elif pnl_pct < -Config.STOP_LOSS_PCT:
                    await self.exchange.create_market_sell_order(symbol, pos['amt'])
                    loss = (current_price - pos['entry']) * pos['amt']
                    self.banked_pnl += loss
                    self.positions.remove(pos)
                    await self.send_telegram(f"⚠️ <b>SL HIT: {symbol}</b>\nLoss: ${loss:.2f}")

    async def inference_loop(self):
        await self.send_telegram(f"🤖 <b>Multi-Scanner Active</b>\nHunting 20 pairs on Railway.")
        while self.is_running:
            for symbol in Config.SYMBOLS:
                try:
                    if len(self.positions) >= Config.MAX_SLOTS:
                        break

                    # Skip if we already hold this coin
                    if any(p['symbol'] == symbol for p in self.positions):
                        continue

                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    features_df = self.engineer_features(df)
                    
                    scaled = self.scaler.fit_transform(features_df.values)
                    inp = torch.FloatTensor(scaled[-Config.SEQUENCE_LENGTH:]).unsqueeze(0)
                    
                    with torch.no_grad():
                        prediction, _ = self.model(inp)[0].tolist()

                    if prediction > Config.AI_THRESHOLD:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        await self.exchange.create_market_buy_order(symbol, amt)
                        self.positions.append({'symbol': symbol, 'entry': price, 'amt': amt})
                        await self.send_telegram(f"🚀 <b>AI ENTRY: {symbol}</b>\nPrice: {price}\nConf: {prediction:.2f}")

                    await asyncio.sleep(1) # Rate limit protection
                except Exception as e:
                    self.logger.error(f"Scanner Error ({symbol}): {e}")
            
            await asyncio.sleep(10) # Wait before next full scan

# ==================== WEBSOCKET REFLEX ====================
async def binance_multi_stream(bot):
    """Monitors prices for all 20 coins via a single connection"""
    streams = [f"{s.replace('/', '').lower()}@ticker" for s in Config.SYMBOLS]
    url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
    
    while bot.is_running:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                while bot.is_running:
                    data = json.loads(await ws.recv())
                    symbol_raw = data['s'] # e.g. BTCUSDT
                    # Convert raw back to our symbol format
                    for original_symbol in Config.SYMBOLS:
                        if original_symbol.replace('/', '') == symbol_raw:
                            current_price = float(data['c'])
                            await bot.manage_risk_multi(original_symbol, current_price)
                            break
        except Exception:
            await asyncio.sleep(5)

async def main():
    bot = DeepAlphaBot()
    await asyncio.gather(
        bot.inference_loop(),
        binance_multi_stream(bot)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

