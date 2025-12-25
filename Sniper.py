
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
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7"
    BINANCE_SECRET = "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH"
    
    SYMBOL = "BTC/USDT"
    BASE_POSITION_USD = 100.0  
    MAX_SLOTS = 10
    
    SEQUENCE_LENGTH = 60 
    INPUT_SIZE = 7 # Optimized for speed on Redmi 9A
    
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
        self.is_running = True
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("AlphaBot")

    def engineer_features(self, df):
        # Optimized feature set for real-time performance
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
        df['ema_25'] = ta.trend.ema_indicator(df['close'], window=25)
        df['bb_w'] = ta.volatility.BollingerBands(df['close']).bollinger_wband()
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        return df[['close', 'volume', 'rsi', 'ema_7', 'ema_25', 'bb_w', 'zscore']].dropna()

    async def manage_risk(self, current_price):
        """Called by WebSocket for instant reactions"""
        for pos in self.positions[:]:
            pnl_pct = (current_price - pos['entry']) / pos['entry']
            
            # Instant Take-Profit
            if pnl_pct > (Config.TARGET_PROFIT_NET + 0.002):
                await self.exchange.create_market_sell_order(Config.SYMBOL, pos['amt'])
                self.positions.remove(pos)
                self.logger.info(f"💰 PROFIT BANKED: {pnl_pct:.2%}")
            
            # Instant Stop-Loss
            elif pnl_pct < -Config.STOP_LOSS_PCT:
                await self.exchange.create_market_sell_order(Config.SYMBOL, pos['amt'])
                self.positions.remove(pos)
                self.logger.warning(f"⚠️ STOP LOSS HIT: {pnl_pct:.2%}")

    async def inference_loop(self):
        """The 'Brain' - Runs deep analysis every few seconds"""
        while self.is_running:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(Config.SYMBOL, '1m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                features_df = self.engineer_features(df)
                
                # Prepare AI Input
                scaled = self.scaler.fit_transform(features_df.values)
                inp = torch.FloatTensor(scaled[-Config.SEQUENCE_LENGTH:]).unsqueeze(0)
                
                with torch.no_grad():
                    prediction, volatility = self.model(inp)[0].tolist()

                # Execution Logic
                if prediction > 0.6 and len(self.positions) < Config.MAX_SLOTS:
                    price = self.last_price if self.last_price > 0 else df['close'].iloc[-1]
                    amt = Config.BASE_POSITION_USD / price
                    await self.exchange.create_market_buy_order(Config.SYMBOL, amt)
                    self.positions.append({'entry': price, 'amt': amt})
                    self.logger.info(f"🚀 AI BUY | Conf: {prediction:.2f} | Price: {price}")

                await asyncio.sleep(5) 
            except Exception as e:
                self.logger.error(f"Inference Error: {e}")
                await asyncio.sleep(10)

# ==================== WEBSOCKET REFLEX ====================
# ==================== STABLE WEBSOCKET REFLEX ====================
async def binance_stream(symbol, bot):
    stream_symbol = symbol.replace('/', '').lower()
    url = f"wss://stream.binance.com:9443/ws/{stream_symbol}@ticker"
    
    bot.logger.info(f"📡 Attempting connection to {symbol} stream...")
    
    while bot.is_running:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                bot.logger.info("✅ WebSocket Connected & Monitoring Ticks")
                while bot.is_running:
                    try:
                        data = json.loads(await ws.recv())
                        current_price = float(data['c'])
                        bot.last_price = current_price
                        
                        if bot.positions:
                            await bot.manage_risk(current_price)
                    except websockets.ConnectionClosed:
                        bot.logger.warning("📡 Connection lost. Reconnecting...")
                        break # Exit inner loop to trigger reconnect
        except Exception as e:
            bot.logger.error(f"📡 Stream Error: {e}. Retrying in 5s...")
            await asyncio.sleep(5)


# ==================== MAIN START ====================
async def main():
    bot = DeepAlphaBot()
    # Run the Brain and the Reflex simultaneously
    await asyncio.gather(
        bot.inference_loop(),
        binance_stream(Config.SYMBOL, bot)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
