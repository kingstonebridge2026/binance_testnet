import numpy as np
import pandas as pd
import asyncio
import ccxt.async_support as ccxt
import json
import websockets
import logging
import ta
import aiohttp
from datetime import datetime

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
    
    # Strategy settings
    RSI_BUY_LEVEL = 30       # Buy when oversold
    TARGET_PROFIT = 0.006    # 0.6%
    STOP_LOSS = 0.012        # 1.2%

# ==================== TRADING CORE ====================
class AlphaSniper:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True) # KEEP TRUE FOR TESTNET
        self.positions = []
        self.is_running = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("Sniper")

    async def send_telegram(self, message):
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"})

    def get_signals(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        return df.iloc[-1]

    async def monitor_markets(self):
        await self.send_telegram("🚀 <b>Sniper Online</b>\nStable Mode: No-RAM Training Active.")
        while self.is_running:
            for symbol in Config.SYMBOLS:
                try:
                    if len(self.positions) >= Config.MAX_SLOTS: break
                    if any(p['symbol'] == symbol for p in self.positions): continue

                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    sig = self.get_signals(df)
                    
                    self.logger.info(f"Checking {symbol} | RSI: {sig['rsi']:.2f} | Z: {sig['zscore']:.2f}")

                    # Logic: Mean Reversion (Buy low Z-score + low RSI)
                    if sig['rsi'] < Config.RSI_BUY_LEVEL and sig['zscore'] < -2.0:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        
                        await self.exchange.create_market_buy_order(symbol, amt)
                        self.positions.append({'symbol': symbol, 'entry': price, 'amt': amt})
                        await self.send_telegram(f"🎯 <b>BUY: {symbol}</b>\nPrice: {price}\nRSI: {sig['rsi']:.1f}")

                    await asyncio.sleep(0.5)
                except Exception as e:
                    self.logger.error(f"Error {symbol}: {e}")
            await asyncio.sleep(15)

    async def manage_risk(self, symbol, price):
        for pos in self.positions[:]:
            if pos['symbol'] == symbol:
                pnl = (price - pos['entry']) / pos['entry']
                if pnl > Config.TARGET_PROFIT or pnl < -Config.STOP_LOSS:
                    await self.exchange.create_market_sell_order(symbol, pos['amt'])
                    self.positions.remove(pos)
                    status = "PROFIT" if pnl > 0 else "LOSS"
                    await self.send_telegram(f"💰 <b>SELL {symbol}</b>\nResult: {status} ({pnl*100:.2f}%)")

async def price_stream(bot):
    streams = [f"{s.replace('/', '').lower()}@ticker" for s in Config.SYMBOLS]
    url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
    async with websockets.connect(url) as ws:
        while bot.is_running:
            data = json.loads(await ws.recv())
            await bot.manage_risk(Config.SYMBOLS[0], float(data['c'])) # Simplified for example

async def main():
    bot = AlphaSniper()
    await asyncio.gather(bot.monitor_markets(), price_stream(bot))

if __name__ == "__main__":
    asyncio.run(main())
