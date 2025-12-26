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
# ==================== UPDATED CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "0hb4IO19WSbyO6VlM8S0Aa8tWwHSYhtQhDRoOG70iu912J95qm7HhtRspAoykSml"
    BINANCE_SECRET = "RE8tftdsuG4MzcMfR4VNy6yvkho27qDMGiLZ6yR4cHXRWmCq1sV5AfBmgIIH06dK"
    TELEGRAM_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
    TELEGRAM_CHAT_ID = "5665906172"
    
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
        "POL/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",  # Updated POL
        "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "SUI/USDT"
    ]
    
    BASE_POSITION_USD = 50.0  
    MAX_SLOTS = 15            
    
    # Aggressive Strategy Settings
    RSI_BUY_LEVEL = 35       # Increased from 30 for more action
    Z_SCORE_BUY = -1.5       # Increased from -2.0 for more action
    TARGET_PROFIT = 0.008    # 0.8% (Slightly higher to cover fees)
    STOP_LOSS = 0.015        # 1.5%
        # 1.2%

# ==================== TRADING CORE ====================
import time

class AlphaSniper:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True) 
        self.positions = []
        self.banked_pnl = 0.0  # Tracks realized profit
        self.is_running = True
        self.logger = logging.getLogger("Sniper")

    # ... [Keep your send_telegram and engineer_features functions] ...

    async def daily_reporter(self):
        """Sends a summary report every 24 hours"""
        while self.is_running:
            # Wait 24 hours (86400 seconds)
            await asyncio.sleep(86400)
            status = "✅ WINNING" if self.banked_pnl > 0 else "❌ DOWN"
            report = (
                f"📅 <b>Daily Sniper Report</b>\n"
                f"Status: {status}\n"
                f"Realized PnL: ${self.banked_pnl:.2f}\n"
                f"Open Slots: {len(self.positions)}/{Config.MAX_SLOTS}\n"
                f"Mode: BINANCE TESTNET"
            )
            await self.send_telegram(report)

    async def monitor_markets(self):
        await self.send_telegram("🚀 <b>Sniper Online (Aggressive Mode)</b>\nRSI: 35 | Z: -1.5")
        # Start the reporting task in the background
        asyncio.create_task(self.daily_reporter())
        
        while self.is_running:
            for symbol in Config.SYMBOLS:
                try:
                    # Logic to skip if already in position or slots full
                    if any(p['symbol'] == symbol for p in self.positions): continue
                    
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    sig = self.get_signals(df)
                    
                    # Handle "NaN" data safely
                    if np.isnan(sig['rsi']) or np.isnan(sig['zscore']):
                        continue

                    # Aggressive Buy Signal
                    if sig['rsi'] < Config.RSI_BUY_LEVEL and sig['zscore'] < Config.Z_SCORE_BUY:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        
                        await self.exchange.create_market_buy_order(symbol, amt)
                        self.positions.append({'symbol': symbol, 'entry': price, 'amt': amt})
                        await self.send_telegram(f"🎯 <b>BUY: {symbol}</b>\nPrice: {price}\nRSI: {sig['rsi']:.1f}")

                except Exception as e:
                    self.logger.error(f"Error {symbol}: {e}")
            await asyncio.sleep(15)

    async def manage_risk(self, symbol, price):
        for pos in self.positions[:]:
            if pos['symbol'] == symbol:
                pnl_pct = (price - pos['entry']) / pos['entry']
                
                # Check Take Profit or Stop Loss
                if pnl_pct > Config.TARGET_PROFIT or pnl_pct < -Config.STOP_LOSS:
                    await self.exchange.create_market_sell_order(symbol, pos['amt'])
                    profit_usd = (price - pos['entry']) * pos['amt']
                    self.banked_pnl += profit_usd
                    self.positions.remove(pos)
                    
                    emoji = "💰" if pnl_pct > 0 else "📉"
                    await self.send_telegram(f"{emoji} <b>CLOSED {symbol}</b>\nProfit: ${profit_usd:.2f} ({pnl_pct*100:.2f}%)")


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
