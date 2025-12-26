import numpy as np
import pandas as pd
import asyncio
import ccxt.async_support as ccxt
import logging
import ta
import aiohttp
import time

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "r6hhHQubpwwnDYkYhhdSlk3MQPjTomUggf59gfXJ21hnBcfq3K4BIoSd1eE91V3N"
    BINANCE_SECRET = "B7ioAXzVHyYlxPOz3AtxzMC6FQBZaRj6i8A9FenSbsK8rBeCdGZHDhX6Dti22F2x"
    TELEGRAM_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
    TELEGRAM_CHAT_ID = "5665906172"
    
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
        "POL/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",
        "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "SUI/USDT"
    ]
    
    BASE_POSITION_USD = 50
    MAX_SLOTS = 15            
    RSI_BUY_LEVEL = 35       
    Z_SCORE_BUY = -1.5       
    TARGET_PROFIT = 0.008    
    STOP_LOSS = 0.015        

# ==================== TRADING CORE ====================
class AlphaSniper:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'fetchCurrencies': False # Bypasses SAPI errors
            }
        })
        
        # Point to the stable 2025 Demo API
        self.exchange.urls['api'] = {
            'public': 'https://demo-api.binance.com/api',
            'private': 'https://demo-api.binance.com/api',
        }
        
        self.positions = []
        self.banked_pnl = 0.0
        self.is_running = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("Sniper")

    async def send_telegram(self, message):
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={"chat_id": Config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"})
        except Exception as e:
            self.logger.error(f"Telegram Error: {e}")

    # --- THE AI SIGNAL BRAIN ---
    def get_ai_signals(self, df):
        """Mathematical AI Model: RSI + Volatility Z-Score"""
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            rolling_mean = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            df['zscore'] = (df['close'] - rolling_mean) / rolling_std
            
            # AI Logic: Trend Confirmation
            df['sma_fast'] = ta.trend.sma_indicator(df['close'], window=7)
            df['sma_slow'] = ta.trend.sma_indicator(df['close'], window=25)
            
            return df.iloc[-1]
        except Exception as e:
            self.logger.error(f"AI Signal error: {e}")
            return None

    async def monitor_markets(self):
        await self.send_telegram("🚀 <b>AI Sniper Online</b>\nDemo Mode: ACTIVE")
        
        while self.is_running:
            for symbol in Config.SYMBOLS:
                try:
                    if len(self.positions) >= Config.MAX_SLOTS: break
                    if any(p['symbol'] == symbol for p in self.positions): continue

                    # Fetch Data
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Get AI Signals
                    sig = self.get_ai_signals(df)
                    if sig is None or np.isnan(sig['rsi']) or np.isnan(sig['zscore']):
                        continue

                    # EXECUTION LOGIC
                    if sig['rsi'] < Config.RSI_BUY_LEVEL and sig['zscore'] < Config.Z_SCORE_BUY:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        
                        # Market Order
                        await self.exchange.create_market_buy_order(symbol, amt)
                        
                        self.positions.append({
                            'symbol': symbol,
                            'entry': price,
                            'amt': amt,
                            'timestamp': time.time()
                        })
                        
                        await self.send_telegram(f"🎯 <b>AI BUY: {symbol}</b>\nPrice: ${price:.4f}\nRSI: {sig['rsi']:.1f}")

                    await asyncio.sleep(0.5) # Anti-ban delay
                except Exception as e:
                    self.logger.error(f"Error {symbol}: {e}")
            
            await asyncio.sleep(15)

    async def manage_positions(self):
        """Risk Management AI: Dynamic TP/SL"""
        while self.is_running:
            for pos in self.positions[:]:
                try:
                    ticker = await self.exchange.fetch_ticker(pos['symbol'])
                    curr_price = ticker['last']
                    pnl = (curr_price - pos['entry']) / pos['entry']

                    if pnl > Config.TARGET_PROFIT or pnl < -Config.STOP_LOSS:
                        await self.exchange.create_market_sell_order(pos['symbol'], pos['amt'])
                        
                        profit_usd = (curr_price - pos['entry']) * pos['amt']
                        self.banked_pnl += profit_usd
                        self.positions.remove(pos)
                        
                        emoji = "💰" if profit_usd > 0 else "📉"
                        await self.send_telegram(f"{emoji} <b>CLOSED {pos['symbol']}</b>\nProfit: ${profit_usd:.2f}")
                except Exception as e:
                    self.logger.error(f"Exit Error: {e}")
            await asyncio.sleep(5)

async def main():
    bot = AlphaSniper()
    # Runs the scanner and the risk manager at the same time
    await asyncio.gather(bot.monitor_markets(), bot.manage_positions())

if __name__ == "__main__":
    asyncio.run(main())
