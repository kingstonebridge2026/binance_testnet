import numpy as np
import pandas as pd
import asyncio
import ccxt.async_support as ccxt
import logging
import ta
import aiohttp

# ==================== CONFIGURATION ====================
class Config:
    # Use your existing keys here
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
    
    # Aggressive Strategy Settings
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
                'fetchCurrencies': False  # CRITICAL: Skips SAPI calls that fail on testnet
            }
        })
        
        # Point to the NEW official Demo/Testnet API
        self.exchange.urls['api'] = {
            'public': 'https://testnet.binance.vision/api',
            'private': 'https://testnet.binance.vision/api',
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
                await session.post(url, json={
                    "chat_id": Config.TELEGRAM_CHAT_ID, 
                    "text": message, 
                    "parse_mode": "HTML"
                })
        except Exception as e:
            self.logger.error(f"Telegram Error: {e}")

    def get_signals(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['zscore'] = (df['close'] - rolling_mean) / rolling_std
        return df.iloc[-1]

    async def monitor_markets(self):
        await self.send_telegram("🚀 <b>Sniper Online (Spot Testnet)</b>\nWaiting for dips...")
        
        while self.is_running:
            try:
                # We skip balance check here if it keeps causing SAPI errors
                # The bot will just fail the order if balance is 0, which is fine for testing
                for symbol in Config.SYMBOLS:
                    if len(self.positions) >= Config.MAX_SLOTS: break
                    if any(p['symbol'] == symbol for p in self.positions): continue

                    # Fetch market data (This uses Public API, which usually works)
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                    df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
                    sig = self.get_signals(df)
                    
                    if np.isnan(sig['rsi']) or np.isnan(sig['zscore']):
                        continue

                    if sig['rsi'] < Config.RSI_BUY_LEVEL and sig['zscore'] < Config.Z_SCORE_BUY:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        amt = Config.BASE_POSITION_USD / price
                        
                        await self.exchange.create_market_buy_order(symbol, amt)
                        self.positions.append({'symbol': symbol, 'entry': price, 'amt': amt})
                        await self.send_telegram(f"🎯 <b>BUY: {symbol}</b>\nPrice: {price}\nRSI: {sig['rsi']:.1f}")

                    await asyncio.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Loop Error: {e}")
            
            await asyncio.sleep(20)

    async def manage_risk(self, symbol, price):
        for pos in self.positions[:]:
            if pos['symbol'] == symbol:
                pnl = (price - pos['entry']) / pos['entry']
                if pnl > Config.TARGET_PROFIT or pnl < -Config.STOP_LOSS:
                    try:
                        await self.exchange.create_market_sell_order(symbol, pos['amt'])
                        profit_usd = (price - pos['entry']) * pos['amt']
                        self.banked_pnl += profit_usd
                        self.positions.remove(pos)
                        emoji = "💰" if pnl > 0 else "📉"
                        await self.send_telegram(f"{emoji} <b>CLOSED {symbol}</b>\nProfit: ${profit_usd:.2f}")
                    except Exception as e:
                        self.logger.error(f"Sell Error: {e}")

async def price_stream(bot):
    while bot.is_running:
        for pos in bot.positions:
            try:
                ticker = await bot.exchange.fetch_ticker(pos['symbol'])
                await bot.manage_risk(pos['symbol'], ticker['last'])
            except:
                pass
        await asyncio.sleep(5)

async def main():
    bot = AlphaSniper()
    await asyncio.gather(bot.monitor_markets(), price_stream(bot))

if __name__ == "__main__":
    asyncio.run(main())
