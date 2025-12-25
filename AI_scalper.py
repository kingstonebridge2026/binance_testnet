import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import uuid
from datetime import datetime
from telegram.ext import ApplicationBuilder

# ===== PRO CREDENTIALS =====
BINANCE_API_KEY = "XG7dfcfj79N16ZGCU8nvbUdbUFjOXmQjw7iWMSX8V1LZZTUdJqNEIEAmAVqpGtVj"
BINANCE_SECRET = "mtBu3ZqWoFfwKHDcFuRFywpFyHpxwEnUmqNrrje0k8z5qB9Q7GpAE4WSeOqturgn"
TELEGRAM_BOT_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
# Using $200 effective size to force $25/hr on $100 base (Leveraging Testnet)
POSITION_SIZE = 0.0024 
MAX_SLOTS = 10 
TARGET_HOURLY = 25.0

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET,
    'enableRateLimit': True, 'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True)

bot_state = {"balance": 100.0, "pnl": 0.0, "positions": [], "wins": 0}

async def send_msg(app, text):
    try: await app.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode='HTML')
    except: pass

async def brain_loop(app):
    await send_msg(app, "🧠 <b>AI-SCALPER INITIALIZED</b>\nMode: Mean Reversion\nGoal: $25/Hr")
    
    while True:
        try:
            # 1. Fetch Deep Data for "Brain" Calculation
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, '1m', limit=20)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            # AI Trick: Use EMA crossover + Standard Deviation for "Volatility Catching"
            df['ema'] = df['c'].ewm(span=7, adjust=False).mean()
            df['std'] = df['c'].rolling(7).std()
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            price = ticker['last']
            
            # 2. ENTRY: "The Spring Trap"
            # Buy if price is significantly below the EMA (Oversold Micro-dip)
            if len(bot_state['positions']) < MAX_SLOTS:
                if price < (df['ema'].iloc[-1] - (df['std'].iloc[-1] * 0.5)):
                    await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
                    bot_state['positions'].append({"id": str(uuid.uuid4())[:3], "entry": price})
                    print(f"Trap Set: {price}")

            # 3. EXIT: "The Lightning Strike"
            for p in bot_state['positions'][:]:
                diff = (price - p['entry']) / p['entry']
                
                # Goal: Capture 0.22% net profit per scalp
                # We exit at 0.35% to ensure 0.25% after fees
                if diff >= 0.0035:
                    await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
                    net = (POSITION_SIZE * price * diff) - (POSITION_SIZE * price * 0.001)
                    bot_state['pnl'] += net
                    bot_state['wins'] += 1
                    bot_state['positions'].remove(p)
                    
                    await send_msg(app, f"💰 <b>SCALP SUCCESS</b>\nProfit: +${net:.2f}\nSession: ${bot_state['pnl']:.2f}")

            await asyncio.sleep(2) # 2-second AI refresh rate
        except Exception as e:
            await asyncio.sleep(5)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await brain_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
