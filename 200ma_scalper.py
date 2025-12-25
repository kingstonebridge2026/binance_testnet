import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION (Keys Pre-Loaded) =====
BINANCE_API_KEY = "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7"
BINANCE_SECRET = "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH"
TELEGRAM_BOT_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
# Increased Position Size to ~0.0018 BTC (~$150) to make $0.50/hr achievable via scalping
POSITION_SIZE = 0.0018 
MAX_TRADES = 4           # Keep 4 slots open to catch different price dips
MIN_PROFIT_TARGET = 0.006 # 0.6% target (Covers 0.2% fees + leaves 0.4% profit)
STOP_LOSS = -0.015       # 1.5% Stop Loss for safety

# Initialize Exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True) # KEEP TRUE FOR DEMO/TESTNET

bot_state = {
    "virtual_balance": 100.0,
    "realized_pnl": 0.0,
    "trades_count": 0,
    "last_price": 0.0,
    "positions": [],
    "start_time": datetime.now(),
    "hourly_pnl": 0.0
}

# 

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

# ===== TRADING EXECUTION =====
async def open_position(price, application):
    try:
        await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
        new_pos = {"id": str(uuid.uuid4())[:5], "entry_price": price, "time": datetime.now()}
        bot_state['positions'].append(new_pos)
        msg = f"🟢 <b>TRADE OPENED</b>\nEntry: ${price:,.2f}\nSlots: {len(bot_state['positions'])}/{MAX_TRADES}"
        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Open Error: {e}")

async def close_position(pos, price, application):
    try:
        await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        raw_pnl = (price - pos['entry_price']) * POSITION_SIZE
        fee = (POSITION_SIZE * price) * 0.002 # 0.2% Round trip
        net = raw_pnl - fee
        
        bot_state['virtual_balance'] += net
        bot_state['realized_pnl'] += net
        bot_state['hourly_pnl'] += net
        bot_state['trades_count'] += 1
        bot_state['positions'].remove(pos)
        
        icon = "💰" if net > 0 else "⚠️"
        msg = f"{icon} <b>PROFIT REALIZED</b>\nNet: ${net:.4f}\nTotal P&L: ${bot_state['realized_pnl']:.2f}"
        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Close Error: {e}")

# ===== MAIN LOOP =====
async def trading_loop(application):
    print("🚀 TARGET: $0.50/HOUR SNIPER ACTIVE...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            df['rsi'] = calculate_rsi(df['close'], 14)
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            rsi = df['rsi'].iloc[-1]

            # 1. ENTRY LOGIC: Buy only on deep RSI dips (< 32)
            if len(bot_state['positions']) < MAX_TRADES and rsi < 32:
                await open_position(bot_state['last_price'], application)

            # 2. EXIT LOGIC: Check each position for the 0.6% target
            for pos in bot_state['positions'][:]:
                change = (bot_state['last_price'] - pos['entry_price']) / pos['entry_price']
                
                # Sell if profit target met OR RSI is overbought (> 70)
                if (change >= MIN_PROFIT_TARGET) or (rsi > 70 and change > 0.002):
                    await close_position(pos, bot_state['last_price'], application)
                elif change <= STOP_LOSS:
                    await close_position(pos, bot_state['last_price'], application)

            await asyncio.sleep(20)
        except Exception as e:
            await asyncio.sleep(10)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
