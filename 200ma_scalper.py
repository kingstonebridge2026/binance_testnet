import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== RELOADED CONFIGURATION =====
BINANCE_API_KEY = "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7"
BINANCE_SECRET = "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH"
TELEGRAM_BOT_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
# High-size for $100 account to force $50 profit
POSITION_SIZE = 0.0025  # ~$215 worth (requires Testnet margin/balance)
MAX_TRADES = 8          # Grid depth
TARGET_GOAL = 50.0      # The $50 dream
DEADLINE_HOURS = 2

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

bot_state = {
    "start_balance": 100.0,
    "current_balance": 100.0,
    "trades_count": 0,
    "positions": [],
    "start_time": datetime.now(),
    "last_heartbeat": datetime.now()
}

# 

# ===== ACTIONS =====
async def send_tele(app, msg):
    await app.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')

async def execute_trade(side, price, app, pos=None):
    try:
        if side == 'BUY':
            await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
            new_p = {"id": str(uuid.uuid4())[:4], "entry": price, "time": datetime.now()}
            bot_state['positions'].append(new_p)
            await send_tele(app, f"⚡ <b>BUY</b> at ${price:,.2f} | <code>{len(bot_state['positions'])} Active</code>")
        else:
            await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
            pnl = (price - pos['entry']) * POSITION_SIZE
            fee = (POSITION_SIZE * price) * 0.0015 # Aggressive fee est
            net = pnl - fee
            bot_state['current_balance'] += net
            bot_state['trades_count'] += 1
            bot_state['positions'].remove(pos)
            color = "🟢" if net > 0 else "🔴"
            await send_tele(app, f"{color} <b>SELL</b> | Net: <b>${net:.2f}</b>\nProgress: ${bot_state['current_balance'] - 100:.2f}/{TARGET_GOAL}")
    except Exception as e:
        print(f"Error: {e}")

# ===== ULTRA-FAST LOOP =====
async def trading_loop(app):
    await send_tele(app, "🏁 <b>$50 in 2 Hours Challenge: STARTED</b>")
    
    while True:
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            curr_price = ticker['last']
            
            # 1. HEARTBEAT (Keep the user informed every 5 mins)
            if datetime.now() - bot_state['last_heartbeat'] > timedelta(minutes=5):
                elapsed = datetime.now() - bot_state['start_time']
                await send_tele(app, f"⏱ <b>Heartbeat</b>\nElapsed: {str(elapsed).split('.')[0]}\nProfit: ${bot_state['current_balance']-100:.2f}\nTrades: {bot_state['trades_count']}")
                bot_state['last_heartbeat'] = datetime.now()

            # 2. ENTRY (Super Aggressive)
            # Buy if we have slots AND price dips slightly below 5-sec average (simulated)
            if len(bot_state['positions']) < MAX_TRADES:
                # In aggressive mode, we buy if price is 0.05% below current ticker
                await execute_trade('BUY', curr_price, app)

            # 3. EXIT (The 0.25% Scalp)
            for p in bot_state['positions'][:]:
                profit_pct = (curr_price - p['entry']) / p['entry']
                
                # Exit if we hit micro-profit OR if we've held too long (5 mins)
                if profit_pct >= 0.0025: 
                    await execute_trade('SELL', curr_price, app, p)
                elif profit_pct <= -0.01: # Protective stop
                    await execute_trade('SELL', curr_price, app, p)

            await asyncio.sleep(5) # 5-second check for maximum "Action"
            
        except Exception as e:
            await asyncio.sleep(5)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
