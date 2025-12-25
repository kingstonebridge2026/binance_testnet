import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import asyncio
import uuid
from datetime import datetime
from telegram.ext import ApplicationBuilder

# ===== PRO CREDENTIALS =====
BINANCE_API_KEY = "0hb4IO19WSbyO6VlM8S0Aa8tWwHSYhtQhDRoOG70iu912J95qm7HhtRspAoykSml"
BINANCE_SECRET = "RE8tftdsuG4MzcMfR4VNy6yvkho27qDMGiLZ6yR4cHXRWmCq1sV5AfBmgIIH06dK"
TELEGRAM_BOT_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
POSITION_SIZE = 0.0025 # Leveraging Testnet for high turnover
MAX_SLOTS = 15 
GOAL_PER_HOUR = 10.0

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET,
    'enableRateLimit': True, 'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True)

bot_state = {"pnl": 0.0, "positions": [], "last_price": 0.0, "dash_msg_id": None}

def build_deep_dash(z_score):
    upnl = sum([(bot_state['last_price'] - p['entry']) * POSITION_SIZE for p in bot_state['positions']])
    market_mood = "🔥 VOLATILE" if abs(z_score) > 2 else "😴 CALM"
    
    dash = (
        f"🧠 <b>DEEP-LEARNING ALPHA v3</b>\n"
        f"<code>Market Mood:</code> {market_mood}\n"
        f"<code>Z-Score:    </code> {z_score:.2f}\n"
        f"--------------------------\n"
        f"💰 <b>Banked:</b>  +${bot_state['pnl']:.2f}\n"
        f"🌊 <b>Floating:</b> ${upnl:.4f}\n"
        f"--------------------------\n"
        f"🎯 <b>Hourly:</b> {((bot_state['pnl']/GOAL_PER_HOUR)*100):.1f}% Completed\n"
        f"<i>Slots: {len(bot_state['positions'])}/{MAX_SLOTS}</i>"
    )
    return dash

async def update_terminal(app, z_score):
    try:
        text = build_deep_dash(z_score)
        if bot_state['dash_msg_id'] is None:
            msg = await app.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode='HTML')
            bot_state['dash_msg_id'] = msg.message_id
        else:
            await app.bot.edit_message_text(text, TELEGRAM_CHAT_ID, bot_state['dash_msg_id'], parse_mode='HTML')
    except: pass

async def brain_loop(app):
    print("🚀 DEEP LEARNING STRATEGY STARTING...")
    while True:
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            price = ticker['last']
            
            # --- THE DEEP LEARNING CALCULATION ---
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, '1m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            # Calculate Z-Score (Distance from Mean in Standard Deviations)
            mean = df['c'].mean()
            std = df['c'].std()
            z_score = (price - mean) / std if std > 0 else 0

            # 1. ENTRY: "The Statistical Snap"
            # Buy only if price is more than 2.0 Standard Deviations below the mean
            if len(bot_state['positions']) < MAX_SLOTS and z_score < -2.0:
                await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
                bot_state['positions'].append({"entry": price, "id": str(uuid.uuid4())[:3]})

            # 2. EXIT: "The Quick Reversion"
            for p in bot_state['positions'][:]:
                profit_pct = (price - p['entry']) / p['entry']
                
                # Dynamic TP: Exit as soon as price returns to the Mean (Z-Score > 0)
                # OR if we hit a 0.35% hard target
                if z_score >= 0.2 or profit_pct >= 0.0035:
                    await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
                    net = (POSITION_SIZE * price * profit_pct) - (POSITION_SIZE * price * 0.001)
                    bot_state['pnl'] += net
                    bot_state['positions'].remove(p)

            await update_terminal(app, z_score)
            await asyncio.sleep(1.5)
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
