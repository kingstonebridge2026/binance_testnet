import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import uuid
from datetime import datetime
from telegram.ext import ApplicationBuilder
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# ===== PRO CREDENTIALS =====
BINANCE_API_KEY = "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7"
BINANCE_SECRET = "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH"
TELEGRAM_BOT_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
POSITION_SIZE = 0.0024 
MAX_SLOTS = 10 
TARGET_HOURLY = 25.0

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET,
    'enableRateLimit': True, 'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True)

bot_state = {
    "balance": 100.0, 
    "pnl": 0.0, 
    "positions": [], 
    "wins": 0, 
    "last_price": 0.0,
    "dash_msg_id": None
}

# ===== NEW: INTERACTIVE DASHBOARD BUILDER =====
def build_dashboard():
    upnl = 0
    for p in bot_state['positions']:
        upnl += (bot_state['last_price'] - p['entry']) * POSITION_SIZE
    
    status_icon = "🟢" if upnl >= 0 else "🔴"
    pos_count = len(bot_state['positions'])
    
    # Visual Price Bar (Like MetaTrader)
    bar = "▒▒▒▒▒▒▒▒▒▒"
    if pos_count > 0:
        progress = min(int((pos_count / MAX_SLOTS) * 10), 10)
        bar = "█" * progress + "▒" * (10 - progress)

    dash = (
        f"🖥 <b>AI LIVE TERMINAL</b>\n"
        f"<code>Market:</code> ${bot_state['last_price']:,.2f}\n"
        f"<code>Load:  </code> [{bar}] {pos_count}/{MAX_SLOTS}\n"
        f"--------------------------\n"
        f"💰 <b>Realized:</b>  +${bot_state['pnl']:.2f}\n"
        f"⏳ <b>Open P&L:</b> {status_icon} ${upnl:.4f}\n"
        f"--------------------------\n"
        f"🎯 <b>Hourly Goal:</b> ${bot_state['pnl']:.2f}/$25.00"
    )
    return dash

async def update_dashboard(app):
    text = build_dashboard()
    try:
        if bot_state['dash_msg_id'] is None:
            msg = await app.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode='HTML')
            bot_state['dash_msg_id'] = msg.message_id
        else:
            await app.bot.edit_message_text(
                text, TELEGRAM_CHAT_ID, bot_state['dash_msg_id'], parse_mode='HTML'
            )
    except:
        pass # Handle Telegram rate limits

# ===== THE BRAIN (Logic Intact) =====
async def brain_loop(app):
    print("🧠 AI-SCALPER INITIALIZED")
    
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, '1m', limit=20)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['ema'] = df['c'].ewm(span=7, adjust=False).mean()
            df['std'] = df['c'].rolling(7).std()
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            price = ticker['last']
            
            # ENTRY (Logic Unchanged)
            if len(bot_state['positions']) < MAX_SLOTS:
                if price < (df['ema'].iloc[-1] - (df['std'].iloc[-1] * 0.5)):
                    await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
                    bot_state['positions'].append({"id": str(uuid.uuid4())[:3], "entry": price})

            # EXIT (Logic Unchanged)
            for p in bot_state['positions'][:]:
                diff = (price - p['entry']) / p['entry']
                if diff >= 0.0035:
                    await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
                    net = (POSITION_SIZE * price * diff) - (POSITION_SIZE * price * 0.001)
                    bot_state['pnl'] += net
                    bot_state['wins'] += 1
                    bot_state['positions'].remove(p)
            
            # Update the Visual Dashboard every loop
            await update_dashboard(app)
            await asyncio.sleep(2) 
        except Exception as e:
            print(f"Loop error: {e}")
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
 
