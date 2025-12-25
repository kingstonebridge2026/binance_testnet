import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import uuid
from datetime import datetime, timedelta
from telegram.ext import ApplicationBuilder

# ===== PRO CREDENTIALS =====
BINANCE_API_KEY = "0hb4IO19WSbyO6VlM8S0Aa8tWwHSYhtQhDRoOG70iu912J95qm7HhtRspAoykSml"
BINANCE_SECRET = "RE8tftdsuG4MzcMfR4VNy6yvkho27qDMGiLZ6yR4cHXRWmCq1sV5AfBmgIIH06dK"
TELEGRAM_BOT_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TELEGRAM_CHAT_ID = "5665906172"

SYMBOL = "BTC/USDT"
# Using high relative size to achieve the 100%/hr goal
POSITION_SIZE = 0.0015 # ~ $130 (Utilizing Testnet margin/liquidity)
MAX_SLOTS = 12 
GOAL_PER_HOUR = 10.0

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY, 'secret': BINANCE_SECRET,
    'enableRateLimit': True, 'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True)

bot_state = {
    "balance": 10.0,
    "pnl": 0.0,
    "positions": [],
    "wins": 0,
    "last_price": 0.0,
    "trend": "NEUTRAL",
    "dash_msg_id": None,
    "start_time": datetime.now()
}

def build_advanced_dash():
    upnl = sum([(bot_state['last_price'] - p['entry']) * POSITION_SIZE for p in bot_state['positions']])
    elapsed = (datetime.now() - bot_state['start_time']).seconds / 3600
    expected = GOAL_PER_HOUR * elapsed
    
    # Trend Visualizer
    trend_arrow = "➡️" if bot_state['trend'] == "NEUTRAL" else ("🚀" if bot_state['trend'] == "BULL" else "📉")
    
    dash = (
        f"🤖 <b>AI ALPHA v2 (2025)</b>\n"
        f"<code>Trend:</code> {bot_state['trend']} {trend_arrow}\n"
        f"<code>Price:</code> ${bot_state['last_price']:,.2f}\n"
        f"--------------------------\n"
        f"💰 <b>Banked:</b>  +${bot_state['pnl']:.2f}\n"
        f"🌊 <b>Floating:</b> ${upnl:.4f}\n"
        f"--------------------------\n"
        f"📈 <b>Progress:</b> {((bot_state['pnl']/GOAL_PER_HOUR)*100):.1f}% of Hr Goal\n"
        f"<i>Active Traps: {len(bot_state['positions'])}/{MAX_SLOTS}</i>"
    )
    return dash

async def update_terminal(app):
    try:
        text = build_advanced_dash()
        if bot_state['dash_msg_id'] is None:
            msg = await app.bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode='HTML')
            bot_state['dash_msg_id'] = msg.message_id
        else:
            await app.bot.edit_message_text(text, TELEGRAM_CHAT_ID, bot_state['dash_msg_id'], parse_mode='HTML')
    except: pass

async def ai_engine(app):
    print("🚀 AI ALPHA v2 STARTING...")
    while True:
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            price = ticker['last']
            
            # AI DATA CRUNCH
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, '1m', limit=30)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            # Predictive Indicator: Kalman-style EMA
            fast_ema = df['c'].ewm(span=5).mean().iloc[-1]
            slow_ema = df['c'].ewm(span=15).mean().iloc[-1]
            bot_state['trend'] = "BULL" if fast_ema > slow_ema else "BEAR"

            # 1. SMART ENTRY (DCA Gapping)
            if len(bot_state['positions']) < MAX_SLOTS:
                # Only buy if price is below fast EMA (Mean Reversion)
                if price < fast_ema * 0.9995: 
                    await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
                    bot_state['positions'].append({"entry": price, "id": str(uuid.uuid4())[:3]})

            # 2. SMART EXIT (Trailing-style)
            for p in bot_state['positions'][:]:
                profit_pct = (price - p['entry']) / p['entry']
                
                # Dynamic TP: If trend is BULL, wait for 0.4%, if BEAR, exit at 0.2%
                target = 0.004 if bot_state['trend'] == "BULL" else 0.002
                
                if profit_pct >= target:
                    await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
                    net = (POSITION_SIZE * price * profit_pct) - (POSITION_SIZE * price * 0.001)
                    bot_state['pnl'] += net
                    bot_state['wins'] += 1
                    bot_state['positions'].remove(p)

            await update_terminal(app)
            await asyncio.sleep(1.5) # Hyper-refresh
        except Exception as e:
            await asyncio.sleep(5)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await ai_engine(app)

if __name__ == "__main__":
    asyncio.run(main())

