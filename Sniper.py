import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
import uuid
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION =====
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "your_key")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "your_secret")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_id")

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.0012 
MAX_TRADES = 5  # Allow up to 5 simultaneous trades
MIN_PROFIT_TARGET = 0.005 # 0.5% (Covers 0.2% fees + 0.3% pure profit)
ROUND_TRIP_FEE = 0.002    # 0.2% total for buy + sell

# Initialize Exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

# Shared Bot State (Multi-Trade)
bot_state = {
    "virtual_balance": 100.0,
    "realized_pnl": 0.0,
    "trades_count": 0,
    "last_price": 0.0,
    "positions": [], # List of active trade dicts
    "last_trade_time": datetime.now() - timedelta(minutes=60)
}

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

# ===== TELEGRAM COMMANDS =====
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pos_count = len(bot_state['positions'])
    msg = f"📊 <b>Market: ${bot_state['last_price']:,.2f}</b>\n"
    msg += f"Active Trades: {pos_count}/{MAX_TRADES}\n\n"
    
    for i, p in enumerate(bot_state['positions']):
        pnl = (bot_state['last_price'] - p['entry_price']) * POSITION_SIZE
        msg += f"Trade #{i+1}: Entry ${p['entry_price']:,.2f} | P&L: ${pnl:.4f}\n"
    
    await update.message.reply_text(msg, parse_mode='HTML')

# ===== TRADING EXECUTION =====
async def open_position(price, application):
    try:
        # Check if we have room and it's been at least 10 mins since last trade
        if len(bot_state['positions']) >= MAX_TRADES: return
        
        await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
        new_pos = {
            "id": str(uuid.uuid4())[:8],
            "entry_price": price,
            "time": datetime.now()
        }
        bot_state['positions'].append(new_pos)
        bot_state['last_trade_time'] = datetime.now()
        
        msg = f"🟢 <b>MULTI-TRADE OPENED</b>\nEntry: ${price:,.2f}\nPositions: {len(bot_state['positions'])}/{MAX_TRADES}"
        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Open Error: {e}")

async def close_position(pos, price, application):
    try:
        await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        pnl = (price - pos['entry_price']) * POSITION_SIZE
        fee_cost = (POSITION_SIZE * price) * ROUND_TRIP_FEE
        net_profit = pnl - fee_cost
        
        bot_state['virtual_balance'] += net_profit
        bot_state['realized_pnl'] += net_profit
        bot_state['trades_count'] += 1
        bot_state['positions'].remove(pos)
        
        msg = (f"🔴 <b>POSITION CLOSED</b>\n"
               f"Profit: ${net_profit:.4f}\n"
               f"Wallet: ${bot_state['virtual_balance']:.2f}")
        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Close Error: {e}")

# ===== MAIN LOOP =====
async def trading_loop(application):
    print("🚀 MULTI-TRADE SNIPER ACTIVE...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            df['rsi'] = calculate_rsi(df['close'], 14)
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            current_rsi = df['rsi'].iloc[-1]
            
            # 1. CHECK FOR NEW ENTRIES
            # Strategy: Only buy if RSI < 30 and we haven't traded in 10 minutes
            time_since_last = datetime.now() - bot_state['last_trade_time']
            if len(bot_state['positions']) < MAX_TRADES and current_rsi < 30 and time_since_last.seconds > 600:
                await open_position(bot_state['last_price'], application)
            
            # 2. CHECK ALL OPEN POSITIONS FOR PROFIT
            for pos in bot_state['positions'][:]: # Use slice to avoid list mutation errors
                profit_pct = (bot_state['last_price'] - pos['entry_price']) / pos['entry_price']
                
                # GUARANTEED PROFIT RULE:
                # Sell only if RSI is high (overbought) OR we hit the +0.5% target
                if profit_pct >= MIN_PROFIT_TARGET or current_rsi > 70:
                    # Double check: Never sell at a loss unless it's an emergency (-2%)
                    if profit_pct > 0.002 or profit_pct < -0.02:
                        await close_position(pos, bot_state['last_price'], application)

            await asyncio.sleep(20) 
        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(10)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("status", status_command))
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
