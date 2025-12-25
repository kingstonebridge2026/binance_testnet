import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION =====
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "pvoWQGHiBWeNOohDhZacMUqI3JEkb4HLMTm0xZL2eEFCLtwNTYxNThbZB4HFHIo7")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "12Psc7IH3VVJRwWb05MvgpWrsSKz6CWmXqnHxYJKeHPljBeQ68Xv5hUnoLsaV7kH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.0015  # Increased slightly for better fee clearance

# Initialize Exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

bot_state = {
    "virtual_balance": 100.0,
    "realized_pnl": 0.0,
    "trades_count": 0,
    "last_price": 0.0,
    "last_rsi": 0.0,
    "in_position": False,
    "entry_price": 0.0,
    "start_time": datetime.now()
}

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

# ===== TELEGRAM COMMANDS =====
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upnl = (bot_state['last_price'] - bot_state['entry_price']) * POSITION_SIZE if bot_state['in_position'] else 0
    status = "🎯 LONG (In Trade)" if bot_state['in_position'] else "💤 SNIPING (Waiting)"
    msg = (f"📊 <b>Bot Status: {status}</b>\n"
           f"Price: ${bot_state['last_price']:,.2f}\n"
           f"RSI: {bot_state['last_rsi']:.2f}\n"
           f"Unrealized P&L: ${upnl:.4f}")
    await update.message.reply_text(msg, parse_mode='HTML')

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (f"💰 <b>Wallet</b>\nBalance: ${bot_state['virtual_balance']:.2f}\n"
           f"Trades: {bot_state['trades_count']}\n"
           f"Total P&L: ${bot_state['realized_pnl']:.4f}")
    await update.message.reply_text(msg, parse_mode='HTML')

# ===== TRADING EXECUTION =====
async def execute_trade(side, price, application):
    try:
        if side == 'BUY':
            await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
            bot_state['in_position'] = True
            bot_state['entry_price'] = price
            msg = f"🟢 <b>SNIPER BUY</b>\nPrice: ${price:,.2f}\n<i>Waiting for +0.6% profit...</i>"
        else:
            await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
            pnl = (price - bot_state['entry_price']) * POSITION_SIZE
            fee = (POSITION_SIZE * price) * 0.002 # Accounting for buy+sell fee
            net = pnl - fee
            bot_state['virtual_balance'] += net
            bot_state['realized_pnl'] += net
            bot_state['trades_count'] += 1
            bot_state['in_position'] = False
            icon = "✅" if net > 0 else "❌"
            msg = f"{icon} <b>SNIPER SELL</b>\nNet Profit: ${net:.4f}\nBalance: ${bot_state['virtual_balance']:.2f}"

        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Execution Error: {e}")

# ===== MAIN LOOP =====
async def trading_loop(application):
    print("🚀 SNIPER MODE ACTIVE - Patient Scalping...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # Use standard 14 period for more reliable signals
            df['rsi'] = calculate_rsi(df['close'], 14)
            df['sma_20'] = df['close'].rolling(20).mean()
            df['std'] = df['close'].rolling(20).std()
            df['lower_band'] = df['sma_20'] - (df['std'] * 2.0)
            df['upper_band'] = df['sma_20'] + (df['std'] * 2.0)
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            bot_state['last_rsi'] = df['rsi'].iloc[-1]
            
            l = df.iloc[-1]

            if not bot_state['in_position']:
                # ONLY BUY when price is actually low (Bottom of Band AND RSI < 35)
                if l['close'] <= l['lower_band'] and l['rsi'] < 35:
                    await execute_trade('BUY', bot_state['last_price'], application)
            else:
                profit_pct = (bot_state['last_price'] - bot_state['entry_price']) / bot_state['entry_price']
                
                # ONLY SELL if we actually covered fees (0.6% target)
                # OR if RSI is extremely overbought
                if profit_pct >= 0.006 or l['rsi'] > 75:
                    await execute_trade('SELL', bot_state['last_price'], application)
                
                # Safety Stop Loss (1.5%)
                elif profit_pct <= -0.015:
                    await execute_trade('SELL', bot_state['last_price'], application)

            await asyncio.sleep(15) # Check every 15s to save CPU
        except Exception as e:
            print(f"Loop error: {e}")
            await asyncio.sleep(10)

async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("balance", balance_command))
    
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        await app.bot.send_message(TELEGRAM_CHAT_ID, "🎯 <b>Sniper Bot Online</b>\nMinimum profit target: 0.6%")
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())


