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
POSITION_SIZE = 0.0012  # Slightly above min to ensure it clears fees

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
    status = "📈 LONG" if bot_state['in_position'] else "😴 IDLE"
    msg = (f"📊 <b>Bot Status: {status}</b>\n"
           f"Price: ${bot_state['last_price']:,.2f}\n"
           f"RSI: {bot_state['last_rsi']:.2f}\n"
           f"Unrealized: ${upnl:.4f}")
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
            # In Spot Testnet, sometimes market buy needs the "cost" in USDT
            await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
            bot_state['in_position'] = True
            bot_state['entry_price'] = price
            msg = f"🟢 <b>BUY ORDER</b>\nPrice: ${price:,.2f}"
        else:
            await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
            pnl = (price - bot_state['entry_price']) * POSITION_SIZE
            fee = (POSITION_SIZE * price) * 0.001 # 0.1% Spot Fee
            net = pnl - fee
            bot_state['virtual_balance'] += net
            bot_state['realized_pnl'] += net
            bot_state['trades_count'] += 1
            bot_state['in_position'] = False
            msg = f"🔴 <b>SELL ORDER</b>\nProfit: ${net:.4f}\nBalance: ${bot_state['virtual_balance']:.2f}"

        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Trade Error: {e}")

# ===== MAIN LOOP =====
async def trading_loop(application):
    print("🚀 SPOT SCALPER ACTIVE...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=30)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            df['sma_20'] = df['close'].rolling(20).mean()
            df['std'] = df['close'].rolling(20).std()
            df['upper'] = df['sma_20'] + (df['std'] * 1.3)
            df['lower'] = df['sma_20'] - (df['std'] * 1.3)
            df['rsi'] = calculate_rsi(df['close'], 3)
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            bot_state['last_rsi'] = df['rsi'].iloc[-1]
            
            l = df.iloc[-1]

            if not bot_state['in_position']:
                # BUY if price is near lower band OR RSI is oversold
                if l['close'] < l['lower'] or l['rsi'] < 40:
                    await execute_trade('BUY', bot_state['last_price'], application)
            else:
                profit = (bot_state['last_price'] - bot_state['entry_price']) / bot_state['entry_price']
                # SELL if price hits upper band OR we have 0.3% profit
                if l['close'] > l['upper'] or l['rsi'] > 80 or profit > 0.003:
                    await execute_trade('SELL', bot_state['last_price'], application)

            await asyncio.sleep(10)
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
        # drop_pending_updates fixes the Conflict error
        await app.updater.start_polling(drop_pending_updates=True)
        await app.bot.send_message(TELEGRAM_CHAT_ID, "⚡ <b>Spot Scalper Online</b>\nHigh-Frequency Mode")
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())

