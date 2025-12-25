
import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION =====
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "ZWHzZmOXxWwB6Qo2PXM5oiC799JidzTsXiLkcIVTWJLceMxjq3Qy0xPbph229M3Q")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "QacEqE0bSlOvZitUSnybJaBTYalrRAG2nslB4aYCJraOtCsPUr2fQUxB5e0TsntL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.0001 

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

bot_state = {
    "virtual_balance": 10.0,
    "last_price": 0.0,
    "last_rsi": 0.0,
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "in_position": False,
    "entry_price": 0.0
}

# ===== TELEGRAM COMMANDS =====
async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (f"💰 <b>Simulation Account</b>\n"
           f"Balance: ${bot_state['virtual_balance']:.4f}\n"
           f"Status: {'In Trade' if bot_state['in_position'] else 'Idle'}")
    await update.message.reply_text(msg, parse_mode='HTML')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (f"📊 <b>Market</b>\nPrice: ${bot_state['last_price']:,.2f}\nRSI: {bot_state['last_rsi']:.2f}")
    await update.message.reply_text(msg, parse_mode='HTML')

# ===== TRADING LOGIC =====
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

async def execute_trade(side, price, application):
    try:
        if bot_state['virtual_balance'] < 5.0 and side == 'BUY':
            await application.bot.send_message(TELEGRAM_CHAT_ID, "⚠️ <b>Simulation Over:</b> Balance < $5")
            return

        # Real execution on Testnet
        order = await (exchange.create_market_buy_order(SYMBOL, POSITION_SIZE) if side == 'BUY' 
                       else exchange.create_market_sell_order(SYMBOL, POSITION_SIZE))
        
        fee_cost = (POSITION_SIZE * price) * 0.0004 
        
        if side == 'BUY':
            bot_state['in_position'] = True
            bot_state['entry_price'] = price
            msg = f"📈 <b>BUY (Simulated)</b>\nPrice: ${price}"
        else:
            pnl = (price - bot_state['entry_price']) * POSITION_SIZE if bot_state['entry_price'] > 0 else 0
            net_result = pnl - fee_cost
            bot_state['virtual_balance'] += net_result
            bot_state['in_position'] = False
            msg = f"📉 <b>SELL (Simulated)</b>\nNet P&L: ${net_result:.4f}\nBalance: ${bot_state['virtual_balance']:.2f}"

        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Trade Error: {e}")

async def trading_loop(application):
    print("Trading loop started...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            df['sma_fast'] = df['close'].rolling(window=5).mean()
            df['sma_slow'] = df['close'].rolling(window=20).mean()
            df['rsi'] = calculate_rsi(df['close'])
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            bot_state['last_rsi'] = df['rsi'].iloc[-1]
            
            latest, prev = df.iloc[-1], df.iloc[-2]
            
            # Logic: Cross + RSI
            if not bot_state['in_position'] and prev['sma_fast'] <= prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
                await execute_trade('BUY', bot_state['last_price'], application)
            elif bot_state['in_position'] and (prev['sma_fast'] >= prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow'] or latest['rsi'] > 75):
                await execute_trade('SELL', bot_state['last_price'], application)

            await asyncio.sleep(30) # Check every 30s
        except Exception as e:
            print(f"Loop Error: {e}")
            await asyncio.sleep(10)

# ===== MAIN RUNNER =====
async def main():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Start bot and trading loop concurrently
    async with application:
        await application.start()
        await application.updater.start_polling()
        print("Telegram Bot & Trading Loop Online...")
        await trading_loop(application) # Keeps the loop alive
        await application.updater.stop()
        await application.stop()

if __name__ == "__main__":
    asyncio.run(main())
