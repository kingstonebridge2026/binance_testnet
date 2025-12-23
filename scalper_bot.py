import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
import time
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION =====
# Pulls from Railway Variables; uses your provided keys as defaults
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "ZWHzZmOXxWwB6Qo2PXM5oiC799JidzTsXiLkcIVTWJLceMxjq3Qy0xPbph229M3Q")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "QacEqE0bSlOvZitUSnybJaBTYalrRAG2nslB4aYCJraOtCsPUr2fQUxB5e0TsntL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "8287625785:AAEr1IXBXadMg20hehUrwBoMEYaBBOY4OMU")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.001  # BTC amount

# Initialize async exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

# Shared memory for Telegram commands
bot_state = {
    "last_price": 0.0,
    "last_rsi": 0.0,
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# ===== TELEGRAM COMMANDS =====
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /start"""
    await update.message.reply_text(
        f"🤖 <b>Scalper Bot Active</b>\n"
        f"Tracking: {SYMBOL}\n"
        f"Started at: {bot_state['start_time']}\n"
        f"Use /status for updates.",
        parse_mode='HTML'
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /status"""
    msg = (f"📊 <b>Market Status</b>\n"
           f"Price: ${bot_state['last_price']:,.2f}\n"
           f"RSI: {bot_state['last_rsi']:.2f}\n"
           f"Time: {datetime.now().strftime('%H:%M:%S')}")
    await update.message.reply_text(msg, parse_mode='HTML')

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /balance"""
    try:
        bal = await exchange.fetch_balance()
        usdt = bal['total'].get('USDT', 0)
        btc = bal['total'].get('BTC', 0)
        await update.message.reply_text(f"💰 <b>Balance</b>\nUSDT: {usdt:.2f}\nBTC: {btc:.5f}", parse_mode='HTML')
    except Exception as e:
        await update.message.reply_text(f"❌ Error fetching balance: {e}")

# ===== TRADING LOGIC =====
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

async def execute_trade(side, price, application):
    """Executes a real trade on Testnet"""
    try:
        if side == 'BUY':
            order = await exchange.create_market_buy_order(SYMBOL, POSITION_SIZE)
        else:
            order = await exchange.create_market_sell_order(SYMBOL, POSITION_SIZE)
        
        emoji = "📈" if side == 'BUY' else "📉"
        msg = (f"{emoji} <b>{side} ORDER EXECUTED</b>\n"
               f"Price: ${price:,.2f}\n"
               f"Amount: {POSITION_SIZE} BTC\n"
               f"Order ID: {order['id']}")
        
        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
        
        # Log to file
        with open('trades.csv', 'a') as f:
            f.write(f"{datetime.now()},{side},{price},{POSITION_SIZE}\n")
            
    except Exception as e:
        await application.bot.send_message(TELEGRAM_CHAT_ID, f"❌ <b>Trade Error:</b>\n{str(e)[:100]}", parse_mode='HTML')

async def trading_loop(application):
    """The background task that analyzes the market"""
    print("Trading loop started...")
    while True:
        try:
            # 1. Fetch Data
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # 2. Indicators
            df['sma_fast'] = df['close'].rolling(window=5).mean()
            df['sma_slow'] = df['close'].rolling(window=20).mean()
            df['rsi'] = calculate_rsi(df['close'])
            
            # 3. Update Global State
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            bot_state['last_rsi'] = df['rsi'].iloc[-1]
            
            # 4. Signal Logic
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Golden Cross (BUY)
            if prev['sma_fast'] <= prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
                await execute_trade('BUY', bot_state['last_price'], application)
            
            # Death Cross (SELL)
            elif prev['sma_fast'] >= prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow']:
                await execute_trade('SELL', bot_state['last_price'], application)

            # RSI Signals (Optional addition)
            elif latest['rsi'] < 30:
                await execute_trade('BUY', bot_state['last_price'], application)
            elif latest['rsi'] > 70:
                await execute_trade('SELL', bot_state['last_price'], application)

            await asyncio.sleep(60) # Wait for next candle
            
        except Exception as e:
            print(f"Error in loop: {e}")
            await asyncio.sleep(30)

# ===== STARTUP =====
if __name__ == "__main__":
    # Initialize Telegram Bot
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("balance", balance_command))
    
    # Start the background trading loop
    asyncio.get_event_loop().create_task(trading_loop(application))
    
    # Run the bot
    print("Bot is online and listening...")
    application.run_polling()
