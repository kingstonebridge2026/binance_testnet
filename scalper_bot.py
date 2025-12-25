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
POSITION_SIZE = 0.0001  # Set to 0.0001 for 10$ realism (Value approx $8.70)

# Initialize async exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

# Shared memory / Virtual Wallet
bot_state = {
    "virtual_balance": 10.0, # THE FAKE 10$ BALANCE
    "last_price": 0.0,
    "last_rsi": 0.0,
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "in_position": False,
    "entry_price": 0.0
}

# ===== TELEGRAM COMMANDS =====
async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows the Fake 10$ balance instead of the real 10k"""
    msg = (f"💰 <b>Virtual Account (Simulation)</b>\n"
           f"Current Balance: ${bot_state['virtual_balance']:.4f}\n"
           f"Status: {'In Trade' if bot_state['in_position'] else 'Idle'}")
    await update.message.reply_text(msg, parse_mode='HTML')

# ===== TRADING LOGIC =====
async def execute_trade(side, price, application):
    try:
        # Check if we have enough fake money ($5 minimum for Binance notional)
        if bot_state['virtual_balance'] < 5.0 and side == 'BUY':
            await application.bot.send_message(TELEGRAM_CHAT_ID, "⚠️ <b>Simulation Over:</b> Virtual balance too low to trade!")
            return

        # Execute on Testnet
        order = await (exchange.create_market_buy_order(SYMBOL, POSITION_SIZE) if side == 'BUY' 
                       else exchange.create_market_sell_order(SYMBOL, POSITION_SIZE))
        
        fee_cost = (POSITION_SIZE * price) * 0.0004 # Estimate 0.04% fee
        
        if side == 'BUY':
            bot_state['in_position'] = True
            bot_state['entry_price'] = price
            msg = f"📈 <b>BUY (Fake $10 Account)</b>\nPrice: ${price}\nSize: {POSITION_SIZE} BTC"
        else:
            # Calculate Profit/Loss for the virtual balance
            pnl = (price - bot_state['entry_price']) * POSITION_SIZE if bot_state['entry_price'] > 0 else 0
            net_result = pnl - fee_cost
            bot_state['virtual_balance'] += net_result
            bot_state['in_position'] = False
            
            color = "🟢" if net_result > 0 else "🔴"
            msg = (f"{color} <b>SELL (Trade Closed)</b>\n"
                   f"P&L: ${net_result:.4f}\n"
                   f"New Balance: ${bot_state['virtual_balance']:.2f}")

        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
            
    except Exception as e:
        print(f"Trade Error: {e}")

# (Rest of indicators and trading_loop stay same, calling execute_trade logic above)

