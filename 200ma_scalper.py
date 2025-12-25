import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===== CONFIGURATION (Use Railway Variables or Replace) =====
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "0NLIHcV6lIWDuCakzAAUSE2mq6BrxmDNHCn6l0lCPgq7AAFWcPiqkz2Q9eTbW9Ye")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "5LVq1iHl5MRAS56SHsrMmx4wAqe1TvURAvNLrlUR4hGcru6F8CpMjRzJK8BqtNiF")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "8488789199:AAGDbx-hu2993dG5O6LJEiSN0nEpFWuVWwk")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5665906172")


SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.0001  # Min lot for 10$ realism

# Initialize Exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})
exchange.set_sandbox_mode(True)

# Shared Bot State (Virtual Wallet)
bot_state = {
    "virtual_balance": 10.0,
    "realized_pnl": 0.0,
    "trades_count": 0,
    "last_price": 0.0,
    "last_rsi": 0.0,
    "in_position": False,
    "entry_price": 0.0,
    "start_time": datetime.now()
}

# ===== INDICATORS =====
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

# ===== TELEGRAM COMMANDS =====
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 <b>Bot Started!</b> Tracking BTC/USDT (1m).\nUse /status or /balance to check progress.", parse_mode='HTML')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows Market data + Unrealized P&L"""
    upnl = 0.0
    if bot_state['in_position']:
        upnl = (bot_state['last_price'] - bot_state['entry_price']) * POSITION_SIZE
    
    msg = (f"📊 <b>Market Status</b>\n"
           f"Price: ${bot_state['last_price']:,.2f}\n"
           f"RSI: {bot_state['last_rsi']:.2f}\n\n"
           f"🎯 <b>Current Trade</b>\n"
           f"Status: {'LONG' if bot_state['in_position'] else 'IDLE'}\n"
           f"Entry: ${bot_state['entry_price']:,.2f}\n"
           f"Unrealized P&L: {'+' if upnl >= 0 else ''}${upnl:.4f}")
    await update.message.reply_text(msg, parse_mode='HTML')

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows the Fake 10$ Wallet and Total Profits"""
    uptime = datetime.now() - bot_state['start_time']
    msg = (f"💰 <b>Virtual Wallet ($10 Start)</b>\n"
           f"Current Balance: <b>${bot_state['virtual_balance']:.2f}</b>\n"
           f"Total Realized P&L: ${bot_state['realized_pnl']:.4f}\n"
           f"Trades Completed: {bot_state['trades_count']}\n"
           f"Uptime: {str(uptime).split('.')[0]}")
    await update.message.reply_text(msg, parse_mode='HTML')

# ===== TRADING EXECUTION =====
async def execute_trade(side, price, application):
    try:
        # Binance Execution (Testnet)
        order = await (exchange.create_market_buy_order(SYMBOL, POSITION_SIZE) if side == 'BUY' 
                       else exchange.create_market_sell_order(SYMBOL, POSITION_SIZE))
        
        fee = (POSITION_SIZE * price) * 0.0004 # 0.04% fee
        
        if side == 'BUY':
            bot_state['in_position'] = True
            bot_state['entry_price'] = price
            msg = f"🟢 <b>BUY ORDER</b>\nPrice: ${price:,.2f}\nSize: {POSITION_SIZE} BTC"
        else:
            pnl = (price - bot_state['entry_price']) * POSITION_SIZE
            net_profit = pnl - fee
            bot_state['virtual_balance'] += net_profit
            bot_state['realized_pnl'] += net_profit
            bot_state['trades_count'] += 1
            bot_state['in_position'] = False
            
            msg = (f"🔴 <b>SELL ORDER (Closed)</b>\n"
                   f"Price: ${price:,.2f}\n"
                   f"Trade P&L: ${net_profit:.4f}\n"
                   f"Wallet: ${bot_state['virtual_balance']:.2f}")

        await application.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='HTML')
    except Exception as e:
        print(f"Trade Error: {e}")

# ===== MAIN LOOP =====
# Add this new indicator function
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

async def trading_loop(application):
    print("Advanced Scalping Loop Active...")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=210) # Fetch more for EMA 200
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            
            # --- ADVANCED INDICATORS ---
            df['ema_200'] = calculate_ema(df['close'], 200) # The "Trend Filter"
            df['ema_50'] = calculate_ema(df['close'], 50)   # The "Medium Filter"
            df['rsi'] = calculate_rsi(df['close'], 14)
            
            ticker = await exchange.fetch_ticker(SYMBOL)
            bot_state['last_price'] = ticker['last']
            bot_state['last_rsi'] = df['rsi'].iloc[-1]
            
            l = df.iloc[-1] # Current candle
            p = df.iloc[-2] # Previous candle

            # --- OPTIMIZED SIGNAL LOGIC ---
            if not bot_state['in_position']:
                # CONDITION 1: Price is in an UP-TREND (Price > EMA 200)
                # CONDITION 2: Price is dipping but recovering (Price > EMA 50)
                # CONDITION 3: RSI is not overbought (< 60)
                if (l['close'] > l['ema_200'] and 
                    l['close'] > l['ema_50'] and 
                    l['rsi'] > 45 and p['rsi'] <= 45):
                    
                    await execute_trade('BUY', bot_state['last_price'], application)

            elif bot_state['in_position']:
                # EXIT 1: Profit Target Reached (0.5%)
                profit_pct = (l['close'] - bot_state['entry_price']) / bot_state['entry_price']
                
                # EXIT 2: Trend Reversal (Price drops below EMA 50)
                if profit_pct >= 0.005: # Take Profit at 0.5%
                    await execute_trade('SELL', bot_state['last_price'], application, reason="TAKE PROFIT")
                elif l['close'] < l['ema_50']: # Early Trend Exit
                    await execute_trade('SELL', bot_state['last_price'], application, reason="TREND REVERSAL")
                elif profit_pct <= -0.008: # Tight Stop Loss (0.8%)
                    await execute_trade('SELL', bot_state['last_price'], application, reason="STOP LOSS")

            await asyncio.sleep(15) # Fast scanning
        except Exception as e:
            print(f"Logic Error: {e}")
            await asyncio.sleep(10)


async def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("balance", balance_command))
    
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        await app.bot.send_message(TELEGRAM_CHAT_ID, "✅ <b>System Online</b>\nSimulation Mode: Active")
        await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
