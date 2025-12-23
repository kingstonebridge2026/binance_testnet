import ccxt
import pandas as pd
import time
import os
from datetime import datetime
import requests

# ===== CONFIGURATION (SECURE VERSION) =====
# It will try to get keys from Railway Variables; if not found, it uses your defaults
BINANCE_API_KEY = os.getenv("BINANCE_KEY", "ZWHzZmOXxWwB6Qo2PXM5oiC799JidzTsXiLkcIVTWJLceMxjq3Qy0xPbph229M3Q")
BINANCE_SECRET = os.getenv("BINANCE_SECRET", "QacEqE0bSlOvZitUSnybJaBTYalrRAG2nslB4aYCJraOtCsPUr2fQUxB5e0TsntL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "8287625785:AAEr1IXBXadMg20hehUrwBoMEYaBBOY4OMU")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
POSITION_SIZE = 0.001 

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    }
})

# FIXED: Proper way to enable Testnet in CCXT to avoid -2008 error
exchange.set_sandbox_mode(True)

# ===== TELEGRAM FUNCTIONS =====
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

def send_telegram_alert(signal, price, details=""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"🚨 <b>{signal}</b>\n📊 Pair: {SYMBOL}\n💰 Price: ${price:,.2f}\n🕒 Time: {timestamp}\n{details}"
    send_telegram(message)

# ===== TRADING FUNCTIONS =====
def get_indicators(df):
    df['sma_fast'] = df['close'].rolling(window=5).mean()
    df['sma_slow'] = df['close'].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    return df

def calculate_rsi(prices, period=14):
    """Fixed for Pandas 2.x compatibility"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Prevent division by zero
    rs = gain / loss.replace(0, 0.00001)
    return 100 - (100 / (1 + rs))

def check_signals(df):
    if len(df) < 20: return []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    
    if prev['sma_fast'] <= prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
        signals.append('BUY')
    if prev['sma_fast'] >= prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow']:
        signals.append('SELL')
    if latest['rsi'] < 30: signals.append('RSI_BUY')
    if latest['rsi'] > 70: signals.append('RSI_SELL')
    return signals

def execute_trade(signal, current_price):
    # This is a Paper Trading simulation
    side = 'BUY' if 'BUY' in signal else 'SELL'
    emoji = "📈" if side == 'BUY' else "📉"
    
    send_telegram_alert(f"{side} SIGNAL", current_price, f"{emoji} Executing {side} for {POSITION_SIZE} {SYMBOL}")
    log_trade(side, current_price, POSITION_SIZE)

def log_trade(side, price, quantity):
    with open('trades_log.csv', 'a') as f:
        f.write(f"{datetime.now()},{side},{price},{quantity}\n")

# ===== MAIN LOOP =====
def trading_loop():
    print("Starting scalping bot...")
    send_telegram(f"🤖 Scalping Bot Started\nPair: {SYMBOL}\nTimeframe: {TIMEFRAME}\nMode: TESTNET")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df = get_indicators(df)
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            
            signals = check_signals(df)
            for signal in signals:
                execute_trade(signal, current_price)
            
            # Heartbeat every 15 mins
            if datetime.now().minute % 15 == 0:
                send_telegram(f"📊 Status: BTC ${current_price:,.2f} | RSI: {df.iloc[-1]['rsi']:.2f}")
            
            time.sleep(60)
            
        except Exception as e:
            print(f"Error: {e}")
            send_telegram(f"❌ Bot Error: {str(e)[:100]}")
            time.sleep(60)

if __name__ == "__main__":
    trading_loop()
