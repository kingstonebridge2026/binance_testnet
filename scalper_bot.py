import ccxt
import pandas as pd
import time
import threading
from datetime import datetime
import requests

# ===== CONFIGURATION =====
BINANCE_API_KEY = "ZWHzZmOXxWwB6Qo2PXM5oiC799JidzTsXiLkcIVTWJLceMxjq3Qy0xPbph229M3Q"
BINANCE_SECRET = "QacEqE0bSlOvZitUSnybJaBTYalrRAG2nslB4aYCJraOtCsPUr2fQUxB5e0TsntL"
TELEGRAM_BOT_TOKEN = "8287625785:AAEr1IXBXadMg20hehUrwBoMEYaBBOY4OMU"
TELEGRAM_CHAT_ID ="5665906172"

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"  # 1 minute for scalping
POSITION_SIZE = 0.001  # BTC amount
PROFIT_TARGET = 0.002  # 0.2%
STOP_LOSS = 0.001  # 0.1%

# Initialize exchange (paper trading enabled)
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',  # Use 'future' for futures
        'adjustForTimeDifference': True,
    },
    'urls': {
        'api': {
            'public': 'https://testnet.binance.vision/api/v3',
            'private': 'https://testnet.binance.vision/api/v3',
        }
    }
})

# ===== TELEGRAM FUNCTIONS =====
def send_telegram(message):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

def send_telegram_alert(signal, price, details=""):
    """Send formatted alert"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"""
🚨 <b>{signal}</b>
📊 Pair: {SYMBOL}
💰 Price: ${price:,.2f}
🕒 Time: {timestamp}
{details}
"""
    send_telegram(message)

# ===== TRADING FUNCTIONS =====
def get_indicators(df):
    """Calculate simple indicators"""
    df['sma_fast'] = df['close'].rolling(window=5).mean()
    df['sma_slow'] = df['close'].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_signals(df):
    """Generate trading signals"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    
    # Golden cross
    if prev['sma_fast'] <= prev['sma_slow'] and latest['sma_fast'] > latest['sma_slow']:
        signals.append('BUY')
    
    # Death cross
    if prev['sma_fast'] >= prev['sma_slow'] and latest['sma_fast'] < latest['sma_slow']:
        signals.append('SELL')
    
    # RSI oversold
    if latest['rsi'] < 30:
        signals.append('RSI_BUY')
    
    # RSI overbought
    if latest['rsi'] > 70:
        signals.append('RSI_SELL')
    
    return signals

def execute_trade(signal, current_price):
    """Execute paper trade"""
    if 'BUY' in signal:
        order = {
            'symbol': SYMBOL.replace('/', ''),
            'side': 'BUY',
            'type': 'MARKET',
            'quantity': POSITION_SIZE,
            'timestamp': int(time.time() * 1000),
            'paper': True  # Paper trade flag
        }
        send_telegram_alert("BUY SIGNAL", current_price, 
                          f"📈 Buying {POSITION_SIZE} {SYMBOL}")
        # In paper trading, just log the order
        log_trade('BUY', current_price, POSITION_SIZE)
        return order
        
    elif 'SELL' in signal:
        order = {
            'symbol': SYMBOL.replace('/', ''),
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': POSITION_SIZE,
            'timestamp': int(time.time() * 1000),
            'paper': True
        }
        send_telegram_alert("SELL SIGNAL", current_price,
                          f"📉 Selling {POSITION_SIZE} {SYMBOL}")
        log_trade('SELL', current_price, POSITION_SIZE)
        return order
    
    return None

def log_trade(side, price, quantity):
    """Log trade to file"""
    with open('trades_log.csv', 'a') as f:
        f.write(f"{datetime.now()},{side},{price},{quantity}\n")

# ===== MAIN LOOP =====
def trading_loop():
    """Main trading loop"""
    print("Starting scalping bot...")
    send_telegram("🤖 Scalping Bot Started\n" + 
                  f"Pair: {SYMBOL}\n" +
                  f"Timeframe: {TIMEFRAME}")
    
    while True:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            df = get_indicators(df)
            
            # Get current price
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            
            # Check signals
            signals = check_signals(df)
            
            if signals:
                # Execute trade based on signals
                for signal in signals:
                    if signal in ['BUY', 'SELL', 'RSI_BUY', 'RSI_SELL']:
                        execute_trade([signal], current_price)
            
            # Send periodic update (every 15 minutes)
            if datetime.now().minute % 15 == 0:
                send_telegram(f"📊 Status Update\n"
                            f"Price: ${current_price:,.2f}\n"
                            f"RSI: {df.iloc[-1]['rsi']:.2f}\n"
                            f"Fast SMA: {df.iloc[-1]['sma_fast']:.2f}\n"
                            f"Slow SMA: {df.iloc[-1]['sma_slow']:.2f}")
            
            # Wait for next candle
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error: {e}")
            send_telegram(f"❌ Bot Error: {str(e)[:100]}")
            time.sleep(60)

# ===== SETUP =====
if __name__ == "__main__":
    # First, set up Binance Testnet account
    print("\n" + "="*50)
    print("SCALPING BOT SETUP INSTRUCTIONS:")
    print("="*50)
    print("\n1. Get Binance Testnet API:")
    print("   Visit: https://testnet.binance.vision/")
    print("   Create account and get API keys")
    
    print("\n2. Create Telegram Bot:")
    print("   Message @BotFather on Telegram")
    print("   Send: /newbot")
    print("   Get bot token and chat ID from @userinfobot")
    
    print("\n3. Update config variables in the script")
    print("\n4. Install required packages:")
    print("   pip install ccxt pandas requests")
    
    print("\n5. Run the bot:")
    print("   python scalper_bot.py")
    print("="*50)
    
    # Uncomment to run directly:
    # trading_loop()
