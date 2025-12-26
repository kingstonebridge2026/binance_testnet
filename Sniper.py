import numpy as np
import pandas as pd
import asyncio
import ccxt.async_support as ccxt
import logging
import ta
import aiohttp
import time

# ==================== CONFIGURATION ====================
class Config:
    BINANCE_API_KEY = "r6hhHQubpwwnDYkYhhdSlk3MQPjTomUggf59gfXJ21hnBcfq3K4BIoSd1eE91V3N"
    BINANCE_SECRET = "B7ioAXzVHyYlxPOz3AtxzMC6FQBZaRj6i8A9FenSbsK8rBeCdGZHDhX6Dti22F2x"
    TELEGRAM_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
    TELEGRAM_CHAT_ID = "5665906172"
    
    SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
        "POL/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT", "APT/USDT",
        "ARB/USDT", "OP/USDT", "INJ/USDT", "TIA/USDT", "SUI/USDT"
    ]
    
    BASE_POSITION_USD = 50
    MAX_SLOTS = 15            
    RSI_BUY_LEVEL = 35       
    Z_SCORE_BUY = -1.5       
    TARGET_PROFIT = 0.008    
    STOP_LOSS = 0.015        
    USE_TESTNET = True  # Set to True for testing

# ==================== TRADING CORE ====================
class AlphaSniper:
    def __init__(self):
        # Configure exchange with proper testnet settings
        exchange_config = {
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Using spot trading to avoid FAPI issues
                'adjustForTimeDifference': True,
            }
        }
        
        # Only use testnet URLs for spot trading, not for futures
        if Config.USE_TESTNET:
            exchange_config['urls'] = {
                'api': {
                    'public': 'https://testnet.binance.vision/api',
                    'private': 'https://testnet.binance.vision/api',
                }
            }
            
        self.exchange = ccxt.binance(exchange_config)
        
        # Verify connection
        self.exchange.load_markets()
        
        self.positions = []
        self.banked_pnl = 0.0
        self.is_running = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("Sniper")
        
        # Rate limiting
        self.last_request_time = time.time()
        self.min_request_interval = 0.1  # 100ms between requests

    async def rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    async def send_telegram(self, message):
        """Send message to Telegram"""
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage"
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML"
                }
                async with session.post(url, json=data) as response:
                    if response.status != 200:
                        self.logger.error(f"Telegram API error: {await response.text()}")
        except Exception as e:
            self.logger.error(f"Telegram Error: {e}")

    def get_signals(self, df):
        """Calculate trading signals"""
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            rolling_mean = df['close'].rolling(window=20).mean()
            rolling_std = df['close'].rolling(window=20).std()
            df['zscore'] = (df['close'] - rolling_mean) / rolling_std
            return df.iloc[-1]
        except Exception as e:
            self.logger.error(f"Signal calculation error: {e}")
            return None

    async def fetch_market_data(self, symbol):
        """Fetch OHLCV data with error handling"""
        try:
            await self.rate_limit()
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, 
                '1m', 
                limit=50
            )
            if len(ohlcv) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(ohlcv)} candles")
                return None
            
            df = pd.DataFrame(ohlcv, columns=['t', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            self.logger.error(f"Data fetch error for {symbol}: {e}")
            return None

    async def place_order(self, symbol, side, amount, order_type='market'):
        """Place an order with error handling"""
        try:
            await self.rate_limit()
            
            if order_type == 'market':
                if side == 'buy':
                    order = await self.exchange.create_market_buy_order(symbol, amount)
                else:
                    order = await self.exchange.create_market_sell_order(symbol, amount)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            self.logger.info(f"Order placed: {symbol} {side} {amount}")
            return order
        except Exception as e:
            self.logger.error(f"Order error for {symbol}: {e}")
            return None

    async def monitor_markets(self):
        """Monitor markets for trading opportunities"""
        await self.send_telegram("🚀 <b>Sniper Online</b>\nTesting with Binance Testnet")
        
        while self.is_running:
            try:
                active_symbols = [pos['symbol'] for pos in self.positions]
                
                for symbol in Config.SYMBOLS:
                    if not self.is_running:
                        break
                    
                    # Check if we have available slots
                    if len(self.positions) >= Config.MAX_SLOTS:
                        self.logger.info(f"Max slots reached ({Config.MAX_SLOTS})")
                        await asyncio.sleep(10)
                        continue
                    
                    # Skip if already in position
                    if symbol in active_symbols:
                        continue
                    
                    # Fetch and analyze data
                    df = await self.fetch_market_data(symbol)
                    if df is None:
                        continue
                    
                    sig = self.get_signals(df)
                    if sig is None:
                        continue
                    
                    # Check for NaN values
                    if np.isnan(sig['rsi']) or np.isnan(sig['zscore']):
                        continue
                    
                    # Buy signal conditions
                    if sig['rsi'] < Config.RSI_BUY_LEVEL and sig['zscore'] < Config.Z_SCORE_BUY:
                        await self.rate_limit()
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        
                        # Calculate amount
                        amt = Config.BASE_POSITION_USD / price
                        
                        # Place buy order
                        order = await self.place_order(symbol, 'buy', amt)
                        if order:
                            self.positions.append({
                                'symbol': symbol,
                                'entry': price,
                                'amt': amt,
                                'timestamp': time.time()
                            })
                            message = (
                                f"🎯 <b>BUY SIGNAL: {symbol}</b>\n"
                                f"Entry: ${price:.4f}\n"
                                f"Amount: {amt:.6f}\n"
                                f"RSI: {sig['rsi']:.2f}, Z-Score: {sig['zscore']:.2f}"
                            )
                            await self.send_telegram(message)
                    
                    # Small delay between symbols
                    await asyncio.sleep(0.3)
                
                # Wait before next scan
                await asyncio.sleep(15)
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(30)

    async def manage_positions(self):
        """Monitor and manage open positions"""
        while self.is_running:
            try:
                if not self.positions:
                    await asyncio.sleep(5)
                    continue
                
                for pos in self.positions[:]:  # Use slice copy for safe removal
                    try:
                        await self.rate_limit()
                        ticker = await self.exchange.fetch_ticker(pos['symbol'])
                        current_price = ticker['last']
                        
                        # Calculate P&L
                        pnl_ratio = (current_price - pos['entry']) / pos['entry']
                        pnl_usd = (current_price - pos['entry']) * pos['amt']
                        
                        # Check exit conditions
                        should_exit = False
                        exit_reason = ""
                        
                        if pnl_ratio >= Config.TARGET_PROFIT:
                            should_exit = True
                            exit_reason = f"Target Profit ({Config.TARGET_PROFIT*100:.1f}%)"
                        elif pnl_ratio <= -Config.STOP_LOSS:
                            should_exit = True
                            exit_reason = f"Stop Loss ({Config.STOP_LOSS*100:.1f}%)"
                        
                        if should_exit:
                            # Place sell order
                            order = await self.place_order(pos['symbol'], 'sell', pos['amt'])
                            if order:
                                self.positions.remove(pos)
                                self.banked_pnl += pnl_usd
                                
                                emoji = "💰" if pnl_usd > 0 else "📉"
                                message = (
                                    f"{emoji} <b>POSITION CLOSED: {pos['symbol']}</b>\n"
                                    f"Reason: {exit_reason}\n"
                                    f"Entry: ${pos['entry']:.4f}\n"
                                    f"Exit: ${current_price:.4f}\n"
                                    f"P&L: ${pnl_usd:.2f} ({pnl_ratio*100:.2f}%)\n"
                                    f"Total Banked: ${self.banked_pnl:.2f}"
                                )
                                await self.send_telegram(message)
                    
                    except Exception as e:
                        self.logger.error(f"Position management error for {pos['symbol']}: {e}")
                        await asyncio.sleep(1)
                
                # Check position age (emergency exit after 1 hour)
                current_time = time.time()
                for pos in self.positions[:]:
                    if current_time - pos['timestamp'] > 3600:  # 1 hour
                        self.logger.warning(f"Emergency exit for {pos['symbol']} (1 hour limit)")
                        order = await self.place_order(pos['symbol'], 'sell', pos['amt'])
                        if order:
                            self.positions.remove(pos)
                
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Position manager error: {e}")
                await asyncio.sleep(10)

    async def run_health_check(self):
        """Periodic health checks and status updates"""
        while self.is_running:
            try:
                # Get account balance
                await self.rate_limit()
                balance = await self.exchange.fetch_balance()
                usdt_balance = balance.get('USDT', {}).get('free', 0)
                
                status_message = (
                    f"🤖 <b>Bot Status Update</b>\n"
                    f"Active Positions: {len(self.positions)}\n"
                    f"Banked P&L: ${self.banked_pnl:.2f}\n"
                    f"Available USDT: ${usdt_balance:.2f}\n"
                    f"Max Slots: {Config.MAX_SLOTS}"
                )
                await self.send_telegram(status_message)
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
            
            # Wait 1 hour between status updates
            await asyncio.sleep(3600)

    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        
        # Close all open positions
        if self.positions:
            await self.send_telegram("🛑 <b>Shutting down - Closing all positions</b>")
            
            for pos in self.positions:
                try:
                    await self.place_order(pos['symbol'], 'sell', pos['amt'])
                    await asyncio.sleep(0.5)
                except Exception as e:
                    self.logger.error(f"Shutdown error for {pos['symbol']}: {e}")
        
        # Close exchange connection
        await self.exchange.close()
        
        final_message = (
            f"🔴 <b>Bot Shutdown Complete</b>\n"
            f"Final Banked P&L: ${self.banked_pnl:.2f}\n"
            f"Positions Closed: {len(self.positions)}"
        )
        await self.send_telegram(final_message)

async def main():
    """Main execution function"""
    bot = AlphaSniper()
    
    try:
        # Run all tasks concurrently
        await asyncio.gather(
            bot.monitor_markets(),
            bot.manage_positions(),
            bot.run_health_check()
        )
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        bot.logger.error(f"Fatal error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    # Set up proper event loop for asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot terminated by user")
