import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import feedparser
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import json
import asyncio

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tinhieu_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load biến môi trường
load_dotenv()

# Lớp xử lý webhook
class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            update = Update.de_json(json.loads(post_data), app.bot)
            app.process_update(update)
            self.send_response(200)
            self.end_headers()
        except Exception as e:
            logger.error(f"Webhook handler error: {str(e)}")
            self.send_response(500)
            self.end_headers()

# Hàm chạy server webhook
def run_webhook_server():
    default_port = 8443
    port_str = os.getenv('PORT', str(default_port))
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid port number: {port}")
    except ValueError as e:
        logger.error(f"Invalid PORT value: {port_str}. Using default port {default_port}")
        port = default_port
    
    server = HTTPServer(('0.0.0.0', port), WebhookHandler)
    logger.info(f"Starting webhook server on port {port}...")
    server.serve_forever()

# Lớp phân tích dữ liệu từ OKX
class OKXAnalyzer:
    def __init__(self):
        self.base_url = "https://www.okx.com/api/v5/market/candles"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
        self.timeout = 20
        self.timeframe_map = {
            '15m': '15m',
            '1h': '1H',
            '4h': '4H'
        }

    def get_btc_data(self, timeframe='1H', limit=100):
        """Lấy dữ liệu BTC từ OKX"""
        try:
            timeframe = self.timeframe_map.get(timeframe, '1H')
            params = {
                'instId': 'BTC-USDT',
                'bar': timeframe,
                'limit': limit
            }
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"HTTP Error {response.status_code}: {response.text}")
                return None
                
            data = response.json()
            
            if data.get('code') != '0':
                logger.error(f"OKX API Error: {data.get('msg', 'Unknown error')}")
                return None
                
            if not data.get('data'):
                logger.error("Empty data response")
                return None
                
            # Xử lý dữ liệu
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 
                'vol', 'volCcy', 'volCcyQuote', 'confirm'
            ]
            df = pd.DataFrame(data['data'], columns=columns)
            
            # Chọn các cột cần thiết
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'vol']]
            numeric_cols = ['open', 'high', 'low', 'close', 'vol']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
            
            return df.iloc[::-1]
            
        except Exception as e:
            logger.error(f"Error in get_btc_data: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Tính toán các chỉ báo kỹ thuật"""
        try:
            # Tính MA
            df['MA25'] = df['close'].rolling(window=25).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            
            # Tính RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Tính Bollinger Bands Width (BBW)
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['STD20'] = df['close'].rolling(window=20).std()
            df['UpperBand'] = df['MA20'] + (2 * df['STD20'])
            df['LowerBand'] = df['MA20'] - (2 * df['STD20'])
            df['BBW'] = (df['UpperBand'] - df['LowerBand']) / df['MA20']
            
            # Tính Volume trung bình
            df['AvgVol'] = df['vol'].rolling(window=20).mean()
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Indicator calculation error: {str(e)}")
            return None

    def detect_support_resistance(self, df, window=100):
        """Phát hiện kháng cự/hỗ trợ trong 100 phiên gần nhất"""
        try:
            df = df.tail(window)
            resistance = df['high'].max()
            support = df['low'].min()
            current_price = df.iloc[-1]['close']
            
            # Ngưỡng chạm (1% giá)
            threshold = current_price * 0.01
            
            if abs(current_price - resistance) < threshold:
                return "resistance"
            elif abs(current_price - support) < threshold:
                return "support"
            return None
        except Exception as e:
            logger.error(f"Support/resistance detection error: {str(e)}")
            return None

# Lớp lấy tin tức thị trường
class MarketNews:
    def __init__(self):
        self.news_sources = {
            'Coin68': 'https://coin68.com/feed/',
            'TapChiBitcoin': 'https://tapchibitcoin.io/feed',
            'BitcoinVN': 'https://bitcoinvn.io/feed/'
        }
    
    def get_latest_news(self, limit=3):
        """Lấy tin tức mới nhất từ các nguồn"""
        news_items = []
        for source, url in self.news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit]:
                    news_items.append({
                        'source': source,
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.published
                    })
            except Exception as e:
                logger.error(f"Error fetching news from {source}: {str(e)}")
        
        # Sắp xếp theo thời gian mới nhất
        news_items.sort(key=lambda x: x['published'], reverse=True)
        return news_items[:limit]

# Lớp chính của bot
class TinhieuBTCBot:
    def __init__(self, application):
        self.application = application
        self.bot = application.bot
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TINHIEU_CHAT_ID')
        self.okx = OKXAnalyzer()
        self.news = MarketNews()
        self.current_recommendation = "Đang phân tích..."
        self.last_alert_time = None
        self.last_update_time = None
        
        if not self.token or not self.chat_id:
            raise ValueError("Chưa cấu hình TELEGRAM_BOT_TOKEN hoặc TINHIEU_CHAT_ID")

    async def analyze_market(self, context: ContextTypes.DEFAULT_TYPE):
        """Phân tích thị trường tự động"""
        try:
            # Phân tích đa khung thời gian
            timeframes = ['15m', '1H', '4H']
            signals = []
            sr_alerts = []
            
            for tf in timeframes:
                df = self.okx.get_btc_data(timeframe=tf, limit=100)
                if df is not None:
                    df = self.okx.calculate_indicators(df)
                    latest = df.iloc[-1]
                    
                    # Tín hiệu MA
                    ma_signal = "MA25 > MA50" if latest['MA25'] > latest['MA50'] else "MA25 < MA50"
                    
                    # Tín hiệu Volume
                    vol_signal = "Volume tăng mạnh" if latest['vol'] > 2 * latest['AvgVol'] else "Volume bình thường"
                    
                    # Tín hiệu BBW
                    bbw_signal = "BBW mở rộng" if latest['BBW'] > df['BBW'].mean() else "BBW thu hẹp"
                    
                    # Tín hiệu RSI
                    rsi_signal = latest['RSI']
                    
                    signals.append({
                        'timeframe': tf,
                        'price': latest['close'],
                        'ma_signal': ma_signal,
                        'vol_signal': vol_signal,
                        'bbw_signal': bbw_signal,
                        'rsi': rsi_signal
                    })
                    
                    # Kiểm tra kháng cự/hỗ trợ
                    sr = self.okx.detect_support_resistance(df)
                    if sr:
                        sr_alerts.append(f"⚠️ Giá chạm {sr} ở khung {tf}")
            
            # Tạo khuyến nghị tổng hợp
            recommendation = self.generate_recommendation(signals)
            self.current_recommendation = recommendation
            self.last_update_time = datetime.now()
            
            # Gửi thông báo
            await self.send_analysis(context, signals, recommendation, sr_alerts)
            
            # Lên lịch phân tích tiếp sau 15 phút
            context.job_queue.run_once(
                self.analyze_market, 
                when=timedelta(minutes=15)
            )
            
        except Exception as e:
            logger.error(f"Auto analysis error: {str(e)}")

    def generate_recommendation(self, signals):
        """Tạo khuyến nghị dựa trên tín hiệu"""
        buy_signals = 0
        sell_signals = 0
        
        for signal in signals:
            # Điểm cho tín hiệu MA
            if signal['ma_signal'] == "MA25 > MA50":
                buy_signals += 1
            else:
                sell_signals += 1
                
            # Điểm cho tín hiệu Volume
            if signal['vol_signal'] == "Volume tăng mạnh":
                if signal['ma_signal'] == "MA25 > MA50":
                    buy_signals += 1
                else:
                    sell_signals += 1
                    
            # Điểm cho tín hiệu BBW
            if signal['bbw_signal'] == "BBW mở rộng":
                if signal['rsi'] < 35:
                    buy_signals += 1
                elif signal['rsi'] > 85:
                    sell_signals += 1
                    
            # Điểm cho tín hiệu RSI
            if signal['rsi'] < 35:
                buy_signals += 1
            elif signal['rsi'] > 85:
                sell_signals += 1
        
        if buy_signals >= 8:
            return "MUA MẠNH 🟢"
        elif sell_signals >= 8:
            return "BÁN MẠNH 🔴"
        elif buy_signals > sell_signals:
            return "CÓ THỂ MUA 🟡"
        elif sell_signals > buy_signals:
            return "CÓ THỂ BÁN 🟠"
        else:
            return "CHỜ TÍN HIỆU ⚪"

    async def send_analysis(self, context: ContextTypes.DEFAULT_TYPE, signals, recommendation, sr_alerts):
        """Gửi phân tích lên nhóm"""
        try:
            # Tạo nút cập nhật
            keyboard = [
                [InlineKeyboardButton("🔄 Cập nhật ngay", callback_data='update_analysis')],
                [InlineKeyboardButton("📰 Tin tức mới nhất", callback_data='get_news')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Tạo message
            analysis_text = "📊 <b>PHÂN TÍCH BITCOIN TỰ ĐỘNG</b>\n\n"
            analysis_text += f"⏰ <i>Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>\n\n"
            
            for signal in signals:
                timeframe_icon = "⏳" if signal['timeframe'] == '15m' else "🕒" if signal['timeframe'] == '1H' else "⏱️"
                rsi_color = "🟢" if signal['rsi'] < 30 else "🔴" if signal['rsi'] > 70 else "🟡"
                
                analysis_text += (
                    f"{timeframe_icon} <b>Khung {signal['timeframe'].upper()}</b>\n"
                    f"💰 <b>Giá:</b> ${signal['price']:,.2f}\n"
                    f"📈 <b>MA:</b> {signal['ma_signal'].replace('>', '→').replace('<', '←')}\n"
                    f"📊 <b>Volume:</b> {signal['vol_signal']}\n"
                    f"📉 <b>BBW:</b> {signal['bbw_signal']}\n"
                    f"🧮 <b>RSI:</b> {rsi_color} {signal['rsi']:.1f}\n\n"
                )
            
            rec_color = {
                "MUA MẠNH 🟢": "🟢",
                "BÁN MẠNH 🔴": "🔴",
                "CÓ THỂ MUA 🟡": "🟡",
                "CÓ THỂ BÁN 🟠": "🟠",
                "CHỜ TÍN HIỆU ⚪": "⚪"
            }.get(recommendation, "⚪")
            
            analysis_text += (
                f"🎯 <b>KHuyến nghị:</b> {rec_color} <b>{recommendation}</b>\n\n"
            )
            
            if sr_alerts and (self.last_alert_time is None or (datetime.now() - self.last_alert_time).total_seconds() > 3600):
                analysis_text += "\n".join(sr_alerts) + "\n"
                self.last_alert_time = datetime.now()
            
            await context.bot.send_message(
                chat_id=self.chat_id,
                text=analysis_text,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Send analysis error: {str(e)}")

    async def handle_analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Xử lý lệnh /analyze"""
        try:
            keyboard = [
                [InlineKeyboardButton("🔄 Cập nhật ngay", callback_data='update_analysis')],
                [InlineKeyboardButton("📰 Tin tức mới nhất", callback_data='get_news')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            last_update = f"\n⏰ Cập nhật lần cuối: {self.last_update_time.strftime('%d/%m/%Y %H:%M')}" if self.last_update_time else ""
            
            await update.message.reply_text(
                f"🔍 <b>Khuyến nghị hiện tại:</b>\n\n{self.current_recommendation}{last_update}",
                parse_mode='HTML',
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Analyze command error: {str(e)}")
            await update.message.reply_text("⚠️ Đã xảy ra lỗi. Vui lòng thử lại sau.")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Xử lý nút bấm"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'update_analysis':
            await query.edit_message_text(text="🔄 Đang cập nhật dữ liệu mới nhất...")
            await self.analyze_market(context)
        elif query.data == 'get_news':
            await self.send_latest_news(update, context)
        elif query.data == 'back_to_analysis':
            await self.handle_analyze_command(update, context)

    async def send_latest_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gửi tin tức mới nhất"""
        try:
            news_items = self.news.get_latest_news()
            if not news_items:
                await context.bot.send_message(
                    chat_id=self.chat_id,
                    text="⚠️ Không thể lấy tin tức mới nhất. Vui lòng thử lại sau."
                )
                return
            
            news_text = "📰 <b>TIN TỨC BITCOIN MỚI NHẤT</b>\n\n"
            for item in news_items:
                news_text += (
                    f"📌 <b>{item['source']}</b>\n"
                    f"🔹 {item['title']}\n"
                    f"🔗 <a href='{item['link']}'>Đọc thêm</a>\n"
                    f"⏰ {item['published']}\n\n"
                )
            
            keyboard = [[InlineKeyboardButton("↩️ Quay lại", callback_data='back_to_analysis')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    text=news_text,
                    parse_mode='HTML',
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
            else:
                await context.bot.send_message(
                    chat_id=self.chat_id,
                    text=news_text,
                    parse_mode='HTML',
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
                
        except Exception as e:
            logger.error(f"Send news error: {str(e)}")
            await context.bot.send_message(
                chat_id=self.chat_id,
                text="⚠️ Đã xảy ra lỗi khi lấy tin tức. Vui lòng thử lại sau."
            )

# Hàm xử lý lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /start"""
    keyboard = [
        [InlineKeyboardButton("🔍 Phân tích ngay", callback_data='update_analysis')],
        [InlineKeyboardButton("📰 Tin tức mới nhất", callback_data='get_news')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 <b>Bot Phân Tích Bitcoin Nâng Cao</b>\n\n"
        "Tự động phân tích mỗi 15 phút với:\n"
        "✅ MA25/MA50\n"
        "✅ Volume giao dịch\n"
        "✅ Bollinger Bands Width\n"
        "✅ RSI\n"
        "✅ Kháng cự/hỗ trợ\n\n"
        "Gõ /analyze để xem khuyến nghị hiện tại\n"
        "hoặc nhấn nút bên dưới để tương tác ngay!",
        parse_mode='HTML',
        reply_markup=reply_markup
    )

# Hàm chính
async def main():
    """Hàm chính khởi chạy bot"""
    global app
    try:
        # Khởi tạo ứng dụng
        application = Application.builder().token(os.getenv('TELEGRAM_BOT_TOKEN')).build()
        app = application
        bot = TinhieuBTCBot(application)
        
        # Thêm các handler
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analyze", bot.handle_analyze_command))
        application.add_handler(CallbackQueryHandler(bot.button_handler))
        
        # Lên lịch phân tích đầu tiên
        application.job_queue.run_once(bot.analyze_market, when=3)
        
        # Thiết lập webhook
        webhook_url = os.getenv('WEBHOOK_URL')
        default_port = 8443
        port_str = os.getenv('PORT', str(default_port))
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")
        except ValueError as e:
            logger.error(f"Invalid PORT value: {port_str}. Using default port {default_port}")
            port = default_port
        
        # Khởi động server webhook trong luồng riêng
        threading.Thread(target=run_webhook_server, daemon=True).start()
        
        if webhook_url:
            # Đợi một chút để server webhook khởi động
            await asyncio.sleep(1)
            # Thiết lập webhook
            await application.bot.set_webhook(f"{webhook_url}/webhook")
            logger.info(f"Webhook set to {webhook_url}/webhook")
            
            # Khởi động ứng dụng với webhook
            await application.initialize()
            await application.start()
            await application.updater.start_webhook(
                listen='0.0.0.0',
                port=port,
                url_path='/webhook',
                webhook_url=f"{webhook_url}/webhook"
            )
            logger.info("🤖 Bot đang chạy với webhook...")
            
            # Giữ ứng dụng chạy
            while True:
                await asyncio.sleep(3600)  # Ngủ 1 giờ để giữ luồng chính chạy
        else:
            logger.warning("WEBHOOK_URL not set, falling back to polling")
            await application.run_polling()
        
    except Exception as e:
        logger.error(f"Lỗi khởi động: {str(e)}")
        raise
    finally:
        if 'application' in locals():
            await application.stop()
            await application.shutdown()

if __name__ == '__main__':
    # Kiểm tra kết nối OKX
    print("🔄 Kiểm tra kết nối...")
    analyzer = OKXAnalyzer()
    if analyzer.get_btc_data(limit=1) is not None:
        print("✅ Kết nối OKX ổn định!")
        asyncio.run(main())
    else:
        print("❌ Lỗi kết nối OKX")