import pandas as pd
import numpy as np
import ccxt
import requests
import os
import time
from datetime import datetime

# ======================================================================
# --- 1. CẤU HÌNH BOT LIVE & TELEGRAM (GITHUB ACTIONS MODE) ---
# ======================================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

PAIRS = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
MIN_SCORE_EXECUTE = 4.5

# ĐÃ TỐI ƯU TP (TAKE PROFIT) HỢP LÝ HƠN ĐỂ DỄ CẮN FULL WIN
RISK_MATRIX = {
    'BTC/USDT:USDT': {
        '15m': {'sl_atr': 0.6, 'rr2': 2.5} 
    },
    'ETH/USDT:USDT': {
        '15m': {'sl_atr': 0.65, 'rr2': 3.0}
    }
}

exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

def send_telegram_message(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ LỖI: Chưa cài đặt TELEGRAM_TOKEN/CHAT_ID trong GitHub Secrets.")
        print("Nội dung tin nhắn nháp:\n", msg)
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Lỗi gửi Telegram: {e}")

# ======================================================================
# --- 2. SIGNAL AGENT (BỘ NÃO SMC ĐÃ VÁ SẠCH LỖI INDEX) ---
# ======================================================================
class SignalAgent:
    def __init__(self):
        self.displacement_ratio = 0.65
        self.min_displacement_atr = 1.15
        self.ote_low = 0.705
        self.ote_high = 0.79
        self.min_fvg_atr = 0.80

    def check_fvg(self, df, idx):
        if idx - 2 < 0 or idx >= len(df): return False, None, None, None
        low_p, high_p = df['low'].iloc[idx-2], df['high'].iloc[idx-2]
        low_f, high_f = df['low'].iloc[idx], df['high'].iloc[idx]
        if low_f > high_p: return True, "bullish", high_p, low_f
        if high_f < low_p: return True, "bearish", high_f, low_p
        return False, None, None, None

    def check_strong_displacement(self, df, idx, direction, atr_series):
        if idx >= len(df) or idx < 0: return False
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        if total == 0 or body < total * self.displacement_ratio: return False
        atr = atr_series.iloc[idx] if idx < len(atr_series) else 1
        if body < atr * self.min_displacement_atr: return False
        return (direction == "BUY" and candle['close'] > candle['open']) or \
               (direction == "SELL" and candle['close'] < candle['open'])

    def check_liquidity_sweep(self, df, idx, direction):
        if idx - 45 < 0: return False
        # Dừng quét lookback ở idx-3 để không tự so với OB
        lookback = df.iloc[idx-45:idx-2] 
        liq = lookback['low'].min() if direction == "BUY" else lookback['high'].max()
        
        # Check đúng nến OB (idx-2) thay vì nến đẩy (idx-1)
        ob_candle = df.iloc[idx-2]
        if direction == "BUY": 
            return ob_candle['low'] <= liq * 0.999 and ob_candle['close'] < ob_candle['open']
        else: 
            return ob_candle['high'] >= liq * 1.001 and ob_candle['close'] > ob_candle['open']

    def check_unicorn_breaker(self, df, idx, direction, fvg_bottom, fvg_top):
        if idx - 50 < 0: return False
        # Không lấy dính vào nến OB hiện tại
        lookback = df.iloc[idx-50:idx-2] 
        for i in range(len(lookback)-1, max(0, len(lookback)-25), -1):
            c = lookback.iloc[i]
            if direction == "BUY" and c['close'] < c['open']:
                if fvg_bottom - 0.3*(fvg_top-fvg_bottom) <= c['high'] <= fvg_top: return True
            elif direction == "SELL" and c['close'] > c['open']:
                if fvg_bottom <= c['low'] <= fvg_top + 0.3*(fvg_top-fvg_bottom): return True
        return False

    def calculate_ote_score(self, df, idx, direction, fvg_bottom, fvg_top):
        if idx - 25 < 0: return 0
        imp_low = df['low'].iloc[idx-25:idx+1].min()
        imp_high = df['high'].iloc[idx-25:idx+1].max()
        rng = imp_high - imp_low
        if rng <= 0: return 0
        if direction == "BUY":
            ote_t, ote_b = imp_high - rng * self.ote_low, imp_high - rng * self.ote_high
            return 3 if fvg_top >= ote_b and fvg_bottom <= ote_t else 0
        else:
            ote_b, ote_t = imp_low + rng * self.ote_low, imp_low + rng * self.ote_high
            return 3 if fvg_bottom <= ote_t and fvg_top >= ote_b else 0

    def calculate_setup_score(self, df, idx, direction, fvg_bottom, fvg_top, atr_series):
        score = 0.0
        active_setups = []
        if self.check_liquidity_sweep(df, idx, direction): score += 4.0; active_setups.append("Sweep")
        if self.check_unicorn_breaker(df, idx, direction, fvg_bottom, fvg_top): score += 3.5; active_setups.append("Unicorn")
        if self.calculate_ote_score(df, idx, direction, fvg_bottom, fvg_top) > 0: score += 3.0; active_setups.append("OTE")
        
        if idx >= 30:
            vol_avg = df['vol'].iloc[idx-30:idx-1].mean()
            if df['vol'].iloc[idx-1] > vol_avg * 2.3: score += 2.5; active_setups.append("Momentum")
            elif df['vol'].iloc[idx-1] > vol_avg * 1.55: score += 1.5
            
        if self.check_strong_displacement(df, idx-1, direction, atr_series): score += 1.5
        return round(score, 1), active_setups

# ======================================================================
# --- 3. LIVE EXECUTION LOGIC (TÍCH HỢP QUÉT NẾN AGE & CHỐNG SPAM) ---
# ======================================================================
def fetch_live_data(symbol, timeframe, limit=300):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        return df
    except Exception as e:
        print(f"Lỗi kéo data {symbol} {timeframe}: {e}")
        return None

def main():
    print(f"[{datetime.now()}] 🚀 Kích hoạt quy trình quét thị trường MEXC...")
    
    # Chờ API nhả nến mới nếu GitHub Actions chạy trúng khoảnh khắc chuyển nến
    time.sleep(5) 
    
    signal_agent = SignalAgent()
    current_time = datetime.now()

    for symbol in PAIRS:
        df_15m = fetch_live_data(symbol, '15m')
        df_1h = fetch_live_data(symbol, '1h')
        df_4h = fetch_live_data(symbol, '4h')
        
        if df_15m is None or df_1h is None or df_4h is None: continue
        
        # SỬ DỤNG TRẠNG THÁI HIỆN TẠI (iloc[-1]) CỦA KHUNG LỚN ĐỂ KHÔNG TRỄ NHỊP
        trend_1h = "UP" if df_1h['close'].iloc[-1] > df_1h['ema50'].iloc[-1] > df_1h['ema200'].iloc[-1] * 1.002 else \
                   "DOWN" if df_1h['close'].iloc[-1] < df_1h['ema50'].iloc[-1] < df_1h['ema200'].iloc[-1] * 0.998 else "SIDEWAY"
                   
        trend_4h = "UP" if df_4h['close'].iloc[-1] > df_4h['ema50'].iloc[-1] > df_4h['ema200'].iloc[-1] * 1.002 else \
                   "DOWN" if df_4h['close'].iloc[-1] < df_4h['ema50'].iloc[-1] < df_4h['ema200'].iloc[-1] * 0.998 else "SIDEWAY"
        
        if trend_1h == "UNKNOWN" or trend_4h == "UNKNOWN": continue

        print(f"\n🔍 Đang soi {symbol} (Trend 1H: {trend_1h}, 4H: {trend_4h})")
        alerts_found = 0

        # QUÉT 4 NẾN GẦN NHẤT (-5 đến -2) ĐỂ TÌM SETUP & CHECK AGE
        for idx in range(len(df_15m) - 5, len(df_15m) - 1):
            
            # Tính tuổi của nến (Age)
            candle_ts = df_15m['ts'].iloc[idx]
            candle_time = datetime.fromtimestamp(candle_ts / 1000)
            age_minutes = (current_time - candle_time).total_seconds() / 60.0

            has_fvg, fvg_dir, fvg_bottom, fvg_top = signal_agent.check_fvg(df_15m, idx)
            if not has_fvg: continue

            direction = None
            if fvg_dir == "bullish" and trend_1h in ["UP", "SIDEWAY"] and trend_4h in ["UP", "SIDEWAY"]: direction = "BUY"
            elif fvg_dir == "bearish" and trend_1h in ["DOWN", "SIDEWAY"] and trend_4h in ["DOWN", "SIDEWAY"]: direction = "SELL"
            
            if not direction or not signal_agent.check_strong_displacement(df_15m, idx-1, direction, df_15m['atr']): continue

            score, active = signal_agent.calculate_setup_score(df_15m, idx, direction, fvg_bottom, fvg_top, df_15m['atr'])
            
            if score >= MIN_SCORE_EXECUTE:
                alerts_found += 1
                entry = fvg_top if direction == "BUY" else fvg_bottom
                atr_val = df_15m['atr'].iloc[idx]
                
                base_risk = RISK_MATRIX[symbol]['15m']
                sl_atr, rr2 = base_risk['sl_atr'], base_risk['rr2']
                if "Unicorn" in active: sl_atr *= 0.85; rr2 *= 1.15
                if "Sweep" in active: sl_atr *= 1.2
                
                ob_low, ob_high = df_15m['low'].iloc[idx-2], df_15m['high'].iloc[idx-2]
                sl = (min(ob_low, fvg_bottom) - atr_val * sl_atr) if direction=="BUY" else (max(ob_high, fvg_top) + atr_val * sl_atr)
                
                risk = abs(entry - sl)
                tp1 = entry + risk if direction == "BUY" else entry - risk
                tp2 = entry + risk * rr2 if direction == "BUY" else entry - risk * rr2
                
                # CHỈ BÁO TELEGRAM KHI NẾN CHƯA QUÁ 25 PHÚT (Chống báo lặp)
                if age_minutes <= 25:
                    emoji = "🟢" if direction == "BUY" else "🔴"
                    msg = (
                        f"🚨 <b>SMC SETUP (Score: {score})</b> 🚨\n"
                        f"⏰ <i>Nến tạo lúc: {candle_time.strftime('%H:%M')} (Cách đây {age_minutes:.0f} phút)</i>\n\n"
                        f"🪙 <b>Cặp:</b> {symbol} (15m MEXC)\n"
                        f"{emoji} <b>Hướng:</b> {direction} Limit\n"
                        f"🎯 <b>Entry Limit:</b> <code>{entry:.4f}</code>\n"
                        f"🛑 <b>Stoploss:</b> <code>{sl:.4f}</code>\n\n"
                        f"💰 <b>Kế hoạch:</b>\n"
                        f"1️⃣ <b>TP1 (1R - Chốt 50% & BE):</b> <code>{tp1:.4f}</code>\n"
                        f"2️⃣ <b>TP2 (Đích):</b> <code>{tp2:.4f}</code>\n\n"
                        f"<i>(Lưu ý: Hủy lệnh nếu giá xuyên qua mức {'{:.4f}'.format(ob_low) if direction=='BUY' else '{:.4f}'.format(ob_high)})</i>"
                    )
                    send_telegram_message(msg)
                    print(f" -> 📲 ĐÃ BÁO TELEGRAM: Setup {direction} lúc {candle_time.strftime('%H:%M')} (Age: {age_minutes:.1f}p)")
                else:
                    # Tín hiệu đã cũ thì in ra màn hình để đối chiếu, không báo Telegram làm phiền
                    print(f" -> 🔇 BỎ QUA (NẾN CŨ): Setup {direction} lúc {candle_time.strftime('%H:%M')} (Age: {age_minutes:.1f}p)")

        if alerts_found == 0:
            print(f" -> 📭 Không có setup SMC nào đủ điều kiện ở các nến gần đây.")

    print("\n✅ Quét hoàn tất!")

if __name__ == "__main__":
    main()
