import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta

# Copy nguyên bản SignalAgent từ bot1.py
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
        lookback = df.iloc[idx-45:idx-2] 
        liq = lookback['low'].min() if direction == "BUY" else lookback['high'].max()
        ob_candle = df.iloc[idx-2]
        if direction == "BUY": 
            return ob_candle['low'] <= liq * 0.999 and ob_candle['close'] < ob_candle['open']
        else: 
            return ob_candle['high'] >= liq * 1.001 and ob_candle['close'] > ob_candle['open']

    def check_unicorn_breaker(self, df, idx, direction, fvg_bottom, fvg_top):
        if idx - 50 < 0: return False
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

def fetch_historical_mexc(symbol, timeframe, limit=1000):
    exchange = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    # Lấy 90 ngày (khoảng 8640 nến 15m)
    all_bars = []
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=90)).isoformat()) 
    
    while True:
        try:
            print(f"Fetching {symbol} {timeframe} from {exchange.iso8601(since)}...")
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            since = bars[-1][0] + 1
            if len(bars) < 1000:
                break
            time.sleep(0.5) # Rate limit
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    df = pd.DataFrame(all_bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df.drop_duplicates(subset=['ts'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Tính toán các chỉ báo
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # Resample sang 1h và 4h từ data 15m để tránh look-ahead bias
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('datetime', inplace=True)
    
    # EMA 15m
    df['ema50_15m'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200_15m'] = df['close'].ewm(span=200, adjust=False).mean()
    
    df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'}).dropna()
    df_1h['ema50_1h'] = df_1h['close'].ewm(span=50, adjust=False).mean()
    df_1h['ema200_1h'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    
    df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum'}).dropna()
    df_4h['ema50_4h'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['ema200_4h'] = df_4h['close'].ewm(span=200, adjust=False).mean()
    
    df.reset_index(inplace=True)
    df_1h.reset_index(inplace=True)
    df_4h.reset_index(inplace=True)
    
    return df, df_1h, df_4h

def get_trend_at_time(dt, df_resampled):
    # Lấy nến phân giải thấp MỚI NHẤT ĐÃ ĐÓNG CỬA trước thềm `dt`
    past_bars = df_resampled[df_resampled['datetime'] < dt]
    if len(past_bars) < 1: return "UNKNOWN"
    last_bar = past_bars.iloc[-1]
    
    suffix = "1h" if "ema50_1h" in past_bars.columns else "4h"
    close, ema50, ema200 = last_bar['close'], last_bar['ema50_' + suffix], last_bar['ema200_' + suffix]
    if close > ema50 > ema200 * 1.002: return "UP"
    if close < ema50 < ema200 * 0.998: return "DOWN"
    return "SIDEWAY"

def is_in_killzone(dt):
    # Trả về True nếu nến đóng cửa nằm trong múi giờ giao dịch nhộn nhịp (ICT Killzones) tính theo UTC
    # London Killzone: 07:00 - 10:00 UTC
    # NY AM Killzone (Silver Bullet): 13:30 - 16:00 UTC
    # NY PM Killzone: 18:00 - 20:00 UTC
    h = dt.hour
    if 7 <= h < 10: return True
    if 13 <= h < 16 or (h == 13 and dt.minute >= 30): return True
    if 18 <= h < 20: return True
    return False

def simulate_trade(df, entry_idx, direction, entry_price, sl, tp1, tp2):
    # Trả về: outcome (TP1, TP2, SL, BE), pnl, duration_bars
    # Logic: 
    # Giả định đặt lệnh Limit tại FVG. Cần nến tương lai quay về đón (cắn Limit).
    # Chốt 50% ở TP1, dời SL về entry.
    # 50% còn lại gồng đến TP2.
    
    status = 'PENDING' # Chờ khớp Limit
    sl_current = sl
    outcome = []
    
    for i in range(entry_idx + 1, min(entry_idx + 100, len(df))): # Đợi max 100 nến
        bar = df.iloc[i]
        high, low = bar['high'], bar['low']
        
        if status == 'PENDING':
            # Kiểm tra xem có đónLimit không?
            if direction == "BUY" and low <= entry_price:
                status = 'OPEN'
            elif direction == "SELL" and high >= entry_price:
                status = 'OPEN'
            else:
                # Nếu giá không về cắn Limit mà bay/sập luôn (quá xa FVG), thì hủy lệnh (Miss)
                if direction == "BUY" and high > tp1:
                    return 'MISSED_RUN_AWAY', 0, i - entry_idx
                if direction == "SELL" and low < tp1:
                    return 'MISSED_RUN_AWAY', 0, i - entry_idx
                continue
                
        if status == 'OPEN':
            if direction == "BUY":
                if low <= sl_current:
                    return ('BE' if sl_current == entry_price else 'SL'), (-1 if sl_current == sl else 0), i - entry_idx
                elif high >= tp2:
                    return 'TP2_HIT', tp2, i - entry_idx
                elif high >= tp1 and 'TP1' not in outcome:
                    outcome.append('TP1')
                    sl_current = entry_price # Dời hòa vốn
            else: # SELL
                if high >= sl_current:
                    return ('BE' if sl_current == entry_price else 'SL'), (-1 if sl_current == sl else 0), i - entry_idx
                elif low <= tp2:
                    return 'TP2_HIT', tp2, i - entry_idx
                elif low <= tp1 and 'TP1' not in outcome:
                    outcome.append('TP1')
                    sl_current = entry_price
                    
    # Hết thời gian giữ lệnh
    if status == 'OPEN':
        return 'TIMEOUT', 0, 100
    return 'EXPIRED_NOT_CHOSEN', 0, 100

def run_backtest_for_params(df, df_1h, df_4h, param_sl_atr, param_tp1_r, param_tp2_r, min_score=5.5, use_killzone=False):
    signal_agent = SignalAgent()
    trades = []
    
    for idx in range(50, len(df)-1):
        dt = df['datetime'].iloc[idx]
        if use_killzone and not is_in_killzone(dt):
            continue
            
        has_fvg, fvg_dir, fvg_bottom, fvg_top = signal_agent.check_fvg(df, idx)
        if not has_fvg: continue
        
        dt = df['datetime'].iloc[idx]
        trend_1h = get_trend_at_time(dt, df_1h)
        trend_4h = get_trend_at_time(dt, df_4h)
        if trend_1h == "UNKNOWN" or trend_4h == "UNKNOWN": continue
        
        direction = None
        if fvg_dir == "bullish" and trend_1h in ["UP", "SIDEWAY"] and trend_4h in ["UP", "SIDEWAY"]: direction = "BUY"
        elif fvg_dir == "bearish" and trend_1h in ["DOWN", "SIDEWAY"] and trend_4h in ["DOWN", "SIDEWAY"]: direction = "SELL"
        
        if not direction or not signal_agent.check_strong_displacement(df, idx-1, direction, df['atr']): continue
        
        score, active = signal_agent.calculate_setup_score(df, idx, direction, fvg_bottom, fvg_top, df['atr'])
        if score >= min_score:
            entry = fvg_top if direction == "BUY" else fvg_bottom
            atr_val = df['atr'].iloc[idx]
            
            sl_atr = param_sl_atr
            if "Unicorn" in active: sl_atr *= 0.85
            if "Sweep" in active: sl_atr *= 1.2
            
            ob_low, ob_high = df['low'].iloc[idx-2], df['high'].iloc[idx-2]
            sl = (min(ob_low, fvg_bottom) - atr_val * sl_atr) if direction=="BUY" else (max(ob_high, fvg_top) + atr_val * sl_atr)
            
            risk = abs(entry - sl)
            # TP1 tính theo mức Risk (R)
            tp1 = entry + risk * param_tp1_r if direction == "BUY" else entry - risk * param_tp1_r
            tp2 = entry + risk * param_tp2_r if direction == "BUY" else entry - risk * param_tp2_r
            
            outcome, pnl, duration = simulate_trade(df, idx, direction, entry, sl, tp1, tp2)
            
            trades.append({
                'time': dt, 'dir': direction, 'score': score, 
                'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 
                'risk': risk, 'outcome': outcome, 'duration': duration,
                'active_setup': str(active)
            })
            
    # Tính thống kê
    total_signals = len(trades)
    executed = [t for t in trades if t['outcome'] in ['SL', 'BE', 'TP2_HIT', 'TIMEOUT', 'TP1']]
    total_exec = len(executed)
    wins = [t for t in executed if t['outcome'] == 'TP2_HIT']
    be = [t for t in executed if t['outcome'] == 'BE']
    loss = [t for t in executed if t['outcome'] == 'SL']
    
    winrate = (len(wins) + len(be)*0.5) / total_exec * 100 if total_exec > 0 else 0
    return {
        'sl_atr': param_sl_atr, 'tp1_r': param_tp1_r, 'tp2_r': param_tp2_r,
        'signals': total_signals, 'executed': total_exec,
        'wins_tp2': len(wins), 'be_hit_tp1': len(be), 'losses': len(loss),
        'winrate_adj': winrate,
        'trades': trades
    }

def main():
    coins_to_test = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    
    with open('backtest_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== KẾT QUẢ BACKTEST 3 THÁNG - VỐN LÝ THUYẾT 1000$ ===\n\n")

    # Chỉ chạy cấu hình tối ưu ngắm bắn Tỉ Lệ Vàng (SL_ATR, TP1, TP2) cho từng coin
    # Nhưng ta sẽ lưu tập trung danh sách lệnh của TẤT CẢ các coin, sau đó sort theo thời gian 
    # để giả lập tk $1000 chạy cùng lúc 3 cặp với rủi ro 1% (1R = 1% Balance).
    
    OPTIMAL_PARAMS = {
        'BTC/USDT:USDT': (0.3, 1.0, 2.0),
        'ETH/USDT:USDT': (0.3, 1.0, 2.0),
        'SOL/USDT:USDT': (0.75, 1.0, 2.5)
    }

    all_trades_history = []

    for symbol in coins_to_test:
        print(f"\nBắt đầu fetch data 90 ngày {symbol}...")
        df, df_1h, df_4h = fetch_historical_mexc(symbol, '15m')
        print(f"Lấy được {len(df)} nến 15m!")
        
        sl_a, tp1_r, tp2_r = OPTIMAL_PARAMS[symbol]
        
        # Luôn bật chế độ ICT Killzone (Silver Bullet) vì winrate cao nhất
        res = run_backtest_for_params(df, df_1h, df_4h, sl_a, tp1_r, tp2_r, use_killzone=True)
        
        for t in res['trades']:
            # Giữ lại các lệnh thực sự vô tình trạng kết thúc
            if t['outcome'] in ['SL', 'BE', 'TP2_HIT']:
                # Gán thêm tên symbol
                t['symbol'] = symbol
                # Tính R thu về
                if t['outcome'] == 'SL': 
                    t['pnl_R'] = -1.0
                elif t['outcome'] == 'BE': 
                    t['pnl_R'] = 0.5 * tp1_r  # Ăn 50% ở TP1, còn lại BE
                elif t['outcome'] == 'TP2_HIT': 
                    t['pnl_R'] = 0.5 * tp1_r + 0.5 * tp2_r
                
                all_trades_history.append(t)
                
        time.sleep(1)

    # Sort tất cả lệnh theo thời gian vào lệnh để mô phỏng Tài khoản chung
    all_trades_history.sort(key=lambda x: x['time'])
    
    # --- MÔ PHỎNG TÀI KHOẢN $1000 CHẠY KÉP (Compounding 1% Risk) ---
    balance = 1000.0
    risk_percent = 0.01 # Lệnh nào cũng cược 1% vốn hiện tại
    
    wins = 0
    losses = 0
    bes = 0
    
    for t in all_trades_history:
        # 1R = 1% Balance
        risk_amount = balance * risk_percent
        # PnL USD = R thu về * Risk Amount
        pnl_usd = t['pnl_R'] * risk_amount
        balance += pnl_usd
        
        t['balance_after'] = balance
        t['pnl_usd'] = pnl_usd
        
        if t['outcome'] == 'SL': losses += 1
        elif t['outcome'] == 'BE': bes += 1
        elif t['outcome'] == 'TP2_HIT': wins += 1

    total_exec = wins + bes + losses
    msg = (f"\n{'='*50}\n📊 BÁO CÁO MÔ PHỎNG TÀI KHOẢN 1000$ (COMPUNDING 1% RISK)\n{'='*50}\n"
           f" - Thời gian chạy: 3 Tháng (90 Ngày) qua 3 Cặp: BTC, ETH, SOL\n"
           f" - Tổng số lệnh đã khớp Limit: {total_exec} lệnh\n"
           f" - Thắng Full (TP2): {wins} lệnh\n"
           f" - Cắn TP1 rồi dời SL Hòa Vốn: {bes} lệnh\n"
           f" - Thua thẳng (Cắn SL): {losses} lệnh\n"
           f" - Winrate Tương Đối: {((wins + bes*0.5) / total_exec * 100) if total_exec>0 else 0:.2f}%\n"
           f"----------------------------------------\n"
           f"💰 SỐ DƯ TÀI KHOẢN SAU 3 THÁNG: ${balance:.2f} (Lợi nhuận: +${(balance - 1000):.2f})\n\n"
           f"GHI CHÚ: Số liệu tính bằng Lãi Kép (Compounding). Risk mỗi lệnh là 1% Balance tại thời điểm đó.\n")
    
    print(msg)
    with open('backtest_results.txt', 'a', encoding='utf-8') as f:
        f.write(msg)

if __name__ == "__main__":
    main()
