#!/usr/bin/env python3
"""
================================================================================
    [02] Breakout Range 8 Candles RR2.0
    NAS100 M5 — Michele Piazzoli — Quant Strategy
================================================================================

Standalone backtest script for the #2 ranked strategy.

Strategy:
    - Trigger:      Breakout Range 8 Candles (close breaks above/below 8-bar high/low)
    - HTF Filter:   H1 SMA 20
    - Session:      US Early (13:30 - 18:00)
    - SL:           ATR(14) x 1.0
    - RR:           1:2.0
    - Monthly Cap:  10%

Risk Management (DRS):
    - Initial: $100,000 | Floor: $90,000
    - Risk = (Balance - 90,000) / 20  [$100 - $3,500]

Output:
    - [01] PDF Report/report.html
    - [02] CSV Trade list/trades.csv
    - [03] Report TXT/report.txt
    - [04] Python code/backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
#  CONFIGURATION
# =============================================================================

DATA_PATH       = Path("/Users/capitalpro/Desktop/NAS100/08_M5/1.csv")
OUTPUT_BASE     = Path("/Users/capitalpro/Desktop/Strategia 1/[02] Breakout Range 8 Candles RR2.0")

BACKTEST_START  = "2024-01-01"
BACKTEST_END    = "2026-02-01"

INITIAL_BALANCE = 100_000
DD_FLOOR        = 90_000
RISK_DIVISOR    = 20
MIN_RISK        = 100
MAX_RISK        = 3_500

ATR_PERIOD      = 14
MONTHLY_CAP_PCT = 10.0

# Strategy parameters
BREAKOUT_PERIOD = 8        # Lookback candles for range high/low
HTF_SMA_PERIOD  = 20       # H1 SMA period for bias
SESS_START      = "13:30"  # US Early
SESS_END        = "18:00"
SESS_NAME       = "US_Early"
RR              = 2.0
SL_ATR_MULT     = 1.0

STRATEGY_NAME   = "Breakout Range 8 Candles RR2.0"
ASSET           = "NAS100"
TIMEFRAME       = "M5"


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    time_str = df['time'].astype(str).str[:19]
    df['time'] = pd.to_datetime(time_str, format='mixed')
    df = df[['time', 'open', 'high', 'low', 'close']].copy()
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df = df.sort_values('time').reset_index(drop=True)
    return df


def filter_date_range(df, start, end):
    mask = (df['time'] >= pd.to_datetime(start)) & (df['time'] < pd.to_datetime(end))
    return df[mask].reset_index(drop=True)


# =============================================================================
#  INDICATORS
# =============================================================================

def calc_sma(s, p):
    return s.rolling(window=p, min_periods=p).mean()

def calc_atr(df, p):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def build_h1_bias(df_m5, sma_period):
    df = df_m5.set_index('time').copy()
    h1 = df['close'].resample('1h').last().dropna().to_frame()
    h1['h1_sma'] = calc_sma(h1['close'], sma_period)
    h1['htf_bias'] = np.where(h1['close'] > h1['h1_sma'], 'LONG',
                     np.where(h1['close'] < h1['h1_sma'], 'SHORT', 'NEUTRAL'))
    h1 = h1[['htf_bias']].reset_index()
    h1.columns = ['time', 'htf_bias']
    merged = pd.merge_asof(df_m5.sort_values('time'), h1, on='time', direction='backward')
    merged['htf_bias'] = merged['htf_bias'].fillna('NEUTRAL')
    return merged


# =============================================================================
#  SIGNAL GENERATION — Breakout Range
# =============================================================================

def generate_signals(df, period):
    df = df.copy()
    rh = df['high'].rolling(window=period).max().shift(1)
    rl = df['low'].rolling(window=period).min().shift(1)
    df['sig_long']  = df['close'] > rh
    df['sig_short'] = df['close'] < rl
    df['sig_long']  = df['sig_long'].fillna(False)
    df['sig_short'] = df['sig_short'].fillna(False)
    return df


# =============================================================================
#  DRS
# =============================================================================

def calc_risk(balance):
    margin = balance - DD_FLOOR
    if margin <= 0:
        return MIN_RISK
    return max(MIN_RISK, min(MAX_RISK, margin / RISK_DIVISOR))


# =============================================================================
#  BACKTEST ENGINE
# =============================================================================

def run_backtest(df):
    df = df.copy()
    df = generate_signals(df, BREAKOUT_PERIOD)
    df['atr'] = calc_atr(df, ATR_PERIOD)
    df = build_h1_bias(df, HTF_SMA_PERIOD)

    s_h, s_m = map(int, SESS_START.split(':'))
    e_h, e_m = map(int, SESS_END.split(':'))
    s_min = s_h * 60 + s_m
    e_min = e_h * 60 + e_m

    df['t_min'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['wd'] = df['time'].dt.dayofweek
    df['in_sess'] = (df['t_min'] >= s_min) & (df['t_min'] < e_min) & (df['wd'] < 5)

    df['go_long']  = df['sig_long']  & df['in_sess'] & (df['htf_bias'] == 'LONG')  & (df['atr'] > 0)
    df['go_short'] = df['sig_short'] & df['in_sess'] & (df['htf_bias'] == 'SHORT') & (df['atr'] > 0)

    times  = df['time'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    atrs   = df['atr'].values
    gl     = df['go_long'].values
    gs     = df['go_short'].values
    months = df['time'].dt.to_period('M').astype(str).values
    n      = len(df)

    trades = []
    balance = INITIAL_BALANCE
    monthly_pnl = {}
    i = 0

    while i < n:
        if not gl[i] and not gs[i]:
            i += 1
            continue

        cm = months[i]
        if cm not in monthly_pnl:
            monthly_pnl[cm] = {'pnl': 0, 'start_bal': balance}
        md = monthly_pnl[cm]
        mr = md['pnl'] / md['start_bal'] * 100 if md['start_bal'] > 0 else 0
        if mr >= MONTHLY_CAP_PCT:
            i += 1
            continue

        sl_dist = atrs[i] * SL_ATR_MULT
        tp_dist = sl_dist * RR
        if sl_dist <= 0:
            i += 1
            continue

        risk = calc_risk(balance)
        bal_pre = balance

        if gl[i]:
            d = 'LONG'
            ep = closes[i]
            sl = ep - sl_dist
            tp = ep + tp_dist
        else:
            d = 'SHORT'
            ep = closes[i]
            sl = ep + sl_dist
            tp = ep - tp_dist

        et = pd.Timestamp(times[i])

        j = i + 1
        res = None
        xp = None
        xt = None

        while j < n:
            if d == 'LONG':
                if lows[j] <= sl:
                    res, xp, xt = 'SL', sl, pd.Timestamp(times[j])
                    break
                elif highs[j] >= tp:
                    res, xp, xt = 'TP', tp, pd.Timestamp(times[j])
                    break
            else:
                if highs[j] >= sl:
                    res, xp, xt = 'SL', sl, pd.Timestamp(times[j])
                    break
                elif lows[j] <= tp:
                    res, xp, xt = 'TP', tp, pd.Timestamp(times[j])
                    break
            j += 1

        if res is None:
            i = n
            continue

        pnl = risk * RR if res == 'TP' else -risk
        balance += pnl

        em = xt.strftime('%Y-%m')
        if em not in monthly_pnl:
            monthly_pnl[em] = {'pnl': 0, 'start_bal': bal_pre}
        monthly_pnl[em]['pnl'] += pnl

        trades.append({
            'entry_time': et, 'exit_time': xt, 'direction': d,
            'entry_price': round(ep, 2), 'sl_price': round(sl, 2),
            'tp_price': round(tp, 2), 'exit_price': round(xp, 2),
            'result': res, 'risk': round(risk, 2),
            'pnl': round(pnl, 2),
            'balance_pre': round(bal_pre, 2),
            'balance_post': round(balance, 2),
        })

        i = j + 1

    return trades, monthly_pnl


# =============================================================================
#  METRICS
# =============================================================================

def compute_metrics(trades, monthly_pnl):
    if not trades:
        return None

    total = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'TP')
    losses = total - wins
    wr = wins / total * 100

    total_pnl = sum(t['pnl'] for t in trades)
    final_bal = INITIAL_BALANCE + total_pnl
    total_ret = total_pnl / INITIAL_BALANCE * 100

    eq = INITIAL_BALANCE
    peak = eq
    max_dd = 0
    max_dd_pct = 0
    for t in trades:
        eq += t['pnl']
        if eq > peak: peak = eq
        dd = peak - eq
        dp = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd; max_dd_pct = dp

    gp = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gp / gl if gl > 0 else 0

    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losses > 0 else 0

    monthly = {}
    eq = INITIAL_BALANCE
    for t in trades:
        m = t['entry_time'].strftime('%Y-%m')
        if m not in monthly:
            monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0, 'start_bal': eq, 'max_dd': 0, 'peak': eq}
        monthly[m]['trades'] += 1
        if t['result'] == 'TP': monthly[m]['wins'] += 1
        monthly[m]['pnl'] += t['pnl']
        eq += t['pnl']
        if eq > monthly[m]['peak']: monthly[m]['peak'] = eq
        dd = monthly[m]['peak'] - eq
        if dd > monthly[m]['max_dd']: monthly[m]['max_dd'] = dd

    for m in monthly:
        d = monthly[m]
        d['win_rate'] = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
        d['return_pct'] = d['pnl'] / d['start_bal'] * 100 if d['start_bal'] > 0 else 0
        d['dd_pct'] = d['max_dd'] / d['peak'] * 100 if d['peak'] > 0 else 0
        d['end_bal'] = d['start_bal'] + d['pnl']

    months_total = len(monthly_pnl)
    months_positive = sum(1 for d in monthly_pnl.values() if d['pnl'] > 0)
    months_at_cap = 0
    for d in monthly_pnl.values():
        mr = d['pnl'] / d['start_bal'] * 100 if d['start_bal'] > 0 else 0
        if mr >= MONTHLY_CAP_PCT * 0.9: months_at_cap += 1
    consistency = months_at_cap / months_total * 100 if months_total > 0 else 0

    return {
        'total': total, 'wins': wins, 'losses': losses, 'win_rate': wr,
        'total_pnl': total_pnl, 'final_balance': final_bal, 'total_return': total_ret,
        'max_dd': max_dd, 'max_dd_pct': max_dd_pct, 'profit_factor': pf,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'months_positive': months_positive, 'months_total': months_total,
        'consistency': consistency, 'monthly': monthly,
    }


# =============================================================================
#  OUTPUT — TRADES CSV (with Balance Pre / Balance Post)
# =============================================================================

def save_trades_csv(trades, path):
    rows = [{
        'Entry Time':       t['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'Exit Time':        t['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'Direction':        t['direction'],
        'Entry Price':      t['entry_price'],
        'SL Price':         t['sl_price'],
        'TP Price':         t['tp_price'],
        'Exit Price':       t['exit_price'],
        'Result':           t['result'],
        'Risk ($)':         t['risk'],
        'P/L ($)':          t['pnl'],
        'Balance Pre ($)':  t['balance_pre'],
        'Balance Post ($)': t['balance_post'],
    } for t in trades]
    pd.DataFrame(rows).to_csv(path / "trades.csv", index=False)


# =============================================================================
#  OUTPUT — REPORT TXT
# =============================================================================

def save_report_txt(trades, metrics, path):
    m = metrics
    monthly = m['monthly']

    report = f"""
================================================================================
    {ASSET} {TIMEFRAME} — {STRATEGY_NAME}
    Michele Piazzoli — Quant Strategy
================================================================================

STRATEGY CONFIGURATION
──────────────────────────────────────────────────────────────────────────────
  Asset:              {ASSET}
  Timeframe:          {TIMEFRAME}
  Trigger:            Breakout Range {BREAKOUT_PERIOD} Candles
  HTF Filter:         H1 SMA {HTF_SMA_PERIOD}
  Session:            {SESS_NAME} ({SESS_START} - {SESS_END})
  RR Ratio:           1:{RR}
  SL:                 ATR({ATR_PERIOD}) x {SL_ATR_MULT}
  Monthly Cap:        {MONTHLY_CAP_PCT}%
  Backtest:           {BACKTEST_START} → {BACKTEST_END}

RISK MANAGEMENT (DRS)
──────────────────────────────────────────────────────────────────────────────
  Initial Balance:    ${INITIAL_BALANCE:,.0f}
  DD Floor:           ${DD_FLOOR:,.0f}
  Risk Formula:       (Balance - {DD_FLOOR:,}) / {RISK_DIVISOR}
  Risk Range:         ${MIN_RISK:,} — ${MAX_RISK:,}

================================================================================
                          PERFORMANCE SUMMARY
================================================================================

  Total Trades:       {m['total']}
  Wins:               {m['wins']}
  Losses:             {m['losses']}
  Win Rate:           {m['win_rate']:.2f}%

  Total P/L:          ${m['total_pnl']:+,.2f}
  Final Balance:      ${m['final_balance']:,.2f}
  Total Return:       {m['total_return']:+.2f}%

  Max Drawdown:       ${m['max_dd']:,.2f} ({m['max_dd_pct']:.2f}%)
  Profit Factor:      {m['profit_factor']:.2f}
  Avg Win:            ${m['avg_win']:+,.2f}
  Avg Loss:           ${m['avg_loss']:+,.2f}

  Consistency Score:  {m['consistency']:.1f}%
  Months Positive:    {m['months_positive']}/{m['months_total']}

================================================================================
                          MONTHLY BREAKDOWN
================================================================================

  {"Month":<10} {"Trades":>7} {"Win Rate":>9} {"P/L":>12} {"Return %":>10} {"Max DD":>10} {"DD %":>7} {"End Balance":>14}
  {"─"*85}
"""
    for mo in sorted(monthly.keys()):
        d = monthly[mo]
        report += f"  {mo:<10} {d['trades']:>7} {d['win_rate']:>8.1f}% ${d['pnl']:>+10,.0f} {d['return_pct']:>+9.2f}% ${d['max_dd']:>9,.0f} {d['dd_pct']:>6.2f}% ${d['end_bal']:>13,.0f}\n"

    report += f"""
================================================================================
                          FIRST 20 TRADES
================================================================================

  {"#":>4} {"Time":>20} {"Dir":>6} {"Entry":>10} {"SL":>10} {"TP":>10} {"Exit":>10} {"Result":>7} {"Risk":>8} {"P/L":>10}
  {"─"*100}
"""
    for i, t in enumerate(trades[:20], 1):
        report += f"  {i:>4} {t['entry_time'].strftime('%Y-%m-%d %H:%M'):>20} {t['direction']:>6} {t['entry_price']:>10.2f} {t['sl_price']:>10.2f} {t['tp_price']:>10.2f} {t['exit_price']:>10.2f} {t['result']:>7} ${t['risk']:>7,.0f} ${t['pnl']:>+9,.0f}\n"

    report += f"""
================================================================================
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Michele Piazzoli — Quant Strategy
================================================================================
"""
    with open(path / "report.txt", 'w', encoding='utf-8') as f:
        f.write(report)


# =============================================================================
#  OUTPUT — REPORT HTML
# =============================================================================

def save_report_html(trades, metrics, path):
    m = metrics
    monthly = m['monthly']

    ml = sorted(monthly.keys())
    meq = []
    eq = INITIAL_BALANCE
    for mo in ml:
        eq += monthly[mo]['pnl']
        meq.append(round(eq, 0))

    cr = [0]
    for t in trades:
        cr.append(cr[-1] + t['pnl'] / INITIAL_BALANCE * 100)

    years = sorted(set(mo[:4] for mo in ml))
    mn = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    cal_rows = ""
    for y in years:
        cells = f'<td class="yc">{y}</td>'
        for mi in range(1, 13):
            k = f"{y}-{mi:02d}"
            if k in monthly:
                r = monthly[k]['return_pct']
                c = '#00e676' if r >= 0 else '#ff5252'
                cells += f'<td style="color:{c};font-weight:600">{r:+.1f}%</td>'
            else:
                cells += '<td style="color:#555">—</td>'
        cal_rows += f"<tr>{cells}</tr>"

    det = ""
    for mo in ml:
        d = monthly[mo]
        c = '#00e676' if d['pnl'] >= 0 else '#ff5252'
        det += f'<tr><td>{mo}</td><td>{d["trades"]}</td><td>{d["win_rate"]:.1f}%</td><td style="color:{c}">${d["pnl"]:+,.0f}</td><td style="color:{c}">{d["return_pct"]:+.2f}%</td><td>${d["max_dd"]:,.0f}</td><td>{d["dd_pct"]:.2f}%</td><td>${d["end_bal"]:,.0f}</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{ASSET} {TIMEFRAME} — {STRATEGY_NAME}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a1a;color:#e0e0e0;font-family:'Segoe UI',Arial,sans-serif;padding:40px 60px}}
h1{{text-align:center;font-size:32px;font-weight:700;color:#fff;margin-bottom:10px}}
h2{{font-size:18px;color:#b0b0b0;margin:40px 0 15px;font-weight:600;border-bottom:1px solid #1e1e3a;padding-bottom:8px}}
.sub{{text-align:center;color:#888;font-size:14px;margin-bottom:40px}}
.sg{{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin:30px 0}}
.sc{{background:#12122a;border:1px solid #1e1e3a;border-radius:10px;padding:20px;text-align:center}}
.sc .l{{color:#888;font-size:12px;text-transform:uppercase;letter-spacing:1px}}
.sc .v{{font-size:28px;font-weight:700;margin-top:8px;color:#fff}}
.sc .v.g{{color:#00e676}}.sc .v.r{{color:#ff5252}}
.cc{{background:#12122a;border:1px solid #1e1e3a;border-radius:10px;padding:25px;margin:25px 0}}
canvas{{max-height:350px}}
table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:13px}}
th{{background:#1a1a35;color:#b0b0b0;padding:10px 12px;text-align:center;font-weight:600;text-transform:uppercase;font-size:11px}}
td{{padding:8px 12px;text-align:center;border-bottom:1px solid #1a1a30}}
tr:hover{{background:#151530}}
.yc{{font-weight:700;color:#fff;text-align:left}}
.ft{{text-align:center;color:#555;font-size:12px;margin-top:50px;padding-top:20px;border-top:1px solid #1e1e3a}}
</style>
</head>
<body>

<h1>{ASSET} {TIMEFRAME} — {STRATEGY_NAME}</h1>
<p class="sub">{SESS_NAME} ({SESS_START}–{SESS_END}) | RR 1:{RR} | SL ATR x{SL_ATR_MULT} | HTF SMA {HTF_SMA_PERIOD} | Consistency {m['consistency']:.0f}%</p>

<div class="sg">
<div class="sc"><div class="l">Total Trades</div><div class="v">{m['total']}</div></div>
<div class="sc"><div class="l">Win Rate</div><div class="v {'g' if m['win_rate']>=40 else 'r'}">{m['win_rate']:.1f}%</div></div>
<div class="sc"><div class="l">Total Return</div><div class="v {'g' if m['total_return']>=0 else 'r'}">{m['total_return']:+.1f}%</div></div>
<div class="sc"><div class="l">Profit Factor</div><div class="v {'g' if m['profit_factor']>=1.5 else 'r'}">{m['profit_factor']:.2f}</div></div>
<div class="sc"><div class="l">Total P/L</div><div class="v {'g' if m['total_pnl']>=0 else 'r'}">${m['total_pnl']:+,.0f}</div></div>
<div class="sc"><div class="l">Final Balance</div><div class="v">${m['final_balance']:,.0f}</div></div>
<div class="sc"><div class="l">Max Drawdown</div><div class="v r">${m['max_dd']:,.0f} ({m['max_dd_pct']:.1f}%)</div></div>
<div class="sc"><div class="l">Avg Win / Loss</div><div class="v">${m['avg_win']:,.0f} / ${abs(m['avg_loss']):,.0f}</div></div>
</div>

<div class="cc"><h2 style="border:none;margin:0 0 15px">Equity Curve (Monthly)</h2><canvas id="eq"></canvas></div>
<div class="cc"><h2 style="border:none;margin:0 0 15px">Cumulative Return by Trade</h2><canvas id="cr"></canvas></div>

<h2>Monthly Returns Calendar</h2>
<table><thead><tr><th>Year</th>{''.join(f'<th>{x}</th>' for x in mn)}</tr></thead><tbody>{cal_rows}</tbody></table>

<h2>Monthly Returns Detail</h2>
<table><thead><tr><th>Month</th><th>Trades</th><th>Win Rate</th><th>P/L</th><th>Return %</th><th>Max DD</th><th>DD %</th><th>End Balance</th></tr></thead><tbody>{det}</tbody></table>

<div class="ft">Michele Piazzoli — Quant Strategy | Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

<script>
new Chart(document.getElementById('eq'),{{type:'line',data:{{labels:{ml},datasets:[{{data:{meq},borderColor:'#42a5f5',backgroundColor:'rgba(66,165,245,0.1)',borderWidth:2,pointRadius:4,pointBackgroundColor:'#42a5f5',fill:true,tension:0.3}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#888',maxRotation:45}},grid:{{color:'#1a1a30'}}}},y:{{ticks:{{color:'#888',callback:v=>'$'+v.toLocaleString()}},grid:{{color:'#1a1a30'}}}}}}}}}});
const cd={cr};const st=Math.max(1,Math.floor(cd.length/500));const sc=cd.filter((_,i)=>i%st===0||i===cd.length-1);const sl=sc.map((_,i)=>i*st);
new Chart(document.getElementById('cr'),{{type:'line',data:{{labels:sl,datasets:[{{data:sc,borderColor:'#66bb6a',backgroundColor:'rgba(102,187,106,0.1)',borderWidth:2,pointRadius:0,fill:true,tension:0.3}}]}},options:{{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{x:{{title:{{display:true,text:'Trade Number',color:'#888'}},ticks:{{color:'#888'}},grid:{{color:'#1a1a30'}}}},y:{{title:{{display:true,text:'Return %',color:'#888'}},ticks:{{color:'#888'}},grid:{{color:'#1a1a30'}}}}}}}}}});
</script>
</body></html>"""

    with open(path / "report.html", 'w', encoding='utf-8') as f:
        f.write(html)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    (OUTPUT_BASE / "[01] PDF Report").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "[02] CSV Trade list").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "[03] Report TXT").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "[04] Python code").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  {ASSET} {TIMEFRAME} — {STRATEGY_NAME}")
    print(f"  Michele Piazzoli — Quant Strategy")
    print("=" * 80)
    print()

    print("  Loading data...")
    df_raw = load_data(DATA_PATH)
    df = filter_date_range(df_raw, BACKTEST_START, BACKTEST_END)
    print(f"  Candles: {len(df):,}")
    print()

    print("  Running backtest...")
    trades, monthly_pnl = run_backtest(df)
    metrics = compute_metrics(trades, monthly_pnl)

    if metrics is None:
        print("  [ERROR] No trades generated.")
        return

    print(f"  Trades: {metrics['total']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  P/L: ${metrics['total_pnl']:+,.2f}")
    print(f"  Max DD: ${metrics['max_dd']:,.2f} ({metrics['max_dd_pct']:.2f}%)")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Consistency: {metrics['consistency']:.1f}%")
    print()

    print("  Saving reports...")
    save_report_html(trades, metrics, OUTPUT_BASE / "[01] PDF Report")
    save_trades_csv(trades, OUTPUT_BASE / "[02] CSV Trade list")
    save_report_txt(trades, metrics, OUTPUT_BASE / "[03] Report TXT")

    import shutil
    shutil.copy2(__file__, OUTPUT_BASE / "[04] Python code" / "backtest.py")

    print()
    print("  Output:")
    print(f"    [01] PDF Report/report.html")
    print(f"    [02] CSV Trade list/trades.csv")
    print(f"    [03] Report TXT/report.txt")
    print(f"    [04] Python code/backtest.py")
    print()
    print("=" * 80)
    print(f"  COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
