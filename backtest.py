#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest B3 vs CDI — Meta: superar o CDI (+spread anual)
Saídas em ./outputs: metrics.csv, trades.csv, equity.csv, cdi_index.csv,
equity_curve.png, drawdown.png, excess_return.png
"""
import os, math
from datetime import datetime
import numpy as np
import pandas as pd

# Backend offscreen p/ GitHub Actions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf

# -------- Indicadores --------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ru/(rd+1e-12); return 100-(100/(1+rs))

def macd_hist(s, fast=12, slow=26, signal=9):
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    return m - sig

def bollinger_pctb(s, n=20, k=2):
    ma = s.rolling(n).mean(); std = s.rolling(n).std()
    up = ma + k*std; lo = ma - k*std
    return (s - lo) / (up - lo + 1e-12)

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s, n=200):
    m = s.rolling(n).mean(); sd = s.rolling(n).std()
    return (s - m)/(sd+1e-12)

def composite_ic(df):
    rsi_z = zscore(rsi(df['close']), 200)
    macd_z = zscore(macd_hist(df['close']), 200)
    pctb_z = zscore(bollinger_pctb(df['close']), 200)
    sma50 = df['close'].rolling(50).mean()
    slope_z = zscore(sma50.diff(), 200)
    ic = 0.3*rsi_z + 0.35*macd_z + 0.25*pctb_z + 0.1*slope_z
    return ic.clip(-3,3).fillna(0)

# -------- CDI (SGS + fallback) --------
def fetch_cdi_series(start_date: str, end_date: str, fallback_annual: float|None=None) -> pd.Series:
    try:
        import requests
        def try_code(code:int):
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
            r = requests.get(url, timeout=15); r.raise_for_status(); js = r.json()
            if not js: return None
            df = pd.DataFrame(js); df['data']=pd.to_datetime(df['data'], dayfirst=True)
            df['valor']=pd.to_numeric(df['valor'], errors='coerce')
            return df.set_index('data').sort_index()['valor']
        for code in [4389,4390,12]:
            try:
                s = try_code(code)
                if s is not None and len(s)>0:
                    daily = s/100.0/252.0
                    idx = (1+daily).cumprod()
                    idx = idx.reindex(pd.bdate_range(idx.index.min(), idx.index.max())).ffill()
                    return idx/idx.iloc[0]
            except Exception:
                continue
    except Exception:
        pass
    if fallback_annual is None: fallback_annual = 0.12
    bdays = pd.bdate_range(start_date, end_date)
    daily = (1+fallback_annual)**(1/252)-1
    idx = pd.Series((1+daily)**np.arange(len(bdays)), index=bdays)
    return idx/idx.iloc[0]

# -------- Posição --------
class Position:
    def __init__(self, ticker, entry_date, entry_price, qty, tp_excess, sl, max_hold):
        self.ticker=ticker; self.entry_date=entry_date; self.entry_price=entry_price
        self.qty=qty; self.tp_excess=tp_excess; self.sl=sl; self.days_held=0; self.max_hold=max_hold

def perf_metrics(equity: pd.Series):
    rets = equity.pct_change().fillna(0)
    cagr = (equity.iloc[-1]/equity.iloc[0])**(252/len(equity)) - 1
    sharpe = np.sqrt(252)*(rets.mean()/(rets.std()+1e-12))
    roll_max = equity.cummax(); dd = equity/roll_max - 1
    return {'CAGR':float(cagr),'Sharpe':float(sharpe),'MaxDrawdown':float(dd.min()),'FinalEquity':float(equity.iloc[-1])}

# -------- Backtest --------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', type=str, default='PETR4.SA,VALE3.SA,ITUB4.SA,BBDC4.SA,ABEV3.SA,BBAS3.SA,B3SA3.SA,WEGE3.SA')
    ap.add_argument('--start', type=str, default='2016-01-01')
    ap.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'))
    ap.add_argument('--horizon', type=int, default=30)
    ap.add_argument('--excess-ann', type=float, default=0.05)
    ap.add_argument('--risk-profile', type=str, default='moderado', choices=['conservador','moderado','arrojado'])
    ap.add_argument('--budget', type=float, default=100000.0)
    ap.add_argument('--max-positions', type=int, default=8)
    ap.add_argument('--cdi-annual', type=float, default=None)
    args = ap.parse_args()

    os.makedirs('outputs', exist_ok=True)

    # Baixa dados
    data={}
    for t in [x.strip() for x in args.tickers.split(',') if x.strip()]:
        df = yf.download(t, start=args.start, end=args.end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"[WARN] Sem dados para {t}. Pulando."); continue
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df['ATR'] = atr(df['high'], df['low'], df['close']).fillna(method='bfill')
        df['IC']  = composite_ic(df)
        df['SMA50']= df['close'].rolling(50).mean()
        data[t]=df
    if not data: raise SystemExit("Sem dados. Verifique tickers e conexão (Yahoo).")

    all_dates = pd.bdate_range(min(min(df.index) for df in data.values()),
                               max(max(df.index) for df in data.values()))
    cdi_idx = fetch_cdi_series(str(all_dates[0].date()), str(all_dates[-1].date()), fallback_annual=args.cdi_annual).reindex(all_dates).ffill()

    target_excess_period = (1 + args.excess_ann)**(args.horizon/252) - 1
    rpt = {'conservador':0.01, 'moderado':0.015, 'arrojado':0.025}[args.risk_profile]

    cash = args.budget; open_positions=[]; equity = pd.Series(index=all_dates, dtype=float); trades=[]

    for date in all_dates:
        # Exits
        new_open=[]
        for pos in open_positions:
            df=data[pos.ticker]
            if date not in df.index: new_open.append(pos); continue
            row=df.loc[date]; price=float(row['close']); low=float(row['low']); ic=float(row['IC'])
            pos.days_held += 1
            cdi_factor = float(cdi_idx.loc[date]/cdi_idx.loc[pos.entry_date]) if (pos.entry_date in cdi_idx.index) else 1.0
            trade_factor = price/pos.entry_price
            excess = trade_factor/cdi_factor - 1.0
            exit_flag=False; reason=''; exit_price=price
            if low <= pos.sl: exit_flag=True; reason='SL'; exit_price=pos.sl
            elif excess >= pos.tp_excess: exit_flag=True; reason='TP_excess'
            elif ic < 0: exit_flag=True; reason='IC<0'
            elif pos.days_held >= args.horizon: exit_flag=True; reason='Time'
            if exit_flag:
                cash += pos.qty*exit_price
                ret = (exit_price - pos.entry_price)/pos.entry_price
                trades.append({'ticker':pos.ticker,'entry_date':pos.entry_date.strftime('%Y-%m-%d'),
                               'exit_date':date.strftime('%Y-%m-%d'),'entry':round(float(pos.entry_price),4),
                               'exit':round(float(exit_price),4),'return_pct':float(ret),'cdi_factor':float(cdi_factor),
                               'excess_over_cdi':float((1+ret)/cdi_factor - 1),'days':pos.days_held,'reason':reason})
            else:
                new_open.append(pos)
        open_positions=new_open

        # Entries
        if len(open_positions) < args.max_positions:
            cands=[]
            for t,df in data.items():
                if date not in df.index: continue
                if any(p.ticker==t for p in open_positions): continue
                row=df.loc[date]
                if pd.isna(row['SMA50']): continue
                if row['IC']>1.0 and row['close']>row['SMA50']:
                    cands.append((t,float(row['IC']),float(row['close']),float(row['ATR'])))
            cands.sort(key=lambda x:x[1], reverse=True)
            for t,ic_v,close_v,atr_v in cands[:(args.max_positions-len(open_positions))]:
                entry=close_v; sl = entry - 1.5*atr_v
                risk_per_share = max(entry - sl, 1e-6)
                qty = math.floor((args.budget*rpt)/risk_per_share)
                if qty<=0: continue
                cost = qty*entry
                if cost>cash: continue
                cash -= cost
                open_positions.append(Position(t, date, entry, qty, target_excess_period, sl, args.horizon))

        eq=cash
        for p in open_positions:
            df=data[p.ticker]
            if date in df.index:
                eq += p.qty*float(df.loc[date,'close'])
        equity.loc[date]=eq

    # -------- Saídas --------
    metrics = perf_metrics(equity)
    cdi_equity = (equity.iloc[0]*(cdi_idx/cdi_idx.iloc[0])).reindex(equity.index).ffill()
    excess_curve = equity/cdi_equity - 1

    pd.DataFrame([metrics]).to_csv('outputs/metrics.csv', index=False)
    pd.DataFrame(trades).to_csv('outputs/trades.csv', index=False)
    equity.to_csv('outputs/equity.csv', header=['equity'])
    cdi_equity.to_csv('outputs/cdi_index.csv', header=['cdi_equity'])

    plt.figure(figsize=(10,5))
    plt.plot(equity.index, equity.values, label='Portfolio')
    plt.plot(cdi_equity.index, cdi_equity.values, label='CDI Index')
    plt.title('Equity Curve vs CDI'); plt.legend(); plt.tight_layout()
    plt.savefig('outputs/equity_curve.png', dpi=140); plt.close()

    roll_max = equity.cummax(); dd = equity/roll_max - 1
    plt.figure(figsize=(10,3)); plt.plot(dd.index, dd.values)
    plt.title('Drawdown'); plt.tight_layout()
    plt.savefig('outputs/drawdown.png', dpi=140); plt.close()

    plt.figure(figsize=(10,3)); plt.plot(excess_curve.index, excess_curve.values)
    plt.title('Excesso de Retorno sobre o CDI'); plt.tight_layout()
    plt.savefig('outputs/excess_return.png', dpi=140); plt.close()

    print("METRICS:", metrics)

if __name__ == "__main__":
    main()
