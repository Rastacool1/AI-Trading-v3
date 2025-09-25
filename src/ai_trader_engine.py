
# ai_trader_engine.py
import pandas as pd, numpy as np

def rsi(series, n=14):
    d=series.diff(); up=np.where(d>0,d,0.0); dn=np.where(d<0,-d,0.0)
    ru=pd.Series(up,index=series.index).rolling(n).mean()
    rd=pd.Series(dn,index=series.index).rolling(n).mean()
    rs=ru/(rd+1e-12); return 100-(100/(1+rs))

def roc(series, n=10): return series.pct_change(n)

def macd(series, fast=12, slow=26, signal=9):
    ef=series.ewm(span=fast, adjust=False).mean()
    es=series.ewm(span=slow, adjust=False).mean()
    m=ef-es; s=m.ewm(span=signal, adjust=False).mean(); h=m-s
    return m,s,h

def load_price_csv(csv_path_or_buffer):
    df=pd.read_csv(csv_path_or_buffer)
    cols={c.lower():c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    dc=pick('data','date'); cc=pick('zamkniecie','close','adj close','adj_close')
    hc=pick('najwyzszy','high'); lc=pick('najnizszy','low')
    if dc is None or cc is None: raise ValueError('Brak kolumn Data/Date lub Zamkniecie/Close')
    if hc is None or lc is None:
        base=pd.to_numeric(df[cc], errors='coerce'); df['__high__']=base*1.001; df['__low__']=base*0.999; hc='__high__'; lc='__low__'
    out=pd.DataFrame({'Data':pd.to_datetime(df[dc]), 'Zamkniecie':pd.to_numeric(df[cc],errors='coerce'),
                      'Najwyzszy':pd.to_numeric(df[hc],errors='coerce'), 'Najnizszy':pd.to_numeric(df[lc],errors='coerce')}).dropna()
    out=out.sort_values('Data').set_index('Data')
    out['ret']=out['Zamkniecie'].pct_change(); out['RSI14']=rsi(out['Zamkniecie'],14)
    tr=pd.concat([(out['Najwyzszy']-out['Najnizszy']).abs(),(out['Najwyzszy']-out['Zamkniecie'].shift(1)).abs(),(out['Najnizszy']-out['Zamkniecie'].shift(1)).abs()],axis=1).max(axis=1)
    out['ATR14']=tr.rolling(14).mean(); out['EMA10']=out['Zamkniecie'].ewm(span=10,adjust=False).mean()
    out['EMA50']=out['Zamkniecie'].ewm(span=50,adjust=False).mean(); out['EMA200']=out['Zamkniecie'].ewm(span=200,adjust=False).mean()
    out['ROC10']=roc(out['Zamkniecie'],10); m,s,h=macd(out['Zamkniecie']); out['MACD']=m; out['MACDsig']=s; out['MACDhist']=h
    return out

def _run_lengths(returns):
    s=np.sign(returns.dropna()); up=[]; dn=[]; cur=0; sign=0
    for v in s:
        if v==0:
            if sign>0 and cur>0: up.append(cur)
            if sign<0 and cur>0: dn.append(cur)
            cur=0; sign=0
        else:
            if v==sign: cur+=1
            else:
                if sign>0 and cur>0: up.append(cur)
                if sign<0 and cur>0: dn.append(cur)
                sign=v; cur=1
    if sign>0 and cur>0: up.append(cur)
    if sign<0 and cur>0: dn.append(cur)
    return float(np.mean(up)) if up else 7.0, float(np.mean(dn)) if dn else 7.0

def _avg_daily_moves(returns):
    pos=returns[returns>0]; neg=returns[returns<0]
    au=float(pos.mean()) if len(pos)>0 else 0.0045
    ad=float(-neg.mean()) if len(neg)>0 else 0.006
    return abs(au), abs(ad)

def dynamic_thresholds(rets):
    auL, adL=_run_lengths(rets); au, ad=_avg_daily_moves(rets)
    L_up=int(round(max(3,min(20,auL)))); L_dn=int(round(max(3,min(20,adL))))
    return {'avg_up_len':auL,'avg_dn_len':adL,'avg_up':au,'avg_dn':ad,'L_up':L_up,'L_dn':L_dn,'up_thr':au*auL,'dn_thr':ad*adL}

def _rolling_vol(returns, w=30): return returns.rolling(w).std()

def _trend(df, w=60):
    ema50=df['EMA50']; ema200=df['EMA200']
    ln=pd.Series(np.log(df['Zamkniecie']), index=df.index)
    t=pd.Series(np.arange(len(df),dtype=float), index=df.index)
    cov=ln.rolling(w).cov(t); var=t.rolling(w).var(); slope=cov/(var+1e-12)
    bull=(df['Zamkniecie']>ema200)&(ema50>ema200)&(slope>0)
    bear=(df['Zamkniecie']<ema200)&(ema50<ema200)&(slope<0)
    tr=pd.Series('side',index=df.index); tr[bull]='bull'; tr[bear]='bear'; return tr

def _vol_bucket(vol):
    q1=vol.quantile(1/3); q2=vol.quantile(2/3)
    b=pd.Series('midvol',index=vol.index); b[vol<=q1]='lowvol'; b[vol>=q2]='highvol'; return b

def _hyst(s, k=3):
    out=s.copy(); last=None; cnt=0
    for i,(idx,val) in enumerate(s.items()):
        if last is None: last=val; out.iloc[i]=val; cnt=1; continue
        if val==last: out.iloc[i]=last; cnt=1
        else:
            if cnt+1>=k: last=val; out.iloc[i]=val; cnt=1
            else: out.iloc[i]=last; cnt+=1
    return out

def label_regimes(df, vol_win=30, slope_win=60, hysteresis_days=3):
    vol=_rolling_vol(df['ret'].fillna(0), vol_win); trend=_trend(df, slope_win); vb=_vol_bucket(vol)
    reg=trend+'_'+vb
    allowed={'bear_lowvol','bear_midvol','bull_highvol','bull_midvol','bull_lowvol','side_highvol','side_midvol','side_lowvol'}
    reg=reg.where(reg.isin(allowed),'side_midvol')
    if hysteresis_days>1: reg=_hyst(reg, hysteresis_days)
    return reg

def compute_regime_thresholds(df, vol_win=30, slope_win=60, hysteresis_days=3, ew_halflife=90):
    reg=label_regimes(df, vol_win, slope_win, hysteresis_days); stats={}
    for name in reg.dropna().unique():
        m=reg==name; rets=df.loc[m,'ret'].dropna()
        if len(rets)<60: continue
        w=pd.Series(np.exp(-np.linspace(len(rets),1,len(rets))/ew_halflife), index=rets.index)
        pos=rets[rets>0]; neg=rets[rets<0]
        if len(pos)==0 or len(neg)==0: continue
        au=float((pos*w.reindex(pos.index).fillna(0)).sum()/(w.reindex(pos.index).fillna(0).sum()+1e-12))
        ad=float((-neg*w.reindex(neg.index).fillna(0)).sum()/(w.reindex(neg.index).fillna(0).sum()+1e-12))
        auL, adL=_run_lengths(rets); L_up=int(round(max(3,min(20,auL)))); L_dn=int(round(max(3,min(20,adL))))
        up_thr=au*auL; dn_thr=ad*adL
        k_atr=3.0 if 'highvol' in name else (1.8 if 'lowvol' in name else 2.3)
        stats[name]={'avg_up_len':auL,'avg_dn_len':adL,'avg_up':au,'avg_dn':ad,'L_up':L_up,'L_dn':L_dn,'up_thr':up_thr,'dn_thr':dn_thr,'k_atr':k_atr}
    return reg, stats

class ValidatorState:
    def __init__(self):
        self.post_sell=None; self.post_buy=None; self.last_sell_i=None; self.last_swing_high=None; self.filtered_out=[]

def run_engine_regime(df, start_date=None, entry_rsi=30, overheat_rsi=75, exit_trigger_rsi=70, min_hold_days=7,
                      exit_relax_mult=0.8, dd_stop=0.20, buy_validate_mult=1.0, sell_validate_mult=1.0,
                      allow_min_rebound_buy=True, allow_breakout_buy=True, alloc_per_trade=100.0, enable_regimes=True,
                      vol_win=30, slope_win=60, hysteresis_days=3, use_atr_norm=True, k_atr=2.0, fees_bps=0.0, slippage_bps=0.0,
                      min_gap_days=3, use_filter_roc=True, roc_window=10, roc_min=0.0, use_filter_macd=True,
                      use_filter_atr_band=True, atr_band_mult=1.0):
    core=dynamic_thresholds(df['ret'].dropna()) if df['ret'].notna().any() else {'avg_up_len':7,'avg_dn_len':7,'avg_up':0.0045,'avg_dn':0.006,'L_up':7,'L_dn':7,'up_thr':0.0315,'dn_thr':0.042}
    if enable_regimes: regimes, stats = compute_regime_thresholds(df, vol_win, slope_win, hysteresis_days)
    else: regimes, stats = (pd.Series('side_midvol', index=df.index), {})

    df_bt=df.loc[pd.to_datetime(start_date):].copy() if start_date else df.copy()
    dates=df_bt.index; price=df_bt['Zamkniecie']; rsi14=df_bt['RSI14']; ema=df_bt['EMA10']; atr=df_bt['ATR14']
    roc10=df_bt['ROC10']; macd_line=df_bt['MACD']; macd_sig=df_bt['MACDsig']

    pos=None; entry_p=None; entry_i=None; peak=None; touched=False; last_sig_i=None
    sig=[]; trd=[]; params=[]; V=ValidatorState()

    def get_stats(i):
        d=dates[i]; reg=regimes.reindex([d]).iloc[0] if d in regimes.index else 'side_midvol'
        st=stats.get(reg, core).copy(); st['k_atr_used']=st.get('k_atr', k_atr); return reg, st

    def can_emit(i): return True if last_sig_i is None else (i-last_sig_i)>=min_gap_days

    def passes(i, typ, st, p, e, a):
        if use_filter_atr_band and typ=="SELL" and a==a and e==e and p>(e+atr_band_mult*a): return False
        if use_filter_roc:
            r=roc10.iloc[i]
            if typ=="BUY" and not (r>=roc_min): return False
        if use_filter_macd:
            m=macd_line.iloc[i]; s=macd_sig.iloc[i]
            if typ=="BUY" and not (m>=s or (s-m)<0.0005): return False
            if typ=="SELL" and not (m<=s or (m-s)<0.0005): return False
        return True

    for i in range(1,len(dates)):
        d=dates[i]; p=price.iloc[i]; r1=rsi14.iloc[i-1]; r2=rsi14.iloc[i]; e=ema.iloc[i]; a=atr.iloc[i]
        reg, ST = get_stats(i)

        if V.last_sell_i is not None:
            V.last_swing_high=float(price.iloc[V.last_sell_i:i+1].max())

        if pos is None:
            entered=False
            if (r1>=entry_rsi) and (r2<entry_rsi) and can_emit(i) and passes(i,"BUY",ST,p,e,a):
                pos="long"; entry_p=p; entry_i=i; peak=p; touched=(r2>=overheat_rsi)
                sig.append({'date':d,'type':'BUY','price':p,'note':f'RSI<{entry_rsi} [{reg}]'}); last_sig_i=i; entered=True
            elif V.last_sell_i is not None and pd.notna(ST['up_thr']) and can_emit(i):
                lb=min(int(ST['L_dn']), i-V.last_sell_i) if pd.notna(ST['L_dn']) else 0
                if lb>=1:
                    w=price.iloc[i-lb:i+1]; mn=w.min(); rise=p/mn-1.0
                    if (rise>=ST['up_thr']) and (r2<70) and passes(i,"BUY",ST,p,e,a):
                        pos="long"; entry_p=p; entry_i=i; peak=p; touched=(r2>=overheat_rsi)
                        sig.append({'date':d,'type':'BUY','price':p,'note':f'Re-entry(min) [{reg}]'}); last_sig_i=i; entered=True
            if entered: continue

        if pos=="long":
            peak=max(peak,p) if peak is not None else p
            if r2>=overheat_rsi: touched=True
            cross=(touched and (r1>=exit_trigger_rsi) and (r2<exit_trigger_rsi))
            k_used=ST.get('k_atr_used',k_atr); trail=peak-k_used*a if (a==a and peak is not None) else None
            cond_trail=(trail is not None) and (p<trail); cond_ema=p<e if (e==e) else False
            cand = cross or cond_trail or cond_ema
            drop=1.0-(p/peak) if peak else 0.0; dd=1.0-(p/entry_p)
            dn_thr=ST['dn_thr']; 
            if use_atr_norm and a==a and p>0: dn_thr=max(dn_thr, (k_used*a)/p)
            valid=((i-entry_i)>=min_hold_days) and ((pd.notna(dn_thr) and (drop>=exit_relax_mult*dn_thr)) or (dd>=dd_stop))
            if cand and valid and can_emit(i) and passes(i,"SELL",ST,p,e,a):
                pnl=(p/entry_p-1.0)*100.0
                trd.append({'entry_date':dates[entry_i],'exit_date':d,'entry_price':entry_p,'exit_price':p,'bars':i-entry_i,'pnl_usd':pnl,'reason':'exit'})
                sig.append({'date':d,'type':'SELL','price':p,'note':f'exit [{reg}]'})
                V.last_sell_i=i; pos=None; entry_p=None; entry_i=None; peak=None; touched=False; last_sig_i=i
    return pd.DataFrame(sig), pd.DataFrame(trd), pd.DataFrame(), pd.DataFrame()
