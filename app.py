
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.ai_trader_engine import load_price_csv, run_engine_regime, compute_regime_thresholds, dynamic_thresholds
st.set_page_config(page_title="AI Trader – RSI Regime Engine", layout="wide")
st.title("AI Trader – RSI + Regimes + Validation (by SO)")
with st.sidebar:
    start = st.text_input("Start backtestu (YYYY-MM-DD)", "2025-01-01")
    entry_rsi = st.slider("RSI próg BUY", 10, 50, 30, 1)
    overheat_rsi = st.slider("RSI przegrzanie", 60, 90, 75, 1)
    exit_trigger_rsi = st.slider("RSI cross-down EXIT", 50, 90, 70, 1)
    min_hold_days = st.slider("Min dni trzymania", 1, 20, 7, 1)
    exit_relax_mult = st.slider("Mnożnik progu wyjścia", 0.5, 1.5, 0.8, 0.05)
    dd_stop = st.slider("Hard stop (DD od wejścia)", 0.05, 0.5, 0.20, 0.01)
    buy_validate_mult = st.slider("BUY: mnożnik progu spadku", 0.5, 1.5, 1.0, 0.05)
    sell_validate_mult = st.slider("SELL: mnożnik progu spadku", 0.5, 1.5, 1.0, 0.05)
    min_gap_days = st.slider("Refractory (dni między sygnałami)", 0, 20, 3, 1)
    enable_regimes = st.checkbox("Włącz reżimy", True)
    use_atr_norm = st.checkbox("Normalizuj progi ATR", True)
    k_atr = st.slider("k * ATR (fallback)", 1.0, 4.0, 2.0, 0.1)
    use_filter_roc = st.checkbox("Filtr ROC (BUY wymaga ROC>=0)", True)
    use_filter_macd = st.checkbox("Filtr MACD (BUY: MACD>=signal, SELL: MACD<=signal)", True)
    use_filter_atr_band = st.checkbox("Filtr ATR-band (blokuj SELL przy wybiciach)", True)
    atr_band_mult = st.slider("ATR band mnożnik", 0.5, 2.0, 1.0, 0.1)
    alloc_per_trade = st.number_input("Kwota per transakcja (USD)", 10.0, 100000.0, 100.0, 10.0)
    fees_bps = st.slider("Prowizja (bps, round-trip)", 0, 50, 4, 1)
    slippage_bps = st.slider("Poślizg (bps, round-trip)", 0, 50, 4, 1)

uploaded = st.file_uploader("Wgraj CSV (Stooq lub Date,Close,High,Low)", type=["csv"])
if uploaded is not None:
    df = load_price_csv(uploaded)
    core = dynamic_thresholds(df['ret'].dropna())
    st.write("Progi (core):", core)
    if st.button("Przelicz"):
        sig, trd, par, filt = run_engine_regime(df=df, start_date=start,
            entry_rsi=entry_rsi, overheat_rsi=overheat_rsi, exit_trigger_rsi=exit_trigger_rsi,
            min_hold_days=min_hold_days, exit_relax_mult=exit_relax_mult, dd_stop=dd_stop,
            buy_validate_mult=buy_validate_mult, sell_validate_mult=sell_validate_mult,
            alloc_per_trade=alloc_per_trade, enable_regimes=enable_regimes,
            use_atr_norm=use_atr_norm, k_atr=k_atr, fees_bps=fees_bps, slippage_bps=slippage_bps,
            min_gap_days=min_gap_days, use_filter_roc=use_filter_roc, use_filter_macd=use_filter_macd,
            use_filter_atr_band=use_filter_atr_band, atr_band_mult=atr_band_mult)
        roi = 100.0 + (trd["pnl_usd"].sum() if not trd.empty else 0.0)
        st.write(f"ROI: {roi:.2f} USD")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.loc[start: ].index, df.loc[start: ]['Zamkniecie'], label="Close")
        if not sig.empty:
            bm = sig['type']=="BUY"; sm = sig['type']=="SELL"
            if bm.any(): ax.scatter(pd.to_datetime(sig.loc[bm,'date']), df.loc[pd.to_datetime(sig.loc[bm,'date']),'Zamkniecie'], marker='^', s=70, label="BUY")
            if sm.any(): ax.scatter(pd.to_datetime(sig.loc[sm,'date']), df.loc[pd.to_datetime(sig.loc[sm,'date']),'Zamkniecie'], marker='v', s=70, label="SELL")
        ax.legend(); st.pyplot(fig)
        st.dataframe(sig); st.dataframe(trd)
