
"""
All-Weather Portfolio (AWP) – NTNU Thesis Replication Backtest
----------------------------------------------------------------
This script replicates the data handling and backtesting logic described in
"The all-weather portfolio: A review of Bridgewater Associates investment strategy"
(NTNU, 2021), using freely available data where possible and drop-in CSVs when needed.

Key features implemented to mirror the thesis:
- Quarterly log returns for S&P 500, GSCI (Total Return), Gold.
- Treasury returns constructed from constant-maturity yields using modified duration and
  convexity (Swinkels, 2019) – see Eq. (6)–(8) in the thesis.
- Long-term and Intermediate-term Treasury portfolios per Eq. (9)–(10), then converted
  to quarterly returns via compounding monthly results and applying Eq. (5).
- Sub-portfolio mapping (x1..x4):
    x1: GSCI (high inflation, high growth)
    x2: Gold (high inflation, low growth)
    x3: S&P 500 (low inflation, high growth)
    x4: Treasuries – blend of Long- and Intermediate-term (low inflation, low growth)
- Risk allocation: equal risk contribution (assuming zero cross-correlation between
  sub-portfolios as in the thesis) across x1..x4. Within x4, the LT/IT split is chosen to
  maximize the portfolio Sharpe ratio over the in-sample period (long-only constraint).
- Historical comparison vs 60/40 (S&P500/Intermediate Treasuries) and 100% Equity.
- Risk & drawdown statistics + rolling holding-period stats.

Data dependencies / options:
- FRED via pandas_datareader for: SP500, GOLD (London fix), DGS5/10/20, TB3MS, CPIAUCSL, GDPC1.
- GSCI Total Return is NOT on FRED. Provide a CSV (monthly) at data/gsci_tr.csv.
  Expected format: columns: Date, Value (Date parseable; Value numeric). If not supplied,
  the script falls back to yfinance ticker 'GSG' (starts ~2006) as a proxy (limited history).

Reproducibility:
- The thesis uses 1970-02-01 through 2021-01-01 monthly sources; this script will clip to the
  common intersection of series actually available on your system. To replicate precisely,
  supply a long-history GSCI TR CSV.

Usage:
- Run as a script. Outputs a summary table and saves CSVs under ./outputs/.
- You may tweak CONFIG below (e.g., sample period, paths, plotting toggles).
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd


try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from scipy.optimize import minimize_scalar
except Exception:
    minimize_scalar = None


class CONFIG:
    START = pd.Timestamp("1970-02-01")   # thesis start (monthly)
    END   = pd.Timestamp("2021-01-01")   # thesis end (monthly)


    DATA_DIR = os.path.join(os.getcwd(), "data")
    OUT_DIR  = os.path.join(os.getcwd(), "outputs")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    
    GSCI_CSV = os.path.join(DATA_DIR, "gsci_tr.csv")  # columns: Date, Value (monthly)

    
    PLOT = False  # set True to draw basic figures (requires matplotlib)

    
    RF_FRED = "TB3MS"  # 3M T-bill (% p.a.), monthly

    
    Q_PER_YEAR = 4




def _to_period_end(s: pd.Series, rule: str) -> pd.Series:
    """Resample to period end (e.g., 'M' or 'Q') using last observation in period."""
    return s.resample(rule).last()


def _log_return(price: pd.Series) -> pd.Series:
    """Natural log return ln(P_t / P_{t-1})."""
    return np.log(price / price.shift(1))


def _compound_period(returns: pd.Series, freq_in: str, freq_out: str) -> pd.Series:
    """Compound arithmetic returns from freq_in to freq_out.
    E.g., monthly to quarterly: (1+r_m1)*(1+r_m2)*(1+r_m3)-1
    """
    gross = (1.0 + returns).resample(freq_out).prod()
    return gross - 1.0


def _annualize_from_quarterly(mean_q_excess: float, std_q: float) -> tuple[float, float]:
    """Return (ann_excess_mean, ann_vol) from quarterly stats."""
    ann_mean = mean_q_excess * CONFIG.Q_PER_YEAR
    ann_vol  = std_q * math.sqrt(CONFIG.Q_PER_YEAR)
    return ann_mean, ann_vol


def _cagr(quarterly_returns: pd.Series) -> float:
    gross = (1.0 + quarterly_returns).prod()
    years = len(quarterly_returns) / CONFIG.Q_PER_YEAR
    if years <= 0 or gross <= 0:
        return np.nan
    return gross ** (1.0 / years) - 1.0


def _max_drawdown(quarterly_returns: pd.Series) -> float:
    cum = (1.0 + quarterly_returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()




def load_fred_series(code: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                     q: bool = False) -> pd.Series:
    """Fetch a FRED series using pandas_datareader (if available). Returns a monthly or
    quarterly series as pd.Series with DatetimeIndex.
    """
    if pdr is None:
        raise RuntimeError("pandas_datareader not available; install it or provide CSVs.")
    df = pdr.DataReader(code, "fred", start, end)
    s = df.iloc[:, 0].dropna()
    s.index = pd.to_datetime(s.index)
    if q:
        
        s = _to_period_end(s, "Q")
    else:
        s = _to_period_end(s, "M")
    s.name = code
    return s


def load_sp500_monthly() -> pd.Series:
    """S&P 500 index level (price). Prefer FRED SP500 daily -> monthly last.
    Fallback to yfinance ^GSPC if needed.
    """
    try:
        s = load_fred_series("SP500", CONFIG.START, CONFIG.END)
        s.name = "SP500"
        return s
    except Exception:
        if yf is None:
            raise
        df = yf.download("^GSPC", start=CONFIG.START, end=CONFIG.END, interval="1mo", auto_adjust=True)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index)
        s.name = "SP500"
        return s


def load_gold_monthly() -> pd.Series:
    """London Gold AM Fix in USD from FRED (GOLDAMGBD228NLBM), monthly last."""
    return load_fred_series("GOLDAMGBD228NLBM", CONFIG.START, CONFIG.END)


def load_gsci_tr_monthly() -> pd.Series:
    """GSCI Total Return (monthly). Prefer a local CSV with long history. If missing,
    fallback to yfinance 'GSG' (limited history).
    CSV expected columns: Date, Value
    """
    if os.path.exists(CONFIG.GSCI_CSV):
        df = pd.read_csv(CONFIG.GSCI_CSV)
        
        date_col = "Date"
        val_col = "Value"
        if date_col not in df.columns:
            date_col = df.columns[0]
        if val_col not in df.columns:
            val_col = df.columns[1]
        s = pd.Series(df[val_col].values, index=pd.to_datetime(df[date_col]))
        s = s.sort_index().dropna()
        s = _to_period_end(s, "M")
        s.name = "GSCI_TR"
        return s
  
    if yf is None:
        raise RuntimeError("GSCI CSV missing and yfinance not available. Provide data/gsci_tr.csv.")
    df = yf.download("GSG", start=CONFIG.START, end=CONFIG.END, interval="1mo", auto_adjust=True)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    s.name = "GSCI_TR_proxy_GSG"
    return s


def load_treasury_yields_monthly() -> pd.DataFrame:
    """Load 5Y, 10Y, 20Y constant maturity yields from FRED (percent p.a.). Monthly last."""
    y5  = load_fred_series("DGS5",  CONFIG.START, CONFIG.END)
    y10 = load_fred_series("DGS10", CONFIG.START, CONFIG.END)
    y20 = load_fred_series("DGS20", CONFIG.START, CONFIG.END)
    df = pd.concat([y5, y10, y20], axis=1).dropna()
    df.columns = ["y5", "y10", "y20"]
   
    return df / 100.0


def load_rf_monthly() -> pd.Series:
    """3M T-bill monthly yield (decimal)."""
    rf = load_fred_series(CONFIG.RF_FRED, CONFIG.START, CONFIG.END)
    return rf / 100.0

def modified_duration(y: pd.Series, T: float) -> pd.Series:
    """Eq. (6): D_t = 1/y_t * [1 - 1 / (1 + y_t/2)^(2*T)]"""
    y = y.copy()
    return (1.0 / y) * (1.0 - 1.0 / np.power(1.0 + y/2.0, 2.0*T))


def convexity(y: pd.Series, T: float) -> pd.Series:
    """Eq. (7) as written in the thesis. Implemented algebraically to match monthly compounding.
    C_t = 2/y^2 * [1 - 1/(1 + y/2)^(2T)] - 2*T / [y * (1 + y/2)^(2T+1)]
    """
    y = y.copy()
    term1 = 2.0 / (y**2) * (1.0 - 1.0 / np.power(1.0 + y/2.0, 2.0*T))
    term2 = (2.0 * T) / (y * np.power(1.0 + y/2.0, 2.0*T + 1.0))
    return term1 - term2


def monthly_bond_return(y: pd.Series, T: float) -> pd.Series:
    """Eq. (8): R_t = y_{t-1} - D_t * (y_t - y_{t-1}) + 0.5 * C_t * (y_t - y_{t-1})^2
    y is monthly yield (decimal), T is maturity in years.
    Returns arithmetic monthly total return (not log).
    """
    y = y.dropna().copy()
    D = modified_duration(y, T)
    C = convexity(y, T)
    dy = y.diff()
    ret = y.shift(1) - D * dy + 0.5 * C * (dy**2)
    ret.name = f"bond_ret_T{T:g}"
    return ret


def build_treasury_portfolios_quarterly(yields_m: pd.DataFrame) -> pd.DataFrame:
    """Construct monthly returns for 5y/10y/20y, then form Long/Intermediate per Eq. (9)–(10),
    and aggregate to QUARTERLY arithmetic returns, then to quarterly log returns.
    Output columns: ['LT_q_log', 'IT_q_log']
    """
    r5m  = monthly_bond_return(yields_m["y5"],  5.0)
    r10m = monthly_bond_return(yields_m["y10"], 10.0)
    r20m = monthly_bond_return(yields_m["y20"], 20.0)

    r_long_m = (r20m + r10m) / 2.0
    r_int_m  = (r10m + r5m)  / 2.0

    r_long_q = _compound_period(r_long_m, "M", "Q")
    r_int_q  = _compound_period(r_int_m,  "M", "Q")

    out = pd.DataFrame({
        "LT_q_log": np.log(1.0 + r_long_q),
        "IT_q_log": np.log(1.0 + r_int_q),
    }).dropna()
    return out



def equal_risk_contrib_weights(vol_q: pd.Series) -> pd.Series:
    """Assuming zero correlations between sub-portfolios (as per thesis simplification),
    equal risk contribution across sub-portfolios implies weights proportional to 1/vol.
    Returns weights summing to 1.
    """
    inv_vol = 1.0 / vol_q
    w = inv_vol / inv_vol.sum()
    return w


def sharpe_ratio_q(excess_q: pd.Series) -> float:
    mu = excess_q.mean()
    sd = excess_q.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (mu / sd) * math.sqrt(CONFIG.Q_PER_YEAR)


def choose_treasury_split_to_maximize_sharpe(
    w_sub: pd.Series,
    lt_log_q: pd.Series, it_log_q: pd.Series,
    spx_log_q: pd.Series, gsci_log_q: pd.Series, gold_log_q: pd.Series,
    rf_log_q: pd.Series
) -> float:
    """Find alpha in [0,1] for LT weight inside sub-portfolio x4 (Treasuries) that maximizes
    the overall portfolio Sharpe. Total sub-portfolio weight for Treasuries is fixed at w_sub['x4'].
    Returns alpha (LT share within x4). Requires scipy; otherwise returns 0.8211 (thesis value).
    """
    if minimize_scalar is None:
        return 0.8211  

    df = pd.concat([
        lt_log_q.rename("LT"), it_log_q.rename("IT"),
        spx_log_q.rename("SPX"), gsci_log_q.rename("GSCI"), gold_log_q.rename("GOLD"),
        rf_log_q.rename("RF")
    ], axis=1).dropna()

    w1, w2, w3, w4 = w_sub["x1"], w_sub["x2"], w_sub["x3"], w_sub["x4"]

    def neg_sharpe(alpha: float) -> float:
        al = min(max(alpha, 0.0), 1.0)

        r_spx  = np.exp(df["SPX"])  - 1.0
        r_gsci = np.exp(df["GSCI"]) - 1.0
        r_gold = np.exp(df["GOLD"]) - 1.0
        r_lt   = np.exp(df["LT"])   - 1.0
        r_it   = np.exp(df["IT"])   - 1.0
        r_rf   = np.exp(df["RF"])   - 1.0

        r_tr = al * r_lt + (1.0 - al) * r_it
        r_p  = w1 * r_gsci + w2 * r_gold + w3 * r_spx + w4 * r_tr
        r_p_log = np.log(1.0 + r_p)
        r_ex_log = r_p_log - df["RF"]
        r_ex = (np.exp(r_ex_log) - 1.0)
        s = r_ex.mean() / (r_ex.std(ddof=0) + 1e-12)
        return -s

    res = minimize_scalar(neg_sharpe, bounds=(0.0, 1.0), method="bounded")
    if not res.success:
        return 0.8211
    return float(res.x)


def build_allwp_backtest():
    spx_m   = load_sp500_monthly()
    gold_m  = load_gold_monthly()
    gsci_m  = load_gsci_tr_monthly()

    ylds_m  = load_treasury_yields_monthly()
    rf_m    = load_rf_monthly()

    panel_m = pd.concat([
        spx_m.rename("SPX"), gsci_m.rename("GSCI"), gold_m.rename("GOLD"),
        ylds_m, rf_m.rename("RF")
    ], axis=1).dropna()

    panel_m = panel_m.loc[(panel_m.index >= CONFIG.START) & (panel_m.index <= CONFIG.END)]

    spx_q_log  = _log_return(_to_period_end(panel_m["SPX"],  "Q")).dropna()
    gsci_q_log = _log_return(_to_period_end(panel_m["GSCI"], "Q")).dropna()
    gold_q_log = _log_return(_to_period_end(panel_m["GOLD"], "Q")).dropna()

    lt_it_q = build_treasury_portfolios_quarterly(panel_m[["y5","y10","y20"]])

    rf_m_ret = panel_m["RF"] / 12.0  # decimal monthly
    rf_q_ret = _compound_period(rf_m_ret.rename("rf"), "M", "Q")
    rf_q_log = np.log(1.0 + rf_q_ret).rename("RF_q_log")

    q = pd.concat([
        spx_q_log.rename("SPX"), gsci_q_log.rename("GSCI"), gold_q_log.rename("GOLD"),
        lt_it_q, rf_q_log
    ], axis=1).dropna()


    vol_q = q[["GSCI","GOLD","SPX"]].std(ddof=0)
    tr_eq_log = np.log(1.0 + (np.exp(q["LT"]) - 1.0) * 0.5 + (np.exp(q["IT"]) - 1.0) * 0.5)
    vol_x4 = tr_eq_log.std(ddof=0)
    vol_all = pd.concat([vol_q, pd.Series({"x4": vol_x4})])
    vol_all.index = ["x1","x2","x3","x4"]

    w_sub = equal_risk_contrib_weights(vol_all)

    alpha_lt = choose_treasury_split_to_maximize_sharpe(
        w_sub,
        q["LT"], q["IT"], q["SPX"], q["GSCI"], q["GOLD"], q["RF_q_log"]
    )

    r_spx  = np.exp(q["SPX"])  - 1.0
    r_gsci = np.exp(q["GSCI"]) - 1.0
    r_gold = np.exp(q["GOLD"]) - 1.0
    r_lt   = np.exp(q["LT"])   - 1.0
    r_it   = np.exp(q["IT"])   - 1.0
    r_rf   = np.exp(q["RF_q_log"]) - 1.0

    r_tr = alpha_lt * r_lt + (1.0 - alpha_lt) * r_it

    w1, w2, w3, w4 = w_sub["x1"], w_sub["x2"], w_sub["x3"], w_sub["x4"]
    r_allwp = w1 * r_gsci + w2 * r_gold + w3 * r_spx + w4 * r_tr

    r_6040 = 0.60 * r_spx + 0.40 * r_it
    r_100e = r_spx.copy()  # 100% equity

    log_allwp = np.log(1.0 + r_allwp)
    log_6040  = np.log(1.0 + r_6040)
    log_100e  = np.log(1.0 + r_100e)

    r_ex_allwp = (np.exp(log_allwp - q["RF_q_log"]) - 1.0)
    r_ex_6040  = (np.exp(log_6040  - q["RF_q_log"]) - 1.0)
    r_ex_100e  = (np.exp(log_100e  - q["RF_q_log"]) - 1.0)

    def stats(name: str, rq: pd.Series, rex: pd.Series) -> dict:
        mu_q, sd_q = rq.mean(), rq.std(ddof=0)
        ann_ex, ann_sd = _annualize_from_quarterly(rex.mean(), rex.std(ddof=0))
        out = {
            "Portfolio": name,
            "Start": rq.index[0].date(),
            "End": rq.index[-1].date(),
            "CAGR": _cagr(rq),
            "AnnVol": ann_sd,
            "AnnSharpe": ann_ex / (ann_sd + 1e-12),
            "MaxDD": _max_drawdown(rq),
            "Quarters": len(rq),
        }
        return out

    table = []
    table.append(stats("ALLWP", r_allwp, r_ex_allwp))
    table.append(stats("60/40", r_6040,  r_ex_6040))
    table.append(stats("All-Equity", r_100e, r_ex_100e))

    df_stats = pd.DataFrame(table)

    def rolling_ann_return(r: pd.Series, years: int) -> pd.Series:
        w = CONFIG.Q_PER_YEAR * years
        gross = (1.0 + r).rolling(w).apply(np.prod, raw=True)
        ann = gross ** (1.0 / years) - 1.0
        return ann.dropna()

    roll = {
        "ALLWP": r_allwp,
        "60/40": r_6040,
        "All-Equity": r_100e,
    }
    roll_stats = []
    for label, series in roll.items():
        for yrs in [1,3,5,10]:
            ra = rolling_ann_return(series, yrs)
            if len(ra) == 0:
                continue
            roll_stats.append({
                "Portfolio": label,
                "WindowYears": yrs,
                "MedianAnnRet": ra.median(),
                ">%0Ann": (ra > 0).mean(),
                "WorstAnnRet": ra.min(),
            })
    df_roll = pd.DataFrame(roll_stats)

    df_stats.to_csv(os.path.join(CONFIG.OUT_DIR, "summary_stats.csv"), index=False)
    df_roll.to_csv(os.path.join(CONFIG.OUT_DIR, "rolling_stats.csv"), index=False)

    details = {
        "weights_subportfolios": w_sub.to_dict(),
        "alpha_LT_within_x4": alpha_lt,
        "weights_assets": {
            "GSCI": float(w1),
            "Gold": float(w2),
            "S&P500": float(w3),
            "LT": float(w4 * alpha_lt),
            "IT": float(w4 * (1.0 - alpha_lt)),
        },
        "common_quarter_range": (str(q.index[0].date()), str(q.index[-1].date())),
    }

    if CONFIG.PLOT:
        try:
            import matplotlib.pyplot as plt
            cum = pd.DataFrame({
                "ALLWP": (1.0 + r_allwp).cumprod(),
                "60/40": (1.0 + r_6040).cumprod(),
                "All-Equity": (1.0 + r_100e).cumprod(),
            })
            cum.plot(title="Cumulative Growth of $1 (Quarterly)")
            plt.show()
        except Exception:
            pass

    return df_stats, df_roll, details


if __name__ == "__main__":
    pd.options.display.float_format = "{:.4f}".format
    try:
        stats, rolling, info = build_allwp_backtest()
        print("\n== Summary Stats (saved to ./outputs/summary_stats.csv) ==\n")
        print(stats.to_string(index=False))
        print("\n== Rolling Holding-Period Stats (saved to ./outputs/rolling_stats.csv) ==\n")
        print(rolling.sort_values(["Portfolio","WindowYears"]).to_string(index=False))
        print("\n== Weights (sub-portfolios and assets) ==\n")
        for k, v in info.items():
            print(k, ":", v)
    except Exception as e:
        print("ERROR:", e)
        print("Hint: ensure pandas_datareader/yfinance are installed and provide data/gsci_tr.csv for long history.")

