"""
NSE daily bhavcopy fetch. One file per trading day covers every symbol, so the
request count equals the number of DATES the analysis needs - typically a
couple of dozen, not a full daily history.

Used by the app (attempted server-side first) and by fetch_prices.py (the
local fallback). NSE rejects some hosting providers' addresses, so every
failure path here is expected and reported rather than raised.
"""
from __future__ import annotations
import io
import zipfile

import pandas as pd

SERIES_OK = {"EQ", "BE", "BZ"}
COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def session():
    import requests
    s = requests.Session()
    s.headers.update(HEADERS)
    try:  # NSE rejects cold clients; warm the cookie jar
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:  # noqa: BLE001
        pass
    return s


def fetch_day(s, day: pd.Timestamp, timeout: int = 15):
    """One trading day, normalised. Returns None if unavailable."""
    day = pd.Timestamp(day)
    tag, tag2 = day.strftime("%d%m%Y"), day.strftime("%Y%m%d")

    for host in ("nsearchives.nseindia.com", "archives.nseindia.com"):
        try:
            r = s.get(f"https://{host}/products/content/sec_bhavdata_full_{tag}.csv",
                      timeout=timeout)
            if r.status_code == 200 and r.content[:20].lstrip().upper().startswith(b"SYMBOL"):
                df = pd.read_csv(io.BytesIO(r.content))
                df.columns = [c.strip() for c in df.columns]
                for c in df.select_dtypes("object"):
                    df[c] = df[c].astype(str).str.strip()
                return _norm(df, day, "SYMBOL", "SERIES", "OPEN_PRICE",
                             "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
                             "TTL_TRD_QNTY")
        except Exception:  # noqa: BLE001
            pass

    try:
        r = s.get(f"https://nsearchives.nseindia.com/content/cm/"
                  f"BhavCopy_NSE_CM_0_0_0_{tag2}_F_0000.csv.zip", timeout=timeout)
        if r.status_code == 200 and r.content[:2] == b"PK":
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(zf.open(zf.namelist()[0]))
            return _norm(df, day, "TckrSymb", "SctySrs", "OpnPric", "HghPric",
                         "LwPric", "ClsPric", "TtlTradgVol")
    except Exception:  # noqa: BLE001
        pass
    return None


def _norm(df, day, sym, series, o, h, l, c, v):  # noqa: E741
    out = pd.DataFrame({
        "symbol": df[sym].astype(str).str.strip(),
        "series": df[series].astype(str).str.strip(),
        "open": pd.to_numeric(df[o], errors="coerce"),
        "high": pd.to_numeric(df[h], errors="coerce"),
        "low": pd.to_numeric(df[l], errors="coerce"),
        "close": pd.to_numeric(df[c], errors="coerce"),
        "volume": pd.to_numeric(df.get(v), errors="coerce")})
    out["date"] = day.normalize()
    return out


def fetch(dates, symbols=None, pause: float = 0.25, timeout: int = 15,
          progress=None) -> tuple:
    """Fetch the given trading days. Returns (prices, failed_dates).

    Only the dates asked for are requested, so holidays never enter the result
    and no stale duplicate day can be collected.
    """
    import time
    dates = [pd.Timestamp(d).normalize() for d in sorted(set(dates))]
    syms = {str(x).upper() for x in symbols} if symbols else None
    s = session()
    frames, failed = [], []
    for i, d in enumerate(dates, 1):
        df = fetch_day(s, d, timeout=timeout)
        if df is None:
            failed.append(d)
        else:
            df = df[df["series"].isin(SERIES_OK)]
            if syms:
                df = df[df["symbol"].isin(syms)]
            frames.append(df[COLUMNS])
        if progress:
            progress(i, len(dates))
        if pause:
            time.sleep(pause)
    px = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=COLUMNS))
    return px.drop_duplicates(["symbol", "date"]), failed
