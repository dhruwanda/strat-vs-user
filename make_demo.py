"""Generate a fully fictional, engine-consistent demo dataset.

Six imaginary stocks. The index is built with the SAME methodology the engine
verifies (T1 OHLC-average transitions), the tradebook fires proper baskets,
charges follow the rate card, and one dividend belongs to a non-smallcase
holding so the exclusion logic has something to show. No real data anywhere.
"""
import numpy as np
import pandas as pd
import openpyxl

rng = np.random.default_rng(7)
OUT = "demo_data"

SYMS = ["ALBUS", "BELLATRIX", "CEDERY", "DRACO", "HEDWIG", "LUNA"]
NAMES = {"ALBUS": "Albus Industries Ltd", "BELLATRIX": "Bellatrix Power Ltd",
         "CEDERY": "Cedery Pharma Ltd", "DRACO": "Draco Metals Ltd",
         "HEDWIG": "Hedwig Logistics Ltd", "LUNA": "Luna Textiles Ltd"}
P0 = {"ALBUS": 520.0, "BELLATRIX": 145.0, "CEDERY": 880.0, "DRACO": 62.0,
      "HEDWIG": 240.0, "LUNA": 410.0}
DRIFT = {"ALBUS": .0009, "BELLATRIX": -.0004, "CEDERY": .0006, "DRACO": -.0012,
         "HEDWIG": .0004, "LUNA": .0011}

cal = pd.bdate_range("2026-01-02", "2026-06-30")
px = {}
for s in SYMS:
    steps = rng.normal(DRIFT[s], 0.012, len(cal))
    close = P0[s] * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.004, 0.002, len(cal)))
    px[s] = pd.DataFrame({
        "date": cal, "close": close,
        "open": close * (1 + rng.normal(0, 0.003, len(cal))),
        "high": close * (1 + spread), "low": close * (1 - spread)})
    px[s]["high"] = px[s][["open", "high", "close"]].max(axis=1)
    px[s]["low"] = px[s][["open", "low", "close"]].min(axis=1)

def bar(s, d, f):
    r = px[s][px[s].date == d].iloc[0]
    return float((r.open + r.high + r.low + r.close) / 4) if f == "ohlc" else float(r[f])

def next_td(d):
    return cal[cal > d][0]

# versions and flags
V1 = ("2026-01-02 to 2026-03-01", {"ALBUS": .20, "BELLATRIX": .20, "CEDERY": .20,
                                   "DRACO": .20, "HEDWIG": .20})
V2 = ("2026-03-02 to 2026-05-03", {"ALBUS": .20, "BELLATRIX": .20, "CEDERY": .20,
                                   "LUNA": .20, "HEDWIG": .20})
V3 = ("2026-05-04 to 2026-06-30", {"ALBUS": .24, "BELLATRIX": .19, "CEDERY": .19,
                                   "LUNA": .19, "HEDWIG": .19})
FLAGS = [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-03-02"),
         pd.Timestamp("2026-05-04")]

# ---- official index: fractional quantities, T1-OHLC transitions ----
q = {s: 100 * w / bar(s, cal[0], "close") for s, w in V1[1].items()}
rows = []
for d in cal:
    if d == FLAGS[1] or d == FLAGS[2]:
        pass  # transition happens on T1, below
    rows.append(dict(date=d, q=dict(q)))
# apply transitions
def transition(qold, wnew, t1):
    inter = sum(qq * bar(s, t1, "ohlc") for s, qq in qold.items())
    return {s: inter * w / bar(s, t1, "ohlc") for s, w in wnew.items()}
idx = []
q = {s: 100 * w / bar(s, cal[0], "close") for s, w in V1[1].items()}
t1_2, t1_3 = next_td(FLAGS[1]), next_td(FLAGS[2])
for d in cal:
    if d == t1_2:
        q = transition(q, V2[1], d)
    if d == t1_3:
        q = transition(q, V3[1], d)
    idx.append(dict(date=d, val=sum(qq * bar(s, d, "close") for s, qq in q.items()),
                    flag=d in FLAGS))

# ---- timeline workbook ----
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Historical Index Values"
ws.append(["Date", "Index Value", "Rebalance Occured"])
for r in idx:
    ws.append([r["date"].strftime("%Y-%m-%d"), f"{r['val']:.2f}",
               "True" if r["flag"] else None])
ws2 = wb.create_sheet("Historical Constituents")
ws2.append(["Date Range", "Constituents", "Weightage"])
for rngtxt, w in (V1, V2, V3):
    first = True
    for s, wt in w.items():
        ws2.append([rngtxt if first else None, NAMES[s], wt]); first = False
wb.save(f"{OUT}/timeline.xlsx")

# ---- investor tradebook ----
trades = []
tid = [1000]
def add(sym, day, tstr, side, qty, price):
    tid[0] += 1
    trades.append(dict(symbol=sym, isin=f"INE{tid[0]}D01", trade_date=day.strftime("%Y-%m-%d"),
                       exchange="NSE", segment="EQ", series="EQ", trade_type=side,
                       auction=False, quantity=int(qty), price=round(price, 2),
                       trade_id=tid[0], order_id=tid[0],
                       order_execution_time=f"{day.strftime('%Y-%m-%d')}T{tstr}"))

hold = {}
def vwap(s, d):  # a live-ish price inside the day's range
    r = px[s][px[s].date == d].iloc[0]
    return float(r.low + 0.55 * (r.high - r.low))

# E1: first investment 2026-01-15, Rs 5,00,000
d1 = pd.Timestamp("2026-01-15"); A = 500000.0
for s, w in V1[1].items():
    p = vwap(s, d1); qq = round(A * w / p)
    add(s, d1, "15:20:59", "buy", qq, p); hold[s] = hold.get(s, 0) + qq

# E2: invest-more 2026-02-16, Rs 2,00,000 (top-up)
d2 = pd.Timestamp("2026-02-16"); A2 = 200000.0
V = sum(hold[s] * vwap(s, d2) for s in hold)
for s, w in V1[1].items():
    p = vwap(s, d2)
    need = max(0.0, (V + A2) * w - hold.get(s, 0) * p)
    qq = round(need / p)
    if qq:
        add(s, d2, "09:45:11", "buy", qq, p); hold[s] += qq

# E3: rebalance applied 2026-03-04 (flag 03-02, T1 03-03 -> 1 day late)
def rebalance(day, weights, tsell, tbuy):
    V = sum(hold[s] * vwap(s, day) for s in hold)
    tgt = {s: round(V * w / vwap(s, day)) for s, w in weights.items()}
    for s in list(hold):
        if s not in weights and hold[s] > 0:
            add(s, day, tsell, "sell", hold[s], vwap(s, day)); hold[s] = 0
    for s, t in tgt.items():
        cur = hold.get(s, 0); dq = t - cur
        if dq < 0:
            add(s, day, tsell, "sell", -dq, vwap(s, day))
        elif dq > 0:
            add(s, day, tbuy, "buy", dq, vwap(s, day))
        hold[s] = t
    hold.update({s: q for s, q in hold.items()})

rebalance(pd.Timestamp("2026-03-04"), V2[1], "09:30:10", "09:30:12")
# E4: rebalance applied 2026-05-06 (flag 05-04, T1 05-05 -> 1 day late)
rebalance(pd.Timestamp("2026-05-06"), V3[1], "09:31:00", "09:31:02")

tb = pd.DataFrame(trades)
tb.to_csv(f"{OUT}/tradebook.csv", index=False)

# ---- P&L workbook (holdings snapshot + charge totals + other debits) ----
last = cal[-1]
tb["val"] = tb.quantity * tb.price
buys = tb[tb.trade_type == "buy"].groupby("symbol")["val"].sum()
buyq = tb[tb.trade_type == "buy"].groupby("symbol")["quantity"].sum()
sells = tb[tb.trade_type == "sell"].groupby("symbol")["val"].sum()
sellq = tb[tb.trade_type == "sell"].groupby("symbol")["quantity"].sum()
etf = lambda s: False
stt = float((tb["val"] * 0.001).sum())
stamp = float((tb.loc[tb.trade_type == "buy", "val"] * 0.00015).sum())
exch = float((tb["val"] * 0.0000297).sum())
sebi = float((tb["val"] * 0.000001).sum())
ipft = float((tb["val"] * 0.0000001).sum())
gst = 0.18 * (exch + sebi + ipft)
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Equity"
ws.append(["Client ID", "DEMO42"])
ws.append(["P&L Statement for Equity from 2026-01-01 to 2026-06-30"])
ws.append(["Summary"])
realized_total = 0.0
hrows = []
for s in SYMS:
    bq, bv = float(buyq.get(s, 0)), float(buys.get(s, 0))
    sq, sv = float(sellq.get(s, 0)), float(sells.get(s, 0))
    if bq == 0:
        continue
    avg = bv / bq
    rp = sv - avg * sq
    realized_total += rp
    oq = bq - sq
    oc = avg * oq
    cp = bar(s, last, "close")
    hrows.append([NAMES[s][:4].upper() if False else s, f"INE{s[:3]}01", sq,
                  round(avg * sq, 2), round(sv, 2), round(rp, 2), 0.0,
                  round(cp, 2), oq, "", round(oc, 2),
                  round(oq * cp - oc, 2), 0.0])
unreal_total = sum(r[11] for r in hrows)
ws.append(["Charges", round(stt + stamp + exch + sebi + ipft + gst, 4)])
ws.append(["Other Credit & Debit", -(2 * 118.0 + 5 * 15.0)])
ws.append(["Realized P&L", round(realized_total, 2)])
ws.append(["Unrealized P&L", round(unreal_total, 2)])
ws.append(["Charges"]); ws.append(["Account Head", "Amount"])
for k, v in [("Brokerage", 0.0), ("Exchange Transaction Charges", exch),
             ("Integrated GST", gst), ("Securities Transaction Tax", stt),
             ("SEBI Turnover Fees", sebi), ("Stamp Duty", stamp), ("IPFT", ipft)]:
    ws.append([k, round(v, 4)])
ws.append(["Symbol", "ISIN", "Quantity", "Buy Value", "Sell Value",
           "Realized P&L", "Realized P&L Pct.", "Previous Closing Price",
           "Open Quantity", "Open Quantity Type", "Open Value",
           "Unrealized P&L", "Unrealized P&L Pct."])
for r in hrows:
    ws.append(r)
ws2 = wb.create_sheet("Other Debits and Credits")
ws2.append(["Client ID", "DEMO42"])
ws2.append(["Other Debits and Credits for EQ from 2026-01-01 to 2026-06-30"])
ws2.append(["Particulars", "Posting Date", "Debit", "Credit"])
ws2.append(["Being smallcase fee for 15/01/2026", "2026-01-15", 118.0, 0.0])
ws2.append(["Being smallcase fee for 16/02/2026", "2026-02-16", 118.0, 0.0])
for s in ["DRACO", "ALBUS", "CEDERY", "HEDWIG", "BELLATRIX"]:
    for dd in ["04/03/2026", "06/05/2026"]:
        day = pd.Timestamp(dd[6:] + "-" + dd[3:5] + "-" + dd[:2])
        sold = tb[(tb.symbol == s) & (tb.trade_type == "sell") &
                  (tb.trade_date == day.strftime("%Y-%m-%d"))]
        if len(sold):
            ws2.append([f"DP Charges for Sale of {s} on {dd}",
                        day.strftime("%Y-%m-%d"), 15.0, 0.0])
wb.save(f"{OUT}/pnl.xlsx")

# ---- dividends (one belongs to a holding outside the smallcase) ----
pd.DataFrame([
    dict(**{"Symbol": "ALBUS", "Ex-date": "2026-04-10", "Qty": int(hold["ALBUS"]),
            "Dividend per share": 6.0,
            "Total dividend": 6.0 * int(hold["ALBUS"])}),
    dict(**{"Symbol": "CEDERY", "Ex-date": "2026-06-05", "Qty": int(hold["CEDERY"]),
            "Dividend per share": 11.0,
            "Total dividend": 11.0 * int(hold["CEDERY"])}),
    dict(**{"Symbol": "NIMBUS", "Ex-date": "2026-05-20", "Qty": 40,
            "Dividend per share": 2.5, "Total dividend": 100.0}),
]).to_csv(f"{OUT}/dividends.csv", index=False)

# ---- price cache ----
pd.concat([px[s].assign(symbol=s, volume=0)[
    ["symbol", "date", "open", "high", "low", "close", "volume"]]
    for s in SYMS]).to_csv(f"{OUT}/prices_cache.csv", index=False)
print("demo written; final holdings:", {k: int(v) for k, v in hold.items() if v})
