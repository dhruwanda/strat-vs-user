"""
Fetch the EOD prices this analysis needs, on your own machine.

Only needed if the app could not fetch them itself (NSE refuses some hosting
providers' addresses). Download price_plan.json from the app, put it beside
this file, then:

    pip install pandas requests
    python fetch_prices.py

Writes prices_cache.csv - upload that back into the app.
"""
import json
import sys

import pandas as pd

sys.path.insert(0, ".")
try:
    from smallcase_attribution import nse
except ImportError:  # standalone use, without the package next to it
    sys.exit("Put this file in the project folder (it uses "
             "smallcase_attribution/nse.py), or download the whole repo.")

plan = json.load(open("price_plan.json"))
if isinstance(plan, list):  # older plan format: symbol/start/end rows
    df = pd.DataFrame(plan)
    dates = pd.bdate_range(pd.to_datetime(df["start"]).min(),
                           pd.to_datetime(df["end"]).max())
    symbols = sorted(df["symbol"].unique())
else:
    dates, symbols = plan["dates"], plan.get("symbols")

print(f"{len(symbols) if symbols else 'all'} symbols, {len(dates)} trading days")


def tick(i, n):
    if i % 5 == 0 or i == n:
        print(f"  {i}/{n}")


px, failed = nse.fetch(dates, symbols, progress=tick)
px.to_csv("prices_cache.csv", index=False)
print(f"wrote prices_cache.csv: {len(px)} bars, {px['symbol'].nunique()} symbols")
if failed:
    print("could not fetch:", ", ".join(str(d.date()) for d in failed))
if symbols:
    missing = sorted(set(s.upper() for s in symbols) - set(px["symbol"]))
    if missing:
        print("no data for:", ", ".join(missing))
