"""Smallcase: the strategy vs you - Streamlit MVP.

UI only. Every number comes from the deterministic engine in
smallcase_attribution/; the question box explains those numbers and
calculates nothing.
"""
import json
import os
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from smallcase_attribution import run, Config
from smallcase_attribution import price_data as PD, v2

st.set_page_config(page_title="The strategy vs you", page_icon="⚖️",
                   layout="centered")
st.markdown("""
<style>
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stHeader"] {
      background-color: white;
  }
  
  [data-testid="stSidebar"] {
      background-color: white;
  }
  .block-container {max-width: 860px; padding-top: 2.2rem;}
  h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}
  h1 {font-size: 2.1rem;}
  .stMetric label {color: #6b6b6b !important;}
  .quiet {color:#8a8a8a; font-size:0.9rem;}
  .stButton>button {border-radius: 999px; padding: 0.4rem 1.3rem;}
  footer, #MainMenu {visibility: hidden;}
</style>""", unsafe_allow_html=True)

DEMO = os.path.join(os.path.dirname(__file__), "demo_data")


def _save(upload):
    if upload is None:
        return None
    f = tempfile.NamedTemporaryFile(delete=False,
                                    suffix=os.path.splitext(upload.name)[1])
    f.write(upload.getbuffer())
    f.close()
    return f.name


@st.cache_resource(show_spinner=False)
def analyse(timeline, tradebook, pnl, dividends, prices, subscription, demo):
    cfg = Config(subscription_fee=float(subscription or 0.0),
                 # the demo smallcase holds only 4 names, so a basket there is
                 # 4 simultaneous fills; real smallcases keep the default
                 basket_min_symbols=4 if demo else 5)
    res = run(timeline, tradebook, pnl, cfg, {}, dividends_path=dividends)
    out = None
    if prices and os.path.exists(prices):
        out = v2.run_gap_decomposition(res, PD.PriceStore.from_csv(prices), cfg)
    return res, out


def _rs(x, dec=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"₹{x:,.{dec}f}"


def _llm_context(res, out):
    imp = res["implementation"]
    ctx = {
        "headline": {
            "money_put_in": round(imp["money_put_in"]),
            "current_investment": round(imp["current_investment"]),
            "current_value": round(imp["current_value"]),
            "current_returns_rs": round(imp["current_returns"]),
            "realized_returns_rs": round(imp["realized_returns"]),
            "dividends_rs": round(imp.get("dividends") or 0),
            "total_return_rs": round(imp["total_returns"]),
            "total_return_pct": round(imp["total_return_pct"] * 100, 2),
            "strategy_index_return_pct":
                round(res["strategy_summary"]["absolute_return"] * 100, 2),
        },
        "per_stock_pnl": res["stock_attribution"][
            ["symbol", "realized_pnl", "unrealized_pnl", "total_pnl"]]
            .round(0).to_dict("records"),
        "costs_smallcase_only": res["cost_summary"][["cost_head", "smallcase"]]
            .round(0).to_dict("records"),
        "tax_base_no_rates_applied": res["tax_base"].round(0).to_dict("records"),
        "rebalances": res["event_summary"][
            ["event_id", "date", "kind", "lag_days"]]
            .assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }
    if len(res["dividend_statement"]):
        ctx["dividends_detail"] = (res["dividend_statement"]
                                   [["symbol", "amount", "attributed_amount"]]
                                   .round(0).to_dict("records"))
    if out is not None:
        impl = out["implementation"]
        ctx["gap_attribution"] = {
            "summary": impl["summary"].round(0).to_dict("records"),
            "price_effect_by_stock": impl["by_stock"].round(0).to_dict("records"),
            "by_event": impl["by_event"].round(0).to_dict("records"),
            "leg_classification_counts": impl["snap_counts"].to_dict("records"),
        }
    return ctx


SYSTEM = """You explain a retail investor's smallcase performance using ONLY the
JSON facts provided. Rules:
- Never perform financial calculations beyond simple arithmetic on the given
  numbers (differences, shares of given totals).
- If a question needs data not in the JSON, say what is missing.
- Positive implementation price effect means the investor transacted at better
  prices than the model reference; explain in plain language.
- The engine applies NO tax rates; the tax_base table gives realised
  gains/losses by year, asset class and holding term. You may describe typical
  Indian treatment in general terms but must say the rate depends on the
  user's own slab and full position, and is not advice.
- Be concise, warm, specific. Use rupee figures from the JSON."""


def _secret(name, default=""):
    try:
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(name, default)


def ask_gemini(question, ctx, history):
    key = _secret("GEMINI_API_KEY")
    if not key:
        return ("The question box needs a (free-tier) Gemini API key in the "
                "app's secrets as GEMINI_API_KEY. Everything above works "
                "without it.")
    model = _secret("GEMINI_MODEL", "gemini-2.5-flash")
    contents = [{"role": "user" if r == "user" else "model",
                 "parts": [{"text": t}]} for r, t in history[-6:]]
    contents.append({"role": "user", "parts": [{"text": question}]})
    body = {"system_instruction": {"parts": [{
                "text": SYSTEM + "\n\nFACTS:\n" + json.dumps(ctx)}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}}
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as ex:  # noqa: BLE001
        return f"Could not reach the model ({type(ex).__name__}). Try again."


# ---------------------------------------------------------------- upload ----
st.title("The strategy vs you")
st.markdown('<p class="quiet">Your smallcase had a plan. You had an execution. '
            'This shows both, and explains the difference. Every number is '
            'computed deterministically; the question box only explains.</p>',
            unsafe_allow_html=True)

if "files" not in st.session_state:
    st.session_state.files = None
    st.session_state.chat = []

if st.session_state.files is None:
    st.subheader("Bring your files")
    c1, c2 = st.columns(2)
    with c1:
        f_tl = st.file_uploader("smallcase timeline export (.xlsx)", type=["xlsx"])
        f_tb = st.file_uploader("Zerodha tradebook (.csv)", type=["csv"])
        f_pl = st.file_uploader("Zerodha P&L (.xlsx)", type=["xlsx"])
    with c2:
        f_dv = st.file_uploader("Dividend statement - optional", type=["csv", "xlsx"])
        f_px = st.file_uploader("prices_cache.csv - optional, unlocks the gap "
                                "attribution", type=["csv"])
        sub = st.number_input("smallcase subscription paid over this period (₹)",
                              min_value=0.0, value=0.0, step=500.0)
    b1, b2 = st.columns(2)
    go = b1.button("Analyse", type="primary", disabled=not (f_tl and f_tb and f_pl))
    demo = b2.button("…or explore the demo (fictional data)")
    st.markdown('<p class="quiet">Files never leave the app session. No '
                'account, no database. The demo is an entirely fictional '
                'smallcase: Albus, Bellatrix, Cedery and friends.</p>',
                unsafe_allow_html=True)
    if go:
        st.session_state.files = dict(
            timeline=_save(f_tl), tradebook=_save(f_tb), pnl=_save(f_pl),
            dividends=_save(f_dv), prices=_save(f_px), sub=sub, demo=False)
        st.rerun()
    if demo:
        st.session_state.files = dict(
            timeline=f"{DEMO}/timeline.xlsx", tradebook=f"{DEMO}/tradebook.csv",
            pnl=f"{DEMO}/pnl.xlsx", dividends=f"{DEMO}/dividends.csv",
            prices=f"{DEMO}/prices_cache.csv", sub=2400.0, demo=True)
        st.rerun()
    st.stop()

F = st.session_state.files
with st.spinner("Reading the tape…"):
    try:
        res, out = analyse(F["timeline"], F["tradebook"], F["pnl"],
                           F["dividends"], F["prices"], F["sub"], F.get("demo"))
    except Exception as ex:  # noqa: BLE001
        st.error(f"The engine could not process these files: {ex}")
        if st.button("Start over"):
            st.session_state.files = None
            st.rerun()
        st.stop()

if F.get("demo"):
    st.info("Demo mode — a fictional smallcase. Every mechanism is real; "
            "no number is.")

imp = res["implementation"]
strat = res["strategy_summary"]

# ------------------------------------------------------------- headline ----
r1 = st.columns(3)
r1[0].metric("Current Value", _rs(imp["current_value"]))
r1[1].metric("Total Returns", _rs(imp["total_returns"]),
             f"{imp['total_return_pct']*100:+.2f}%")
r1[2].metric("Money Put In", _rs(imp["money_put_in"]))
r2 = st.columns(4)
r2[0].metric("Current Investment", _rs(imp["current_investment"]))
cr_pct = (imp["current_returns"] / imp["current_investment"] * 100
          if imp["current_investment"] else 0.0)
r2[1].metric("Current Returns", _rs(imp["current_returns"]), f"{cr_pct:+.2f}%")
r2[2].metric("Realized Returns", _rs(imp["realized_returns"]))
r2[3].metric("Dividends", _rs(imp.get("dividends")))

gap_pp = (imp["total_return_pct"] - strat["absolute_return"]) * 100
st.markdown(
    f"Over the same window the **strategy index** moved "
    f"**{strat['absolute_return']*100:+.2f}%**. You are "
    f"**{abs(gap_pp):.2f} percentage points {'ahead of' if gap_pp >= 0 else 'behind'}** it.")

# ------------------------------------------------- why the difference ------
st.subheader("Why the difference?")
costs = float(res["cost_summary"]["smallcase"].sum())
rows = []
if out is not None:
    s = out["implementation"]["summary"].set_index("measure")["amount"]
    rows += [("You transacted at different prices than the model",
              s["Implementation price effect - TOTAL"]),
             ("You held different quantities than the model",
              s["Quantity cash component - TOTAL"])]
rows += [("Dividends you received (the strategy index ignores them)",
          imp.get("dividends") or 0.0),
         ("Costs: charges on trades, smallcase fees, subscription", -costs),
         ("Income tax: no rates applied — see the tax base below", 0.0)]
for label, val in rows:
    c = st.columns([4, 1])
    c[0].write(label)
    c[1].write(f"**{_rs(val)}**")
if out is not None:
    with st.expander("Price effect, stock by stock"):
        d = out["implementation"]["by_stock"][
            ["symbol", "implementation_price_effect"]].copy()
        d.columns = ["Stock", "₹ vs the model's prices"]
        st.dataframe(d.round(0), hide_index=True, use_container_width=True)
else:
    st.warning("The price-level attribution needs EOD prices "
               "(prices_cache.csv). Everything else on this page works "
               "without it.")

# --------------------------------------------------------------- chart -----
if out is not None:
    st.subheader("The model vs you, day by day")
    ds = out["daily_series"].melt("date", var_name="book", value_name="value")
    ch = (alt.Chart(ds.dropna()).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False),
                axis=alt.Axis(format="~s")),
        color=alt.Color("book:N", title=None,
                        scale=alt.Scale(range=["#9aa0a6", "#1a73e8"]),
                        legend=alt.Legend(orient="top"))).properties(height=300))
    st.altair_chart(ch, use_container_width=True)
    st.markdown('<p class="quiet">Same money, same dates. The grey line traded '
                'exactly like the model; the blue one is you.</p>',
                unsafe_allow_html=True)

# ------------------------------------------------- returns, explained ------
st.subheader("Your returns, explained")
t_real, t_open = st.tabs(["Realised", "Current holdings (unrealised)"])
with t_real:
    if len(res["realisations"]):
        rl = (res["realisations"].groupby("symbol")
              .agg(**{"Qty sold": ("qty", "sum"),
                      "Sell value": ("sell_value", "sum"),
                      "Realised P&L": ("realized_pnl", "sum")})
              .reset_index().rename(columns={"symbol": "Stock"}))
        rl.loc[len(rl)] = ["TOTAL", rl["Qty sold"].sum(),
                           rl["Sell value"].sum(), rl["Realised P&L"].sum()]
        st.dataframe(rl.round(0), hide_index=True, use_container_width=True)
    else:
        st.write("Nothing sold yet.")
with t_open:
    po = res["positions"][["symbol", "qty", "avg_price", "current_price",
                           "market_value", "unrealized_pnl"]].copy()
    po.columns = ["Stock", "Qty", "Avg buy price", "Current price",
                  "Market value", "Unrealised P&L"]
    po.loc[len(po)] = ["TOTAL", po["Qty"].sum(), np.nan, np.nan,
                       po["Market value"].sum(), po["Unrealised P&L"].sum()]
    st.dataframe(po.round(2), hide_index=True, use_container_width=True)

st.subheader("What each stock contributed")
view = st.radio("Show", ["Total", "Realised", "Unrealised"], horizontal=True,
                label_visibility="collapsed")
col = {"Total": "total_pnl", "Realised": "realized_pnl",
       "Unrealised": "unrealized_pnl"}[view]
sa = (res["stock_attribution"][["symbol", col]]
      .sort_values(col, ascending=False).rename(
          columns={"symbol": "Stock", col: f"{view} P&L (₹)"}))
st.dataframe(sa.round(0), hide_index=True, use_container_width=True)

# ---------------------------------------------------------- the layers -----
st.caption("The separate layers")
with st.expander("Rebalances"):
    cal_df = (out["timeline"] if out is not None else res["rebalance_calendar"])
    if out is not None:
        t = cal_df[["T0_model_rebalance_date", "investor_applied_on",
                    "lag_trading_days"]].copy()
        t.columns = ["Suggested (T0)", "You executed", "Trading days later"]
    else:
        t = cal_df[["model_rebalance_date", "investor_applied_on", "lag_days"]].copy()
        t.columns = ["Suggested (T0)", "You executed", "Days later"]
    st.write(f"**{len(t)} rebalances** in your window.")
    st.dataframe(t, hide_index=True, use_container_width=True)
with st.expander("Costs (smallcase only)"):
    cs = res["cost_summary"][["cost_head", "smallcase"]].copy()
    cs.columns = ["Cost", "₹"]
    cs.loc[len(cs)] = ["TOTAL", cs["₹"].sum()]
    st.dataframe(cs.round(2), hide_index=True, use_container_width=True)
with st.expander("Dividends, attributed"):
    if len(res["dividend_statement"]):
        dv = res["dividend_statement"][["symbol", "ex_date", "amount",
                                        "attributed_amount", "attribution_note"]].copy()
        dv.columns = ["Stock", "Ex-date", "₹ received", "₹ yours via smallcase",
                      "Note"]
        st.dataframe(dv.round(2), hide_index=True, use_container_width=True)
    else:
        st.write("No statement uploaded; using the configured figure.")
with st.expander("Tax base (no rates applied)"):
    tb_ = res["tax_base"].copy()
    tb_.columns = ["FY", "Asset class", "Term", "Gains (₹)", "Losses (₹)",
                   "Net (₹)", "Lots"]
    st.dataframe(tb_.round(0), hide_index=True, use_container_width=True)
    st.markdown('<p class="quiet">The engine establishes what you realised and '
                'how it is classified; the applicable rate depends on your own '
                'slab and full tax position.</p>', unsafe_allow_html=True)
with st.expander("Notes & limitations"):
    for _, r in res["limitations"].iterrows():
        st.markdown(f"- **{r['limitation']}.** {r['impact']}")

# ------------------------------------------------------------------ ask ----
st.subheader("Ask about your numbers")
ctx = _llm_context(res, out)
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)
q = st.chat_input("Why am I ahead of the strategy? How much did costs eat? "
                  "Which stock helped most?")
if q:
    st.chat_message("user").write(q)
    st.session_state.chat.append(("user", q))
    with st.spinner("Reading the sheets…"):
        a = ask_gemini(q, ctx, st.session_state.chat)
    st.chat_message("assistant").write(a)
    st.session_state.chat.append(("assistant", a))

st.divider()
if st.button("Start over with different files"):
    st.session_state.files = None
    st.session_state.chat = []
    st.cache_resource.clear()
    st.rerun()
