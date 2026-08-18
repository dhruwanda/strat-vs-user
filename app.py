"""Smallcase: the strategy vs you - Streamlit MVP.

UI only. Every number comes from the deterministic engine in
smallcase_attribution/; the LLM box explains those numbers and calculates
nothing. Deployable on Hugging Face Spaces (Streamlit SDK); set GEMINI_API_KEY
in the Space secrets to enable the question box.
"""
import json
import os
import tempfile

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
  .block-container {max-width: 820px; padding-top: 2.2rem;}
  h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}
  h1 {font-size: 2.1rem;}
  .stMetric label {color: #6b6b6b !important;}
  .quiet {color:#8a8a8a; font-size:0.9rem;}
  .stButton>button {border-radius: 999px; padding: 0.4rem 1.3rem;}
  footer, #MainMenu {visibility: hidden;}
</style>""", unsafe_allow_html=True)

DEMO = os.path.join(os.path.dirname(__file__), "demo_data")


# ----------------------------------------------------------------------------
def _save(upload):
    if upload is None:
        return None
    suffix = os.path.splitext(upload.name)[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(upload.getbuffer())
    f.close()
    return f.name


@st.cache_resource(show_spinner=False)
def analyse(timeline, tradebook, pnl, dividends, prices, subscription):
    cfg = Config(subscription_fee=float(subscription or 0.0))
    res = run(timeline, tradebook, pnl, cfg, {}, dividends_path=dividends)
    out = None
    if prices and os.path.exists(prices):
        store = PD.PriceStore.from_csv(prices)
        out = v2.run_gap_decomposition(res, store, cfg)
    return res, out


def _rs(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"₹{x:,.0f}"


def _llm_context(res, out):
    """Compact, structured facts for the LLM. Numbers only, already computed."""
    imp = res["implementation"]
    ctx = {
        "headline": {
            "money_put_in": round(imp["money_put_in"]),
            "current_value": round(imp["current_value"]),
            "total_return_rs": round(imp["total_returns"]),
            "total_return_pct": round(imp["total_return_pct"] * 100, 2),
            "strategy_index_return_pct":
                round(res["strategy_summary"]["absolute_return"] * 100, 2),
            "dividends_rs": round(imp.get("dividends") or 0),
        },
        "costs": res["cost_summary"][["cost_head", "smallcase"]]
                 .round(0).to_dict("records"),
        "tax_base_no_rates_applied": res["tax_base"].round(0).to_dict("records"),
        "rebalance_events": res["event_summary"][
            ["event_id", "date", "kind", "net_cash", "lag_days"]]
            .assign(date=lambda d: d["date"].astype(str)).round(0)
            .to_dict("records"),
    }
    if out is not None:
        impl = out["implementation"]
        ctx["gap_attribution"] = {
            "summary": impl["summary"].round(0).to_dict("records"),
            "by_stock_top": impl["by_stock"].head(12).round(0).to_dict("records"),
            "by_event": impl["by_event"].round(0).to_dict("records"),
            "model_vs_actual_pnl_context": {
                k: round(float(v)) for k, v in out["overall"]
                .set_index("metric")["value"].items()
                if isinstance(v, (int, float)) and np.isfinite(v)},
        }
    return ctx


SYSTEM = """You explain a retail investor's smallcase performance using ONLY the
JSON facts provided. Rules:
- Never perform financial calculations beyond simple arithmetic on the given
  numbers (differences, percentages of given totals).
- If a question needs data not in the JSON, say what is missing.
- Positive implementation price effect means the investor transacted at better
  prices than the model reference; explain in plain language.
- The engine applies NO tax rates; the tax_base table gives realised
  gains/losses by year, asset class and holding term. You may explain typical
  Indian treatment in general terms but must say the rate depends on the
  user's own slab and full tax position, and is not advice.
- Be concise, warm, specific. Use rupee figures from the JSON."""


def _secret(name, default=""):
    """Space secrets arrive as env vars and (on Streamlit Spaces) via st.secrets.
    Accessing st.secrets with no secrets file can raise, so guard both."""
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
    contents = []
    for role, text in history[-6:]:
        contents.append({"role": "user" if role == "user" else "model",
                         "parts": [{"text": text}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    body = {
        "system_instruction": {"parts": [{
            "text": SYSTEM + "\n\nFACTS:\n" + json.dumps(ctx)}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
    }
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as ex:  # noqa: BLE001
        return f"Could not reach the model ({type(ex).__name__}). Try again."


# ----------------------------------------------------------------------------
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
                              min_value=0.0, value=0.0, step=500.0,
                              help="Total you paid the manager, since you "
                                   "subscribed. From your invoices.")
    b1, b2 = st.columns([1, 1])
    go = b1.button("Analyse", type="primary",
                   disabled=not (f_tl and f_tb and f_pl))
    demo = b2.button("…or explore the demo (fictional data)")
    st.markdown('<p class="quiet">Files never leave the app session. No '
                'account, no database. The demo uses an entirely fictional '
                'smallcase: Albus, Bellatrix, Cedery, Draco and Luna.</p>',
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
                           F["dividends"], F["prices"], F["sub"])
    except Exception as ex:  # noqa: BLE001
        st.error(f"The engine could not process these files: {ex}")
        if st.button("Start over"):
            st.session_state.files = None
            st.rerun()
        st.stop()

if F.get("demo"):
    st.info("Demo mode - a fictional smallcase. Every mechanism is real; "
            "no number is.")

imp = res["implementation"]
strat = res["strategy_summary"]
gap_rs = imp["total_returns"] - imp["money_put_in"] * strat["absolute_return"]

m = st.columns(5)
m[0].metric("Money put in", _rs(imp["money_put_in"]))
m[1].metric("Current value", _rs(imp["current_value"]))
m[2].metric("Your return", f"{imp['total_return_pct']*100:,.2f}%")
m[3].metric("Strategy return", f"{strat['absolute_return']*100:,.2f}%")
m[4].metric("You vs strategy", _rs(gap_rs),
            f"{(imp['total_return_pct']-strat['absolute_return'])*100:+.2f} pp")

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
         ("Costs: brokerage, taxes on trades, fees, subscription", -costs),
         ("Income tax: the engine applies no rates - see the tax base below",
          0.0)]
for label, val in rows:
    c = st.columns([4, 1])
    c[0].write(label)
    c[1].write(f"**{_rs(val)}**")
if out is None:
    st.warning("The price-level attribution needs EOD prices. Download the "
               "plan below, run the fetch script once on your machine, and "
               "re-upload the cache it writes.")
    st.download_button("price_plan.json",
                       json.dumps({"note": "generate via run_v2.py"}, indent=1),
                       file_name="price_plan.json")

if out is not None:
    st.subheader("The model vs you, day by day")
    ds = out["daily_series"].set_index("date")
    st.line_chart(ds, height=280)
    st.markdown('<p class="quiet">Same money, same dates. One line traded like '
                'the model; the other is you.</p>', unsafe_allow_html=True)
    st.caption("Attribution detail")
    with st.expander("Which stocks explain it"):
        st.dataframe(out["implementation"]["by_stock"], hide_index=True)
    with st.expander("Event by event"):
        st.dataframe(out["implementation"]["by_event"], hide_index=True)
    with st.expander("Every leg, classified (the snap rule at work)"):
        st.dataframe(out["implementation"]["lines"], hide_index=True)
    with st.expander("Rebalance timeline and conventions"):
        st.dataframe(out["timeline"], hide_index=True)
    with st.expander("Did we get smallcase's methodology right? "
                     "(index replication)"):
        st.dataframe(out["replication_summary"], hide_index=True)

st.caption("The separate layers")
with st.expander("Costs, in full"):
    st.dataframe(res["cost_summary"], hide_index=True)
    st.dataframe(res["unattributed_costs"], hide_index=True)
with st.expander("Dividends, attributed"):
    if len(res["dividend_statement"]):
        st.dataframe(res["dividend_statement"], hide_index=True)
    else:
        st.write("No statement uploaded; using the configured figure.")
with st.expander("Tax base (no rates applied)"):
    st.dataframe(res["tax_base"], hide_index=True)
    for n in res["tax_notes"]:
        st.markdown(f'<p class="quiet">• {n}</p>', unsafe_allow_html=True)
with st.expander("Assumptions and limitations"):
    st.dataframe(res["assumptions"], hide_index=True)
    st.dataframe(res["limitations"], hide_index=True)

st.subheader("Ask about your numbers")
ctx = _llm_context(res, out)
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)
q = st.chat_input("Why did I do better than the strategy? How much did "
                  "costs eat? Which stock helped most?")
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
