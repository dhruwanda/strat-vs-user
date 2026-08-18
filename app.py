"""Smallcase: model vs you — Streamlit app.

UI and information architecture only. Every figure comes from the
deterministic engine in smallcase_attribution/; the question box explains
those figures and calculates nothing.
"""
import json
import os
import tempfile
from contextlib import contextmanager

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from smallcase_attribution import run, Config
from smallcase_attribution import price_data as PD, v2

st.set_page_config(page_title="Model vs you", page_icon="◐", layout="centered")
st.markdown("""
<style>
  .block-container {max-width: 780px; padding-top: 2.4rem;}
  h1,h2,h3 {font-family: Georgia,'Times New Roman',serif; font-weight: 500;
            letter-spacing:-.01em; color:#1c1c1c;}
  h1 {font-size:1.9rem;} h2 {font-size:1.25rem;} h3 {font-size:1.05rem;}
  .quiet {color:#7a7a75; font-size:.86rem; line-height:1.5;}
  .ctx {color:#4a4a45; font-size:.9rem;}
  .cmp {display:flex; gap:2.6rem; align-items:baseline; margin:.4rem 0 .2rem;}
  .cmp .lab {color:#7a7a75; font-size:.8rem; text-transform:uppercase;
             letter-spacing:.06em;}
  .cmp .val {font-family:Georgia,serif; font-size:2rem; color:#1c1c1c;}
  .gap {font-family:Georgia,serif; font-size:2.6rem; color:#2f6f4e;
        line-height:1.1;}
  .gap.neg {color:#a8443a;}
  .gapsub {color:#4a4a45; font-size:.95rem;}
  .stButton>button {border-radius:999px; padding:.4rem 1.4rem;}
  hr {margin:1.6rem 0; border-color:#eceae4;}
  footer, #MainMenu {visibility:hidden;}
  [data-testid="stMetricValue"] {font-size:1.25rem;}
</style>""", unsafe_allow_html=True)

DEMO = os.path.join(os.path.dirname(__file__), "demo_data")
GREEN, RED, GREY = "#2f6f4e", "#a8443a", "#9a9a92"


# ------------------------------------------------------------ file handling --
@contextmanager
def _materialise(blobs):
    """Uploaded bytes -> temp files for the loaders -> deleted on exit.

    Nothing the user uploads survives the analysis call.
    """
    paths, made = {}, []
    try:
        for key, (name, data) in blobs.items():
            if data is None:
                paths[key] = None
                continue
            fd, p = tempfile.mkstemp(suffix=os.path.splitext(name)[1])
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            paths[key] = p
            made.append(p)
        yield paths
    finally:
        for p in made:
            try:
                os.remove(p)
            except OSError:
                pass


@st.cache_resource(show_spinner=False, max_entries=2)
def analyse(_blobs, subscription, demo, key):
    cfg = Config(subscription_fee=float(subscription or 0.0),
                 basket_min_symbols=4 if demo else 5)
    with _materialise(_blobs) as p:
        res = run(p["timeline"], p["tradebook"], p["pnl"], cfg, {},
                  dividends_path=p["dividends"])
        out = (v2.run_gap_decomposition(res, PD.PriceStore.from_csv(p["prices"]),
                                        cfg) if p["prices"] else None)
    return res, out


def _read(upload):
    return (upload.name, upload.getbuffer().tobytes()) if upload else (None, None)


def _disk(path):
    return (os.path.basename(path), open(path, "rb").read())


# -------------------------------------------------------------------- utils --
def _rs(x, dec=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"₹{x:,.{dec}f}"


def _pp(x):
    return "—" if not np.isfinite(x) else f"{x:+.2f} pp"


def _llm_context(res, out, g):
    imp = res["implementation"]
    ctx = {
        "headline": {
            "model_return_on_your_cashflows_pct": round(g["model_return_pct"], 2),
            "your_actual_return_pct": round(g["your_return_pct"], 2),
            "difference_pp": round(g["gap_pp"], 2),
            "difference_rs": round(g["gap_rs"]),
            "strategy_index_return_pct_context": round(g["strategy_index_pct"], 2),
            "money_put_in": round(imp["money_put_in"]),
            "currently_invested": round(imp["current_investment"]),
            "current_value": round(imp["current_value"]),
            "realised_returns": round(imp["realized_returns"]),
            "dividends": round(imp.get("dividends") or 0),
        },
        "gap_contributors_pp_and_rs":
            g["contributors"].round(3).to_dict("records"),
        "per_stock_pnl": res["stock_attribution"][
            ["symbol", "realized_pnl", "unrealized_pnl", "total_pnl"]]
            .round(0).to_dict("records"),
        "costs_smallcase_only": res["cost_summary"][["cost_head", "smallcase"]]
            .round(0).to_dict("records"),
        "tax_base_no_rates_applied": res["tax_base"].round(0).to_dict("records"),
        "rebalances": res["event_summary"][["event_id", "date", "kind", "lag_days"]]
            .assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }
    if out is not None:
        impl = out["implementation"]
        ctx["price_difference_by_stock_rs"] = impl["by_stock"][
            ["symbol", "implementation_price_effect"]].round(0).to_dict("records")
        ctx["price_difference_by_event_rs"] = impl["by_event"].round(0).to_dict("records")
        ctx["leg_classification_counts"] = impl["snap_counts"].to_dict("records")
    return ctx


SYSTEM = """You answer questions about one investor's smallcase results using
ONLY the JSON facts given. Rules:
- Do not calculate returns or attribution yourself; quote and combine the given
  figures. Simple arithmetic on them (differences, shares of a total) is fine.
- If a question needs something not in the JSON, say what is missing.
- "Model" means the strategy applied to this investor's own cash-flow dates and
  amounts. Price differences are their fills against the model's transaction
  prices; quantity differences are share counts against the model's.
- The engine applies no tax rates. The tax base gives realised gains and losses
  by year, asset class and holding term; say that the applicable rate depends
  on the investor's own slab and full position, and is not advice.
- Concise, plain, specific. Use the rupee and pp figures given."""


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
        return ("This box needs a Gemini API key in the app's secrets as "
                "GEMINI_API_KEY. Everything else on the page works without it.")
    model = _secret("GEMINI_MODEL", "gemini-2.5-flash")
    contents = [{"role": "user" if r == "user" else "model", "parts": [{"text": t}]}
                for r, t in history[-6:]]
    contents.append({"role": "user", "parts": [{"text": question}]})
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}",
            json={"system_instruction": {"parts": [{
                      "text": SYSTEM + "\n\nFACTS:\n" + json.dumps(ctx)}]},
                  "contents": contents,
                  "generationConfig": {"temperature": .2, "maxOutputTokens": 800}},
            timeout=60)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as ex:  # noqa: BLE001
        return f"Could not reach the model ({type(ex).__name__}). Try again."


# ------------------------------------------------------------------ landing --
if "job" not in st.session_state:
    st.session_state.job = None
    st.session_state.chat = []

if st.session_state.job is None:
    st.title("Model vs you")
    st.markdown('<p class="ctx">The analysis reconstructs the strategy using '
                'your investment history and compares it with your actual '
                'implementation of the strategy, investments and rebalances.'
                '</p>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        f_tl = st.file_uploader("smallcase timeline export (.xlsx)", type=["xlsx"])
        f_tb = st.file_uploader("Zerodha tradebook (.csv)", type=["csv"])
        f_pl = st.file_uploader("Zerodha P&L (.xlsx)", type=["xlsx"])
    with c2:
        f_dv = st.file_uploader("Dividend statement — optional", type=["csv", "xlsx"])
        f_px = st.file_uploader("prices_cache.csv — optional, needed to explain "
                                "the gap", type=["csv"])
        sub = st.number_input("smallcase subscription paid (₹)", min_value=0.0,
                              value=0.0, step=500.0)
    b1, b2 = st.columns(2)
    go = b1.button("Run the analysis", type="primary",
                   disabled=not (f_tl and f_tb and f_pl))
    demo = b2.button("Explore the demo")
    st.markdown('<p class="quiet">Your files are processed for this session and '
                'aren\'t stored as portfolio records. The demo runs on a '
                'fictional smallcase.</p>', unsafe_allow_html=True)
    if go:
        st.session_state.job = dict(blobs=dict(
            timeline=_read(f_tl), tradebook=_read(f_tb), pnl=_read(f_pl),
            dividends=_read(f_dv), prices=_read(f_px)), sub=sub, demo=False,
            key=f"user-{f_tb.name}-{f_tb.size}")
        st.rerun()
    if demo:
        st.session_state.job = dict(blobs=dict(
            timeline=_disk(f"{DEMO}/timeline.xlsx"),
            tradebook=_disk(f"{DEMO}/tradebook.csv"),
            pnl=_disk(f"{DEMO}/pnl.xlsx"),
            dividends=_disk(f"{DEMO}/dividends.csv"),
            prices=_disk(f"{DEMO}/prices_cache.csv")),
            sub=2400.0, demo=True, key="demo")
        st.rerun()
    st.stop()

J = st.session_state.job
with st.spinner("Reconstructing…"):
    try:
        res, out = analyse(J["blobs"], J["sub"], J["demo"], J["key"])
    except Exception as ex:  # noqa: BLE001
        st.error(f"The engine could not process these files: {ex}")
        if st.button("Start over"):
            st.session_state.job = None
            st.rerun()
        st.stop()

g = v2.gap_reconciliation(res, out)
imp = res["implementation"]

st.title("Model vs you")
if J["demo"]:
    st.caption("Demo — a fictional smallcase. The mechanics are real; the "
               "numbers are invented.")

p1, p2, p3 = st.tabs(["Model vs you", "Why the difference", "Portfolio details"])

# ------------------------------------------------------------------ page 1 --
with p1:
    st.markdown(
        f'<div class="cmp">'
        f'<div><div class="lab">Model on your money</div>'
        f'<div class="val">{g["model_return_pct"]:+.2f}%</div></div>'
        f'<div><div class="lab">You</div>'
        f'<div class="val">{g["your_return_pct"]:+.2f}%</div></div></div>',
        unsafe_allow_html=True)
    ahead = g["gap_pp"] >= 0
    st.markdown(
        f'<div class="gap {"" if ahead else "neg"}">{g["gap_pp"]:+.2f} pp</div>'
        f'<div class="gapsub">{"ahead of" if ahead else "behind"} the model · '
        f'{_rs(g["gap_rs"])} on your money</div>', unsafe_allow_html=True)
    st.markdown('<p class="quiet">The model is the strategy applied to your own '
                'investment dates and amounts, not the strategy\'s headline '
                f'return. Strategy index over the same window: '
                f'{g["strategy_index_pct"]:+.2f}%.</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f'<p class="ctx">{_rs(imp["money_put_in"])} put in · '
        f'{_rs(imp["current_investment"])} currently invested · '
        f'{_rs(imp["current_value"])} current value</p>',
        unsafe_allow_html=True)
    st.markdown('<p class="quiet">Reconstructed from your trades and the '
                'strategy\'s historical data.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ page 2 --
with p2:
    ahead = g["gap_pp"] >= 0
    st.markdown(f"Your implementation was **{abs(g['gap_pp']):.2f} percentage "
                f"points {'ahead of' if ahead else 'behind'}** the model "
                f"({_rs(g['gap_rs'])}).")
    c = g["contributors"]
    show = c.assign(**{"Contribution": c["pp"].map(_pp),
                       "₹": c["rupees"].map(lambda v: _rs(v))})
    st.dataframe(show[["item", "Contribution", "₹", "note"]].rename(
        columns={"item": "", "note": "What it means"}),
        hide_index=True, use_container_width=True)
    if out is None:
        st.info("Upload prices_cache.csv to split the gap into price and "
                "quantity differences. Without it, everything beyond dividends "
                "stays unreconciled.")
        need = PD.required_lookups(res, res["index_values"],
                                   res["strategy_index"]["date"].iloc[-1])
        plan = {"symbols": sorted(need["symbol"].unique().tolist()),
                "dates": sorted({d.strftime("%Y-%m-%d") for d in need["date"]}),
                "note": "exactly the trading days this analysis reads: model "
                        "reference dates, your trade dates and the valuation "
                        "date"}
        st.download_button(
            f"Download price_plan.json ({len(plan['symbols'])} symbols, "
            f"{len(plan['dates'])} days)", json.dumps(plan, indent=1),
            file_name="price_plan.json", mime="application/json")
        st.markdown('<p class="quiet">Run fetch_prices.py with this plan on '
                    'your own machine, then upload the prices_cache.csv it '
                    'writes.</p>', unsafe_allow_html=True)
    else:
        st.markdown("### Which stocks moved the gap")
        bs = out["implementation"]["by_stock"].copy()
        bs["pp"] = 100.0 * bs["implementation_price_effect"] / g["denominator"]
        bs["abs"] = bs["pp"].abs()
        bs = bs.sort_values("abs", ascending=False)
        top = bs.head(5)
        st.dataframe(pd.DataFrame({
            "Stock": top["symbol"],
            "Contribution": top["pp"].map(_pp),
            "₹": top["implementation_price_effect"].map(lambda v: _rs(v))}),
            hide_index=True, use_container_width=True)
        if len(bs) > 5:
            with st.expander(f"The other {len(bs)-5} stocks"):
                rest = bs.iloc[5:]
                st.dataframe(pd.DataFrame({
                    "Stock": rest["symbol"],
                    "Contribution": rest["pp"].map(_pp),
                    "₹": rest["implementation_price_effect"].map(lambda v: _rs(v))}),
                    hide_index=True, use_container_width=True)

        st.markdown("### When the gap opened up")
        ev = v2.event_gap_series(res, out)
        base = alt.Chart(ev).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(grid=False)))
        bars = base.mark_bar(size=16, opacity=.45).encode(
            y=alt.Y("pp:Q", title="percentage points",
                    axis=alt.Axis(grid=True, gridColor="#f0efe9")),
            color=alt.condition(alt.datum.pp >= 0, alt.value(GREEN), alt.value(RED)),
            tooltip=[alt.Tooltip("date:T", title="Event"),
                     alt.Tooltip("label:N", title="Type"),
                     alt.Tooltip("pp:Q", title="This event (pp)", format="+.2f"),
                     alt.Tooltip("cumulative_pp:Q", title="Running (pp)",
                                 format="+.2f")])
        line = base.mark_line(strokeWidth=2, point=True, color="#1c1c1c",
                              interpolate="step-after").encode(
            y=alt.Y("cumulative_pp:Q", title=None))
        st.altair_chart((bars + line).properties(height=260).resolve_scale(
            y="shared").interactive(), use_container_width=True)
        st.markdown('<p class="quiet">Bars are each investment or rebalance\'s '
                    'price difference; the black line is the running total. '
                    'Points exist only where you traded — there is no implied '
                    'daily history.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ page 3 --
with p3:
    st.markdown(f'<p class="ctx">{_rs(imp["current_value"])} current value · '
                f'{_rs(imp["realized_returns"])} realised · '
                f'{_rs(imp.get("dividends"))} dividends</p>',
                unsafe_allow_html=True)
    with st.expander("Holdings", expanded=True):
        po = res["positions"][["symbol", "qty", "avg_price", "current_price",
                               "market_value", "unrealized_pnl"]].copy()
        po.columns = ["Stock", "Qty", "Avg buy", "Current", "Value",
                      "Unrealised P&L"]
        st.dataframe(po.round(2), hide_index=True, use_container_width=True)
    with st.expander("Realised"):
        if len(res["realisations"]):
            rl = (res["realisations"].groupby("symbol")
                  .agg(**{"Qty sold": ("qty", "sum"),
                          "Sell value": ("sell_value", "sum"),
                          "Realised P&L": ("realized_pnl", "sum")}).reset_index()
                  .rename(columns={"symbol": "Stock"}))
            st.dataframe(rl.round(0), hide_index=True, use_container_width=True)
        else:
            st.write("Nothing sold yet.")
    with st.expander("Rebalances"):
        cal = res["event_summary"]
        t = cal[cal["kind"] == "rebalance"][
            ["model_rebalance_date", "date", "lag_days"]].copy()
        t.columns = ["Suggested", "You executed", "Days later"]
        st.write(f"**{len(t)}** rebalances, plus "
                 f"**{int((cal['kind'] == 'invest').sum())}** investments.")
        st.dataframe(t, hide_index=True, use_container_width=True)
    with st.expander("Costs"):
        cs = res["cost_summary"][["cost_head", "smallcase"]].copy()
        cs.columns = ["Cost", "₹"]
        cs.loc[len(cs)] = ["Total", cs["₹"].sum()]
        st.dataframe(cs.round(2), hide_index=True, use_container_width=True)
    with st.expander("Dividends"):
        if len(res["dividend_statement"]):
            dv = res["dividend_statement"][
                ["symbol", "ex_date", "amount", "attributed_amount"]].copy()
            dv.columns = ["Stock", "Ex-date", "Received", "Yours via smallcase"]
            st.dataframe(dv.round(2), hide_index=True, use_container_width=True)
        else:
            st.write("No dividend statement uploaded.")
    with st.expander("Tax base"):
        tb_ = res["tax_base"].copy()
        tb_.columns = ["FY", "Asset class", "Term", "Gains", "Losses", "Net",
                       "Lots"]
        st.dataframe(tb_.round(0), hide_index=True, use_container_width=True)
        st.markdown('<p class="quiet">Realised gains and losses, classified. No '
                    'rates applied — the applicable rate depends on your own '
                    'slab and full tax position.</p>', unsafe_allow_html=True)
    with st.expander("Notes"):
        for _, r in res["limitations"].iterrows():
            st.markdown(f"- **{r['limitation']}.** {r['impact']}")

# --------------------------------------------------------------------- ask --
st.markdown("---")
st.markdown("### Have a question about the results?")
st.markdown('<p class="quiet">Which stocks contributed most? Why was my '
            'implementation ahead of the model? Which rebalance mattered '
            'most?</p>', unsafe_allow_html=True)
ctx = _llm_context(res, out, g)
for role, text in st.session_state.chat:
    st.chat_message(role).write(text)
q = st.chat_input("Ask about these numbers")
if q:
    st.chat_message("user").write(q)
    st.session_state.chat.append(("user", q))
    with st.spinner("Reading the results…"):
        a = ask_gemini(q, ctx, st.session_state.chat)
    st.chat_message("assistant").write(a)
    st.session_state.chat.append(("assistant", a))

if st.button("Start over"):
    st.session_state.job = None
    st.session_state.chat = []
    st.cache_resource.clear()
    st.rerun()
