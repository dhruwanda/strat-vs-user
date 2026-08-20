# The engine, end to end

A complete walkthrough of what the code does, why each choice was made, and
what is still weak. Written to be read by someone who did not write it.

---

## 0. The question

Four questions, in order, each needing the one before it:

1. How did the smallcase strategy perform?
2. How would it have performed **on my money**, given when I actually invested?
3. How did my real implementation perform?
4. Why do 2 and 3 differ?

Question 2 is the one most tools skip, and it is the whole product. The
strategy's headline return assumes one lumpsum on day one. Nobody invests that
way. Comparing your return against that headline is comparing against a
portfolio you never had.

---

## 1. What goes in

| File | What is taken from it |
|---|---|
| smallcase timeline export | daily index values, rebalance flags, constituent names and weights per version |
| Zerodha tradebook | every trade: symbol, date, execution timestamp, buy/sell, quantity, price |
| Zerodha P&L workbook | holdings snapshot, charge totals, the Other Debits and Credits ledger |
| Dividend statement (optional) | symbol, ex-date, quantity, amount |
| Subscription paid | one number you type |
| EOD prices | fetched by the app; only the days the analysis reads |

Nothing is hardcoded to your smallcase. Every constituent name, weight, date
and rate comes from the files or from `Config`.

---

## 2. Which trades belong to the smallcase, which legs together form a cluster

Your demat account is one pot. If you bought GOLDBEES yourself and the
smallcase also holds GOLDBEES, the tradebook cannot tell them apart by symbol.
Something else has to.

**The signal: simultaneity.** A smallcase order routes every leg at once, so
all legs share an execution second. A human placing orders by hand cannot.

**The rule.** Group trades by execution timestamp. A timestamp with 2 or more
distinct symbols is a candidate *leg*. Legs within 10 seconds form a *cluster*.
A cluster qualifies as a smallcase order if either:

- its largest leg has **5 or more** symbols at one second, or
- it contains **both a buy leg and a sell leg**, each with 2 or more symbols.

The second condition exists because a small rebalance can touch as few as two
names per side. My January rebalance was exactly that: 2 sells and 2 buys.
A flat "5 or more" rule would have deleted a real event. Two multi-symbol legs
on opposite sides, seconds apart, cannot be produced by hand.

A lone small leg with no opposite side does **not** qualify. It is listed for
review rather than silently kept or dropped.

**Deferred legs.** A leg can fail on the day, usually a circuit limit, and be
completed days later as a standalone order. That order would look like a
personal trade. It is attached to its event only if all of these hold: the
event shows a model shortfall in that exact symbol and direction, the trade
falls within 5 days, and its quantity fits the shortfall. Zero fired on my
personal data. The mechanism is unit tested so it does not rot.

---

## 3. Matching constituent names to trading symbols

The timeline says "Ashok Leyland Ltd". The tradebook says "ASHOKLEY". These
must be linked, generically, for any smallcase.

The engine scores every possible name-to-symbol pair on string similarity plus
corroborating evidence: was the symbol traded on the event where that name
entered the version, does it leave when the name is dropped. It then solves the
assignment as a whole rather than greedily, so one confident match cannot steal
a symbol another name needed more. Any assignment won by a thin margin is
flagged for review instead of being trusted. On your data, no flags.

---

## 4. Reconstructing what the model told you to buy

This is the heart, and the part with the most subtlety.

### 4a. The formula for a fresh investment

    target value for stock i = (portfolio value before + amount added) x weight
    amount to buy            = max(0, target value - what you already hold x price)
    quantity                 = round(amount to buy / price)

The `max(0, ...)` matters. On a top-up, smallcase does not buy every
constituent pro-rata. It pushes the portfolio **towards** the prescribed
weights, so a name already above its weight gets nothing and simply stays
overweight. The naive "amount x weight / price" is right only for the very
first investment and wrong from the second onwards.

### 4b. The formula for a rebalance

    target quantity for stock i = round(portfolio value x weight / price)

Every touched name goes to its target. Dropped names go to zero. A name whose
weight did not change still trades, because prices moved and its actual weight
drifted away.

### 4c. Which price is `price`

This is the question that took the longest to settle, and the answer is that
there are **two different prices** and conflating them is the classic error.

**The price that sets the quantities you were shown** is a live quote at the
moment the order was constructed. Nobody records it. We can get the live price
at that instant from VWAP (volume weighted average price) from the tradebook.
your own execution price for that stock in that event, taken straight from the 
tradebook (the value-weighted average of your fills). Nothing else observable is closer.

**The price the index transitions at** is different. Per smallcase's published
methodology, a rebalance flagged on T0 is applied by the index on T1, the first
trading day after, at that day's (T1's) OHLC average. This was also verified.

So: your quantities came from a live quote. The index's own transaction happens
at T1's OHLC average. Both are true, and they are used for different jobs.


---

## 5. The ±1 share rule, in full

**The problem.** The engine computes what smallcase WHOULD have recommended.
It compares with what user executed. They rarely match to the share. The
question is whether a mismatch means *you changed the order* or *the engine
cannot see the price smallcase used* (the actual live price at the instant
when order was placed).

Getting this wrong in either direction is bad. Call every mismatch a
modification and the tool accuses you of edits you never made. Ignore every
mismatch and a genuine modification vanishes.

The quantity is `round(target / price)`. (@@) 
Rounding sits on a knife edge at .5. If the true live price was 12.05 and the engine
uses 12.02, the division can land on the other side of that edge and the answer 
moves by a share. On a low-priced stock a few paise is enough.

**The test.** A deviation is treated as the model's own arithmetic, and the
executed quantity is adopted as the model quantity, if **any** of these explain it:

1. **Within ±1 share.** The integer rounding boundary.

2. **Reproducible inside that day's traded range.** Take the day's actual low
   and high. Feed each into the same formula (@@). If the executed quantity falls
   anywhere between the two answers, then some price that genuinely traded that
   day produces it, so the live quote explains it. This catches cases ±1 misses.

3. **The cash-balancer leg.** Found empirically in my data, not assumed.
   GOLDBEES does not trade to its weight at a rebalance. It absorbs whatever
   cash the equity legs leave over, so the rebalance nets close to zero. In
   one event the equity residual implied 272.3 units and I traded 274. The
   test: does this leg's cash approximately cancel the net cash of all the other
   legs? If yes, it is the balancer and its quantity is structural. Residuals
   under about ₹2,500 are simply not traded at all.

4. **A tiny untraded leg.** The model says buy something worth under ₹2,500 and
   nothing was bought. Not a modification, just below the threshold where an
   order is worth placing.

Anything none of these explain is labelled **user modification** and its cash
effect is carried as quantity drift.

**On my data:** 202 legs. 166 exact. 20 within ±1. 14 explained by the day's
range. 2 balancer. **Zero modifications, so quantity drift is exactly ₹0** —
which is the correct answer, because I never modified an order.

**What this rule cannot do.** It cannot detect a modification that happens to
land inside the day's traded range. If you deliberately bought 12 shares where
the model said 10 and the price range makes 12 reachable, it is absorbed. The
rule is deliberately biased towards not accusing you. That is the right bias
for a tool that tells you about your own behaviour, but it is a bias, and it
should be stated.

---

## 6. The three books

Three different portfolios are computed, and keeping them distinct is what
makes the final reconciliation honest.

**Book A: the strategy index.** The official series, rebased to 100 on your
first investment date. One notional lumpsum. Price return only, no dividends,
no costs. Yours: **+1.34%**.

**Book A′: the strategy on your cash flows.** For each event, take the net cash
and grow it by the index from that date to today:

    model value today = net cash x (index today / index on that date)

This is the like-for-like benchmark. Yours: **+0.73%**. It is lower than 1.34%
because your largest tranche, ₹6.25 lakh on 2 January, went in when the index
was at its highest point in the window. Under the strategy that money would
have lost 0.92%. That is contribution timing, and it is yours, not the
strategy's.

**Book M: the reconstructed model share book.** A real portfolio simulated on
your cash flows: whole shares, index price conventions, self-financing
rebalances. It differs from A′ because the index holds fractional units while a
real portfolio buys whole shares and carries small cash residuals. Worth
**+₹2,385** on your data.

**Book B: you.** Realised plus unrealised plus dividends, from your actual
trades. Yours: **+4.58%** before costs.

---

## 7. Costs, dividends and tax, kept separate

**Costs.** Per-trade charges are computed from a rate card, then calibrated so
each head sums exactly to the total your broker reports. Calibration factors
are output, so a wrong rate shows up instead of hiding. smallcase fees and DP
charges are matched from the ledger by keyword and by scrip-and-date.
Subscription is your input. Yours: **₹21,873** total, of which ₹10,000 is
subscription and ₹9,210 is STT.

**Dividends.** Read from the statement and attributed pro-rata to the quantity
the smallcase held before the ex-date. A dividend on shares you bought yourself
attributes to zero. Yours: ₹7,059.80 attributed out of ₹7,345 on the smallcase
page. The ₹285 difference is shown, not forced.

**Tax.** FIFO across all units of a scrip, because a demat account is fungible
and the tax office does not know which lot the smallcase ordered. Same-day
offsetting legs are netted as intraday first, matching how your broker computes
it. This reproduced your Zerodha realised P&L to two paise.

The engine then **stops**. It reports realised gains and losses by financial
year, asset class and holding term, and applies **no rates**. Gold and silver
ETFs are marked non-equity because their treatment differs from shares. The
rate depends on your own slab and your full tax position, neither of which the
files contain.

---

## 8. The reconciliation

The gap between A′ and B is bridged in named steps that sum exactly.

| Step | Your data |
|---|---:|
| A′ strategy index on your cash flows | ₹16,221 |
| + whole shares instead of index units | +₹2,385 |
| = reconstructed model share book | ₹18,606 |
| + your prices and quantities vs the model | +₹70,712 |
| = your pre-cost P&L at exchange closes | ₹89,318 |
| + broker's closing prices vs the exchange's | +₹5,084 |
| + dividends | +₹7,060 |
| **= B your total return** | **₹101,461** |

Gap ₹85,240, equal to +3.85 percentage points of the ₹22.15 lakh you put in.
Every step comes from an existing engine output. Nothing is left over.

Three of these deserve a note:

**Whole shares instead of index units (₹2,385).** The index holds fractional
units and is always exactly on weight. A real book cannot be. This is the price
of being real, and it is small.

**Broker's closes vs the exchange's (₹5,084).** Your P&L statement and the NSE
archive disagree on the valuation-day close for a few stocks, mostly Aurobindo
at 1658 against 1622.10. Neither is wrong; they are different snapshots. 0.2%
of the portfolio. Shown rather than papered over.

**Your prices and quantities (₹70,712).** Measured at the prices you traded,
the same advantage is ₹80,155. The difference is that a rupee gained in
December has since been moved by the market. Both figures appear, connected.

**Per-stock**, this step splits as your book's P&L per symbol minus the model
book's, on identical closing prices, so the parts sum to the whole exactly.
ASHOKLEY +0.47 pp, SHRIRAMFIN +0.44, BSE −0.41, CANBK +0.40.

---

## 9. The chart

One bar per event: how much better or worse your fills were than the model's
reference price that day, as a share of everything you invested. The black line
is the running total. There are no points between bars because you did not
trade between them, and inventing a daily line would imply a portfolio history
that was never observed.

The tall April bar is the 9 April rebalance, applied 8 days after the 1 April
model date. You held the old book through the rebound and it is worth about
2.4 pp on its own.

---

## 10. What is weak

Listed honestly, worst first.

**1. Rebalance quantities are not independently reconstructed.** To compute a
target quantity you need the portfolio value on that day, which needs a price
for every holding including ones that did not trade. The broker files do not
have those. The engine therefore infers portfolio value *from the observed
post-trade quantities* — the median across traded legs of quantity x price /
weight. This is circular: the reconstruction is fitted to what you did, so it
validates that the trades are consistent with the prescribed weights, and does
**not** independently prove the quantities. Investment events are not affected;
they are reconstructed from the amount and weights alone.

The headline is insulated. The ₹70,712 comes from the model share book, which
is simulated independently from cash flows and index prices, never from your
quantities. The circularity only touches the leg-level price table and the
modification classification.

**2. Modification detection is deliberately lenient.** Anything reachable
within the day's price range is treated as innocent. It will not catch a small
deliberate edit. Stated in section 5.

**3. The lag matters more than the tool says.** Your +3.85 pp is dominated by
applying rebalances late into a rising market. The same lag in a falling week
would have cost roughly as much. The tool measures what happened; it does not
claim the behaviour is skilful, and the product should not either.

**4. Two conventions are indistinguishable.** T1-OHLC-average and T1-close both
reproduce the index within noise. Documentation decides it. If smallcase ever
publishes otherwise, one line of config changes.

**5. Rebalance pairing can be ambiguous.** An event is paired with the most
recent rebalance flag on or before its trade date. If you applied a rebalance so
late that another was already flagged, the pairing would be wrong. A check now
runs and flags this. Zero on your data.

**6. Single-user assumptions in the demo.** The demo runs with a basket
threshold of 4 because it holds only four names. Real smallcases use 5. This is
a config value, not a fork in the logic.

**7. The subscription is a number you type.** No invoice is parsed. If you type
nothing, it is zero and the net figure flatters you.

---

## 11. Fixed during this review

**A cross-user cache collision.** The analysis cache was keyed on tradebook
filename plus size. Streamlit's resource cache is shared across sessions, so two
different people uploading files that happened to match on both could have been
served each other's results. Now keyed on a SHA-256 of the actual bytes.

**A stale personal default.** `Config.subscription_fee` defaulted to ₹10,000,
your number, which would have silently applied to anyone using the engine
directly. Now zero.

**Ambiguous rebalance pairing** now raises a flag rather than passing silently.

Earlier in the build, two more: the trading calendar was being taken from
whatever dates the price file happened to contain, so a sparse or holiday-padded
file shifted the T1 reference date; it now comes from the index timeline. And
NSE's archive serves the previous session's file on a market holiday, so a
date-range fetch silently collected 11 stale duplicate days. Both fixed, and the
correction moved the price figure from ₹83,084 to ₹80,155.

---

## 12. What is verified, and how

| Claim | Evidence |
|---|---|
| Trade attribution is right | 447 in, 9 out; independent GOLDBEES excluded correctly |
| Reconstruction of the smallcase page | Money Put In and Current Investment to ₹1, Realized Returns exact |
| FIFO tax base | matches Zerodha's realised P&L to ₹0.002 |
| Index methodology | independent replication, 0.15% mean error, 9 transitions, out of sample |
| Per-leg attribution identity | holds to 1e-11 on every leg |
| Waterfall | sums to the gap exactly, no residual |
| Per-stock split | sums to its step exactly |
| Sparse price data | 403 lookups on 18 days reproduce the full 10,716-row result identically |
| Unit tests | 3, covering the top-up formula, the reconciliation identity, deferred legs |

---

## 13. If you take this further

In rough order of value:

- **Benchmark.** Nifty over the same cash flows. One more index series through
  the same A′ machinery.
- **Break the circularity.** Fetch prices for all held names on rebalance dates
  and compute portfolio value directly instead of inferring it. The price plan
  already knows which dates; it is a small extension.
- **Risk, not just return.** The lag that earned +3.85 pp is exposure. Show its
  size, not only its outcome.
- **Parse the subscription** from the smallcase invoice rather than asking.
