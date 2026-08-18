"""Tunable parameters. Nothing here is specific to any one smallcase or investor."""
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RateCard:
    """
    Statutory / broker rate card used to derive per-trade charges.

    Rates are applied per trade and then CALIBRATED so that each charge head sums
    exactly to the account-level total reported by the broker (see costs.py).
    Calibration factors are reported, so a rate that is materially wrong is visible
    rather than silent.
    """
    stt_delivery_buy: float = 0.001          # 0.1% of buy turnover, equity delivery
    stt_delivery_sell: float = 0.001         # 0.1% of sell turnover, equity delivery
    stt_etf_buy: float = 0.0                 # equity-oriented ETFs: no STT on buy
    stt_etf_sell: float = 0.00001            # 0.001% on sell
    exchange_txn: Dict[str, float] = field(default_factory=lambda: {
        "NSE": 0.0000297,                    # 297 per crore
        "BSE": 0.0000375,
    })
    sebi_turnover: float = 0.000001          # Rs 10 per crore, both sides
    ipft: float = 0.0000001                  # NSE investor protection fund trust
    stamp_duty_buy: float = 0.00015          # 0.015% of buy turnover
    gst_rate: float = 0.18                   # on brokerage + exchange + sebi + ipft
    brokerage_delivery: float = 0.0          # Zerodha equity delivery
    # Substrings that identify an ETF from the instrument symbol, lowercase.
    etf_symbol_markers: tuple = ("bees", "etf", "ietf", "nifty1", "liquid")


@dataclass
class TaxRules:
    """
    Indian listed-equity capital gains, STT-paid (Sections 111A / 112A).
    Verified against public guidance for FY 2026-27; Budget 2026 left equity rates
    unchanged from the Finance (No. 2) Act 2024 regime effective 23 July 2024.
    Override these if the analysis period straddles a rate change.
    """
    long_term_months: int = 12
    stcg_rate: float = 0.20                  # s.111A
    ltcg_rate: float = 0.125                 # s.112A
    ltcg_annual_exemption: float = 125000.0  # per financial year, per taxpayer
    cess_rate: float = 0.04                  # health & education cess on tax
    fy_start_month: int = 4                  # Indian financial year starts April
    # ---- non-equity-oriented listed ETFs (gold, silver): s.112, not 111A/112A
    nonequity_long_term_months: int = 12
    nonequity_stcg_rate: float = 0.30        # SLAB-RATE PLACEHOLDER - depends on
                                             # the investor's income; flagged as
                                             # an assumption in the output
    nonequity_ltcg_rate: float = 0.125       # s.112, listed, no indexation
    # the Rs 1.25 lakh exemption belongs to s.112A and applies to the equity
    # class only; non-equity LTCG gets no exemption


@dataclass
class Config:
    # ---- event detection -------------------------------------------------
    basket_min_symbols: int = 5
    """A leg of at least this many distinct symbols at ONE execution second
    qualifies a cluster as a smallcase order on its own. Inspection of the
    validation data set the default: every invest event fires 20+ legs
    simultaneously, which no manual order can."""

    pair_leg_min_symbols: int = 2
    """A cluster whose largest leg is below basket_min_symbols still qualifies
    if it contains BOTH a buy leg and a sell leg, each of at least this many
    simultaneous symbols, within the merge window. Rationale: a small rebalance
    can touch as few as two names per side (the 2026-01-02 rebalance in the
    validation data is exactly [2 sell, 2 buy]), and two multi-symbol legs on
    opposite sides seconds apart cannot be produced manually. A bare small leg
    with no opposite side does NOT qualify and is listed for review."""

    event_merge_seconds: int = 10
    """Sell leg and buy leg of one rebalance land a couple of seconds apart.
    Keep this well below the gap between genuinely separate orders."""

    # ---- reconstruction --------------------------------------------------
    qty_tolerance_shares: int = 1
    qty_tolerance_pct: float = 0.01
    """A reconstructed quantity counts as consistent if it is within 1 share or
    1% of the observed quantity. Exact reproduction is impossible because
    smallcase sizes orders off a live quote we cannot observe; we only see the
    execution price."""

    quantity_snap: bool = True
    """User rule: a reconstructed-vs-executed quantity deviation is treated as
    the model's own mechanics (and the executed quantity adopted as the model
    quantity) when it is explainable WITHOUT user modification:
      1. within +/-1 share (integer rounding boundary), or
      2. reproducible with some price inside that day's traded low-high range
         (the live order-time quote is unobservable; a few paise flips the
         rounding on low-priced stocks), or
      3. the leg is the basket's cash balancer: its cash approximately equals
         the other legs' net cash residual (empirically, the gold ETF absorbs
         the equity residual so a rebalance nets to ~zero), or
      4. an untraded model leg whose value is below no_trade_value_threshold
         (tiny residual targets are simply not traded).
    Anything larger is classified a user modification and carried as quantity
    drift. Set False to disable snapping entirely."""

    balancer_abs_tol: float = 2500.0
    balancer_rel_tol: float = 0.15
    no_trade_value_threshold: float = 2500.0

    repair_window_days: int = 5
    """A smallcase leg can fail on the day (circuit limit, illiquidity) and be
    repaired by a standalone order over the following days. A standalone trade
    is attached to an event as a deferred leg only if the event shows a model
    shortfall in that symbol and direction, the trade falls within this window,
    and its quantity does not exceed the shortfall (with tolerance)."""

    repair_qty_tolerance: float = 0.10
    """Deferred-leg quantity may exceed the shortfall by this fraction."""

    round_amount_grid: tuple = (100000, 50000, 25000, 10000, 5000, 1000)
    """Candidate 'round' investment amounts a human is likely to have typed.
    Used only as a reported sensitivity, never as the primary basis."""

    # ---- entity resolution ----------------------------------------------
    mapping_overrides: Dict[str, str] = field(default_factory=dict)
    """constituent name -> broker symbol. Anything supplied here is taken as
    ground truth and excluded from automatic assignment."""

    mapping_margin_warn: float = 0.25
    """Assignments won by less than this score margin are reported for review
    instead of being trusted silently."""

    # ---- reference data --------------------------------------------------
    rates: RateCard = field(default_factory=RateCard)
    tax: TaxRules = field(default_factory=TaxRules)

    smallcase_fee_keywords: tuple = ("smallcase",)
    dp_charge_keywords: tuple = ("dp charges",)
    amc_keywords: tuple = ("amc",)

    valuation_date: Optional[str] = None
    """Date the 'current' prices represent. Informational only."""

    # ---- v2: gap decomposition ----------------------------------------
    model_invest_price_field: str = "close"
    """Model book's reference for invest events (the strategy's EOD basis)."""

    model_rebalance_price_field: str = "ohlc_avg"
    """smallcase's index applies a rebalance on T+1 at that day's OHLC average
    (published return-calculation methodology)."""

    nonequity_etf_markers: tuple = ("gold", "silver")
    """Symbols containing these are taxed as non-equity listed ETFs."""

    apply_tax_rates: bool = False
    """The engine establishes realised gains/losses, term and asset class; it
    does NOT apply tax rates by default. Enable only with rates you supply."""

    subscription_fee: float = 10000.0
    """Manager subscription for the analysis period. CONFIGURED PLACEHOLDER -
    the eventual product takes this from the user / actual invoices."""

    dividends_override: Optional[float] = None
    """Dividends received, if known from outside the broker files (e.g. the
    smallcase page). Used when no dividend report file is supplied. Kept
    strictly out of the execution/timing attribution."""

    yahoo_symbol_overrides: dict = field(default_factory=dict)
    """broker symbol -> Yahoo symbol, where 'SYMBOL.NS' is not correct."""
