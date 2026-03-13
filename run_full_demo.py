#!/usr/bin/env python3
"""
Energy Commodities Systematic Trading Platform -- Full Demo Pipeline
Usage: python run_full_demo.py
"""

import sys
import time
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import ENERGY_PRODUCTS, RESULTS_DIR, DATA_DIR, PLOT_DIR, DEFAULT_VALUATION_DATE
from module_a_curves.data_loader import CommodityDataLoader
from module_a_curves.curve_bootstrapper import (
    ForwardCurve, ForwardCurveBootstrapper, FuturesSettlement,
)
from module_a_curves.seasonal_model import SeasonalForwardCurve
from module_b_trading.futures_pricer import FuturesPricer, FuturesPosition, CONTRACT_SPECS
from module_b_trading.risk_analytics import RiskAnalytics
from module_b_trading.scenario_engine import ScenarioEngine, STANDARD_SCENARIOS
from module_b_trading.alpha_signals import CompositeAlphaModel
from module_b_trading.rfq_generator import RFQGenerator
from module_b_trading.win_probability import WinProbabilityModel
from module_b_trading.quote_optimizer import QuoteOptimizer
from module_b_trading.carry_rolldown import RollYieldCalculator
from module_b_trading.markout_pnl import MarkoutAnalyzer
from module_c_execution.market_impact import AlmgrenChrissModel, ENERGY_FUTURES
from module_c_execution.execution_scheduler import TWAPScheduler, VWAPScheduler, AdaptiveScheduler
from module_c_execution.order_simulator import OrderSimulator
from shared.kdb_interface import KDBInterface, KDBConfig
from shared.databento_loader import DatabentoLoader, DatabentoConfig
from shared.plot_style import apply_style, COLOURS

SHOW_PLOTS = True

start_time = time.time()
valuation_date = date.fromisoformat(DEFAULT_VALUATION_DATE)

print("\n" + "=" * 72)
print("  ENERGY COMMODITIES SYSTEMATIC TRADING PLATFORM")
print("  Full Demo Pipeline -- Steps 0 through 7")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════
# STEP 0: Environment Setup
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 0: ENVIRONMENT SETUP\n{'='*72}\n")

for d in [RESULTS_DIR, DATA_DIR, PLOT_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  Directory ready: {d}")

if SHOW_PLOTS:
    apply_style()

print(f"\n  Valuation date: {valuation_date}")
print(f"  Products: {list(ENERGY_PRODUCTS.keys())}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 1: DATA LOADING -- EIA INVENTORY & FUTURES STRIPS\n{'='*72}\n")

from config import get_databento_api_key as _get_db_key, get_eia_api_key as _get_eia_key
loader = CommodityDataLoader(
    eia_api_key=_get_eia_key(),
    databento_api_key=_get_db_key(),
)

print("--- Futures Strip Data ---")
strips = {}
for product in ["CL", "HO", "RB", "NG"]:
    strip_data = loader.get_strip_for_date(date=DEFAULT_VALUATION_DATE, product=product)
    strips[product] = strip_data
    if strip_data and len(strip_data) > 0:
        print(f"  {product}: {len(strip_data)} contract months loaded")
        if len(strip_data) >= 2:
            print(f"    Front: {strip_data[0]['settlement']:.2f}  "
                  f"Back: {strip_data[-1]['settlement']:.2f}")
    else:
        print(f"  {product}: No data available (set EIA_API_KEY or populate cache)")

print("\n--- EIA Spot Prices ---")
spot_prices = {}
for product in ["CL", "HO", "RB", "NG"]:
    spot = loader.fetch_spot_price(product=product, date=DEFAULT_VALUATION_DATE)
    if spot is not None:
        spot_prices[product] = spot
        print(f"  {product}: ${spot:.4f}")
    else:
        print(f"  {product}: No spot data (using front futures as proxy)")

print("\n--- EIA Inventory Data ---")
inventory = loader.fetch_inventory_history(end=DEFAULT_VALUATION_DATE)
if inventory is not None and len(inventory) > 0:
    print(f"  Loaded {len(inventory)} inventory observations")
else:
    print("  No inventory data available (set EIA_API_KEY for live data)")

print("\n--- Natural Gas Storage ---")
ng_storage = loader.fetch_ng_storage_history(end=DEFAULT_VALUATION_DATE)
if ng_storage is not None and len(ng_storage) > 0:
    print(f"  Loaded {len(ng_storage)} NG storage observations")
    print(f"  Latest: {ng_storage.iloc[-1]['ng_storage_bcf']:.0f} Bcf")
else:
    print("  No NG storage data available (set EIA_API_KEY for live data)")

print("\n--- Inventory Z-Scores (Petroleum + NG Storage) ---")
try:
    zscores = loader.get_inventory_zscore(date=DEFAULT_VALUATION_DATE)
    for series, z in zscores.items():
        print(f"  {series}: z-score = {z:+.2f}")
except Exception as e:
    print(f"  Z-score calculation: {e}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Forward Curve Bootstrapping & Seasonal Decomposition
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 2: FORWARD CURVE BOOTSTRAPPING & SEASONAL DECOMPOSITION\n{'='*72}\n")

bootstrapper = ForwardCurveBootstrapper()
curves = {}

print("--- Forward Curve Construction ---")
for product in ["CL", "HO", "RB", "NG"]:
    strip = strips.get(product)
    if strip and len(strip) > 0:
        settlements = [
            FuturesSettlement(
                product=item["product"],
                contract_code=item["contract_code"],
                settlement=item["settlement"],
                time_to_expiry=item["time_to_expiry"],
            )
            for item in strip
        ]
        spot = spot_prices.get(product)  # None if EIA spot unavailable
        curve = bootstrapper.bootstrap(
            settlements, valuation_date=DEFAULT_VALUATION_DATE,
            product=product, spot_price=spot,
        )
    else:
        times = [(i + 1) / 12 for i in range(12)]
        # Realistic synthetic curves: steep front contango flattening out,
        # with product-specific shapes
        _synth = {
            "CL": (72.50, [0.00, 0.80, 1.30, 1.55, 1.65, 1.68, 1.60, 1.45, 1.20, 0.85, 0.40, -0.10]),
            "HO": (2.35,  [0.00, 0.04, 0.06, 0.05, 0.02, -0.02, -0.06, -0.10, -0.13, -0.15, -0.16, -0.15]),
            "RB": (2.45,  [0.00, 0.06, 0.10, 0.11, 0.08, 0.03, -0.04, -0.10, -0.15, -0.18, -0.19, -0.18]),
            "NG": (3.80,  [0.00, -0.12, -0.20, -0.22, -0.15, -0.03, 0.15, 0.38, 0.55, 0.60, 0.48, 0.30]),
        }
        base, offsets = _synth.get(product, (72.50, [0.10 * i for i in range(12)]))
        prices = [base + o for o in offsets]
        spot = spot_prices.get(product)
        curve = ForwardCurve(times, prices, product=product,
                             spot_price=spot if spot is not None else prices[0])

    curves[product] = curve
    spot_label = f"spot={curve.spot_price:.4f}" if product in spot_prices else f"spot(proxy)={curve.spot_price:.4f}"
    print(f"  {product}: {len(curve.times)} tenors, "
          f"front={curve.forward_price(curve.times[0]):.4f}, "
          f"{spot_label}, "
          f"contango={'Yes' if curve.is_contango() else 'No'}")

cl_curve = curves["CL"]

print("\n--- Convenience Yield Extraction (CL) ---")
for t in [0.25, 0.5, 1.0]:
    cy = cl_curve.convenience_yield(t)
    print(f"  CY at {t:.2f}y: {cy*100:.2f}%")

print("\n--- Seasonal Decomposition ---")
try:
    seasonal = SeasonalForwardCurve(cl_curve)
    times_arr = np.array(cl_curve.times)
    prices_arr = np.array([cl_curve.forward_price(t) for t in cl_curve.times])
    seasonal.calibrate(times_arr, prices_arr)

    pattern = seasonal.extract_seasonal_pattern(n_points=12)
    print(f"  Seasonal coefficients calibrated")
    print(f"  Seasonal range: {pattern['seasonal_adjustment'].min():.2f} to "
          f"{pattern['seasonal_adjustment'].max():.2f}")
except Exception as e:
    print(f"  Seasonal calibration: {e}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Risk Analytics & Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 3: RISK ANALYTICS & SCENARIO ANALYSIS\n{'='*72}\n")

cl_strip = strips.get("CL", [])
if cl_strip:
    cl_settlements = [
        FuturesSettlement(
            product=s["product"], contract_code=s["contract_code"],
            settlement=s["settlement"], time_to_expiry=s["time_to_expiry"],
        ) for s in cl_strip
    ]
else:
    cl_settlements = [
        FuturesSettlement(
            product="CL", contract_code=f"CL{i+1:02d}",
            settlement=cl_curve.forward_price(t), time_to_expiry=t,
        ) for i, t in enumerate(cl_curve.times)
    ]

pricer = FuturesPricer(cl_curve)
risk = RiskAnalytics(bootstrapper, cl_settlements)

print("--- Position Setup ---")
portfolio = [
    FuturesPosition("CLK26", "CL", 50, "long", 72.50, "2026-04-20"),
    FuturesPosition("CLN26", "CL", 30, "long", 73.20, "2026-06-22"),
    FuturesPosition("CLZ26", "CL", 20, "short", 74.50, "2026-11-20"),
    FuturesPosition("HOK26", "HO", 15, "long", 2.35, "2026-04-30"),
    FuturesPosition("NGK26", "NG", 25, "long", 3.80, "2026-04-28"),
]
for pos in portfolio:
    print(f"  {pos}")

print("\n--- Portfolio Mark-to-Market ---")
mtm_df = pricer.portfolio_mtm(portfolio)
print(mtm_df.to_string(index=False))
print(f"\n  Total CL MTM: ${mtm_df['mtm_usd'].sum():,.0f}")

print("\n--- Parallel Delta (CL curve) ---")
try:
    total_delta = 0.0
    for pos in portfolio:
        delta = risk.parallel_delta(pos)
        total_delta += delta
        print(f"  {pos.ticker}: delta = ${delta:,.0f}")
    print(f"  Portfolio parallel delta: ${total_delta:,.0f}")
except Exception as e:
    print(f"  Delta calculation: {e}")

print("\n--- Scenario Analysis ---")
scenario_engine = ScenarioEngine(bootstrapper, cl_settlements)
for name, scenario in list(STANDARD_SCENARIOS.items())[:4]:
    try:
        results = scenario_engine.run_scenario(scenario, portfolio)
        total_pnl = results["scenario_pnl"].sum() if "scenario_pnl" in results.columns else 0
        print(f"  {name:25s} -> P&L: ${total_pnl:>12,.0f}")
    except Exception as e:
        print(f"  {name:25s} -> Error: {e}")

print("\n--- Roll Yield Analysis ---")
roll_calc = RollYieldCalculator(cl_curve)
matrix = roll_calc.roll_yield_matrix("CL")
if len(matrix) > 0:
    print(matrix[["front_tenor", "back_tenor", "roll_yield_ann", "regime"]].to_string(index=False))
    best = roll_calc.best_roll_trades("CL", top_n=3)
    if (best["roll_yield_ann"] < 0).all():
        label = "Largest carry cost (contango)"
    elif (best["roll_yield_ann"] > 0).all():
        label = "Best carry trades (backwardation)"
    else:
        label = "Top roll-yield trades"
    print(f"\n  {label}:")
    for _, row in best.iterrows():
        print(f"    {row['front_tenor']:.2f}y -> {row['back_tenor']:.2f}y: "
              f"{row['roll_yield_ann']*100:+.1f}% ({row['regime']})")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Alpha Signal Generation
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 4: ALPHA SIGNAL GENERATION\n{'='*72}\n")

print("--- Composite Alpha Model (CL) ---")
alpha_model = CompositeAlphaModel.default_crude_model()

front_price = cl_curve.forward_price(cl_curve.times[0])
deferred_price = cl_curve.forward_price(cl_curve.times[-1])

alpha_result = alpha_model.compute_composite(
    forward_curve=cl_curve,
    front_price=front_price,
    deferred_price=deferred_price,
    inventory_level=440.0,
    month=3,
    price=front_price,
    cl_price=front_price,
    ho_price=2.35,
    rb_price=2.45,
)

print("  Signal breakdown:")
for name, val in alpha_result.items():
    bar = "+" * max(0, int(val * 5)) + "-" * max(0, int(-val * 5))
    print(f"    {name:20s}: {val:+.3f}  {bar}")

composite = alpha_result["composite"]
if composite > 0.5:
    print(f"\n  >> BULLISH bias ({composite:+.2f}): consider adding long CL exposure")
elif composite < -0.5:
    print(f"\n  >> BEARISH bias ({composite:+.2f}): consider reducing/shorting CL")
else:
    print(f"\n  >> NEUTRAL ({composite:+.2f}): no strong directional signal")

print("\n--- Composite Alpha Model (NG) ---")
ng_alpha_model = CompositeAlphaModel.default_ng_model()

ng_curve = curves.get("NG")
ng_front = ng_curve.forward_price(ng_curve.times[0]) if ng_curve else 3.80
ng_deferred = ng_curve.forward_price(ng_curve.times[-1]) if ng_curve else 4.10

# Use actual NG storage level if available, otherwise a reasonable default
ng_storage_level = 1800.0  # default March Bcf
if ng_storage is not None and len(ng_storage) > 0:
    ng_storage_level = ng_storage.iloc[-1]["ng_storage_bcf"]

ng_alpha_result = ng_alpha_model.compute_composite(
    forward_curve=ng_curve,
    front_price=ng_front,
    deferred_price=ng_deferred,
    ng_storage_level=ng_storage_level,
    month=3,
    price=ng_front,
)

print("  Signal breakdown:")
for name, val in ng_alpha_result.items():
    bar = "+" * max(0, int(val * 5)) + "-" * max(0, int(-val * 5))
    print(f"    {name:20s}: {val:+.3f}  {bar}")

ng_composite = ng_alpha_result["composite"]
if ng_composite > 0.5:
    print(f"\n  >> BULLISH bias ({ng_composite:+.2f}): consider adding long NG exposure")
elif ng_composite < -0.5:
    print(f"\n  >> BEARISH bias ({ng_composite:+.2f}): consider reducing/shorting NG")
else:
    print(f"\n  >> NEUTRAL ({ng_composite:+.2f}): no strong directional signal")


# ═══════════════════════════════════════════════════════════════════════
# STEP 5: RFQ Generation & Quote Optimisation
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 5: RFQ GENERATION, WIN PROBABILITY & QUOTE OPTIMISATION\n{'='*72}\n")

print("--- Generate RFQ Flow ---")
rfq_gen = RFQGenerator(seed=42, rfqs_per_day=50)
rfqs = rfq_gen.generate_batch(n=50)
print(f"  Generated {len(rfqs)} RFQs")
print(f"  Product mix: {rfqs['product'].value_counts().to_dict()}")
print(f"  Segment mix: {rfqs['client_segment'].value_counts().to_dict()}")

print("\n--- Win Probability Model ---")
win_model = WinProbabilityModel()
sample = rfqs.head(5).copy()
sample["spread_bps"] = 0.5
sample["volatility"] = 0.25
probs = win_model.predict_proba(sample)
print(f"  Sample win probabilities (at 0.5bps): {[f'{p:.1%}' for p in probs]}")

print("\n--- Quote Optimisation ---")
optimizer = QuoteOptimizer(win_model=win_model)
mid_prices = {"CL": 72.50, "HO": 2.35, "RB": 2.45, "NG": 3.80}
quotes = optimizer.optimize_batch(rfqs, mid_prices)
print(f"  Optimised {len(quotes)} quotes")
print(f"  Avg win probability: {quotes['win_probability'].mean():.1%}")
print(f"  Total E[PnL]: ${quotes['expected_pnl'].sum():,.0f}")
print(f"\n  By product:")
for product in ["CL", "HO", "RB", "NG"]:
    mask = quotes["product"] == product
    if mask.any():
        sub = quotes[mask]
        print(f"    {product}: {len(sub)} quotes, "
              f"avg spread={sub['spread'].mean():.4f}, "
              f"E[PnL]=${sub['expected_pnl'].sum():,.0f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: Execution Planning & Market Impact
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 6: EXECUTION PLANNING & MARKET IMPACT ANALYSIS\n{'='*72}\n")

print("--- Almgren-Chriss Impact Model ---")
impact_model = AlmgrenChrissModel()
for product, size in [("CL", 100), ("HO", 50), ("RB", 50), ("NG", 75)]:
    est = impact_model.estimate_impact(product, size)
    print(f"  {product} {size} lots: cost=${est.total_cost_usd:,.0f} "
          f"({est.cost_bps:.1f}bps), participation={est.participation_rate:.1%}")

print("\n--- Strategy Comparison (CL 200 lots) ---")
comparison = impact_model.compare_strategies("CL", 200)
print(comparison.to_string(index=False))

print("\n--- Optimal Execution Trajectory ---")
trajectory = impact_model.optimal_trajectory("CL", 200, n_slices=10)
for i, frac in enumerate(trajectory):
    bar = "#" * int(frac * 40)
    print(f"  Slice {i:2d}: {frac:5.1%} {bar}")

print("\n--- Execution Schedules ---")
for SchedulerClass, name in [(TWAPScheduler, "TWAP"), (VWAPScheduler, "VWAP"),
                              (AdaptiveScheduler, "Adaptive")]:
    if name == "Adaptive":
        scheduler = SchedulerClass(urgency=0.7)
    else:
        scheduler = SchedulerClass()

    sched = scheduler.schedule("CL", 100) if name == "VWAP" else scheduler.schedule("CL", 100, n_slices=10)
    df = sched.to_dataframe()
    total = df["target_contracts"].sum()
    print(f"\n  {name} ({sched.strategy}):")
    print(f"    Slices: {len(df)}, Total contracts: {total}")
    if len(df) > 0:
        print(f"    First: {df.iloc[0]['time_label']} ({df.iloc[0]['target_contracts']} lots)")
        print(f"    Last:  {df.iloc[-1]['time_label']} ({df.iloc[-1]['target_contracts']} lots)")

print("\n--- Order Book Simulation ---")
sim = OrderSimulator(seed=42)
book = sim.generate_book("CL", 72.50)
print(f"  CL book: mid={book.mid_price:.2f}, spread={book.spread:.4f}")
print(f"  Top of book: bid={book.bids[0].price:.2f}x{book.bids[0].size} "
      f"/ ask={book.asks[0].price:.2f}x{book.asks[0].size}")

exec_df = sim.simulate_execution("CL", "buy", 50, 72.50, n_slices=5)
if len(exec_df) > 0:
    vwap = (exec_df["price"] * exec_df["size"]).sum() / exec_df["size"].sum()
    avg_slip = (exec_df["slippage"] * exec_df["size"]).sum() / exec_df["size"].sum()
    print(f"\n  Execution result: VWAP={vwap:.4f}, avg slippage={avg_slip:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: Portfolio Summary & KDB+ Storage
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 7: PORTFOLIO SUMMARY & KDB+ STORAGE\n{'='*72}\n")

print("--- KDB+ Storage ---")
from config import KDB_HOST, KDB_PORT
import subprocess, socket

def _kdb_listening(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False

if not _kdb_listening(KDB_HOST, KDB_PORT):
    print(f"  Starting KDB+ on {KDB_HOST}:{KDB_PORT}...")
    import os
    env = os.environ.copy()
    env["QHOME"] = "C:/q"
    try:
        subprocess.Popen(
            ["C:/q/w64/q.exe", "-p", str(KDB_PORT)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import time as _t
        _t.sleep(2)
        print(f"  KDB+ server started (pid launched)")
    except Exception as e:
        print(f"  Could not start KDB+: {e}")

try:
    kdb = KDBInterface(KDBConfig(host=KDB_HOST, port=KDB_PORT))
    kdb.create_tables()

    fwd_rows = []
    for t in cl_curve.times:
        fwd_rows.append({
            "dt": str(valuation_date), "product": "CL", "tenor": t,
            "price": cl_curve.forward_price(t),
        })
    kdb.insert_forwards(pd.DataFrame(fwd_rows))
    print(f"  Stored {len(fwd_rows)} forward curve points")

    kdb.insert_inventory(pd.DataFrame([
        {"dt": str(valuation_date), "series": "crude_stocks",
         "val": 440.0, "unt": "MMbbl"},
        {"dt": str(valuation_date), "series": "ng_storage",
         "val": ng_storage_level, "unt": "Bcf"},
    ]))
    print(f"  Stored inventory snapshots (crude + NG storage)")
    print(f"  Table counts: {kdb.table_counts()}")
except Exception as e:
    print(f"  KDB+ connection failed: {e}")
    print(f"  (Requires running KDB+ instance on {KDB_HOST}:{KDB_PORT})")

print("\n--- Databento L2 Data ---")
from config import get_databento_api_key
from shared.databento_loader import front_month_symbol
db_key = get_databento_api_key()
if db_key:
    try:
        db_loader = DatabentoLoader(DatabentoConfig(api_key=db_key))
        db_date = "2025-01-15"
        cl_symbol = front_month_symbol("CL", db_date)
        print(f"  Symbol: {cl_symbol} on {db_date}")
        books = db_loader.load_book_snapshots(cl_symbol, db_date, n_snapshots=50)
        trades = db_loader.load_trades(cl_symbol, db_date, n_trades=200)
        print(f"  Book snapshots: {len(books)} rows")
        print(f"  Trade records: {len(trades)} rows")
        if len(books) > 0:
            best_bid = books["bid_px_00"].iloc[0]
            best_ask = books["ask_px_00"].iloc[0]
            print(f"  First snapshot: bid={best_bid:.2f}, ask={best_ask:.2f}")
    except Exception as e:
        print(f"  Databento API error: {e}")
        print(f"  (Requires valid Databento subscription and available data)")
else:
    print("  Databento API key not set; skipping L2 data load")

print("\n--- Markout P&L Analysis ---")
analyzer = MarkoutAnalyzer()
if quotes is not None and len(quotes) > 0:
    trade_sample = quotes.head(20).copy()
    trade_sample["fill_price"] = trade_sample["mid_price"]
    trade_sample["edge"] = trade_sample["spread"] / 2
    markouts = analyzer.simulate_markouts(trade_sample, seed=42)
    pnl_df = analyzer.compute_markout_pnl(markouts)
    print(f"  Markout P&L computed for {len(trade_sample)} trades")
else:
    print("  No quotes available for markout analysis")

print("\n--- Final Summary ---")
elapsed = time.time() - start_time
print(f"  Valuation date:  {valuation_date}")
print(f"  Products traded: {list(ENERGY_PRODUCTS.keys())}")
print(f"  Curves built:    {len(curves)}")
print(f"  RFQs processed:  {len(rfqs)}")
print(f"  Quotes made:     {len(quotes)}")
print(f"  Elapsed time:    {elapsed:.1f}s")

if SHOW_PLOTS:
    import matplotlib.pyplot as plt
    print("\n--- Generating Plots ---")

    # 1. Forward Curves (2x2 grid — products have very different price scales)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _units = {"CL": "$/bbl", "HO": "$/gal", "RB": "$/gal", "NG": "$/MMBtu"}
    for ax, (product, curve) in zip(axes.flat, curves.items()):
        tenors = curve.times
        prices = [curve.forward_price(t) for t in tenors]
        ax.plot(tenors, prices, 'o-', color=COLOURS[product], linewidth=2, markersize=5)
        ax.fill_between(tenors, prices, min(prices) - (max(prices) - min(prices)) * 0.1,
                         alpha=0.15, color=COLOURS[product])
        pmin, pmax = min(prices), max(prices)
        margin = max((pmax - pmin) * 0.3, pmax * 0.005)
        ax.set_ylim(pmin - margin, pmax + margin)
        ax.set_xlabel("Tenor (years)")
        ax.set_ylabel(f"Price ({_units.get(product, '$/unit')})")
        shape = "contango" if curve.is_contango() else "backwardation"
        ax.set_title(f"{ENERGY_PRODUCTS[product]} ({product}) — {shape}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Energy Forward Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "forward_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved forward_curves.png")

    # 2. Scenario Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    scen_data = []
    scen_labels = []
    for name, scenario in list(STANDARD_SCENARIOS.items())[:6]:
        try:
            results = scenario_engine.run_scenario(scenario, portfolio[:3])
            if "scenario_pnl" in results.columns:
                scen_data.append([results["scenario_pnl"].iloc[i] for i in range(min(3, len(results)))])
                scen_labels.append(name[:30])
        except Exception:
            pass
    if scen_data:
        data = np.array(scen_data)
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
        ax.set_yticks(range(len(scen_labels)))
        ax.set_yticklabels(scen_labels, fontsize=9)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels([p.ticker for p in portfolio[:3]], fontsize=10)
        ax.set_title("Scenario P&L by Position ($)")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"${data[i,j]:,.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "scenario_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scenario_heatmap.png")

    # 3. Execution Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: optimal trajectory
    axes[0].plot(range(len(trajectory)), trajectory, 'o-', color=COLOURS["primary"], linewidth=2)
    axes[0].fill_between(range(len(trajectory)), trajectory, alpha=0.2, color=COLOURS["primary"])
    axes[0].set_xlabel("Time Slice")
    axes[0].set_ylabel("Cumulative Fraction Executed")
    axes[0].set_title("Almgren-Chriss Optimal Trajectory (CL 200 lots)")
    axes[0].grid(True, alpha=0.3)
    # Right: cost vs horizon
    axes[1].plot(comparison["horizon_min"], comparison["cost_bps"], 's-', color=COLOURS["secondary"], linewidth=2)
    axes[1].set_xlabel("Execution Horizon (minutes)")
    axes[1].set_ylabel("Total Cost (bps)")
    axes[1].set_title("Market Impact vs Execution Horizon")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "execution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved execution_analysis.png")

    # 4. Carry & Rolldown
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(matrix) > 0:
        yields = matrix["roll_yield_ann"].values * 100
        labels = [f"{r['front_tenor']:.1f}-{r['back_tenor']:.1f}" for _, r in matrix.iterrows()]
        colors = [COLOURS["accent"] if y > 0 else COLOURS["warning"] for y in yields]
        ax.bar(range(len(yields)), yields, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Tenor Pair (years)")
        ax.set_ylabel("Annualized Roll Yield (%)")
        ax.set_title("CL Roll Yield by Tenor Pair")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "carry_rolldown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved carry_rolldown.png")

    # 5. Alpha Signals
    fig, ax = plt.subplots(figsize=(10, 5))
    signal_names = list(alpha_result.keys())
    signal_vals = list(alpha_result.values())
    colors = [COLOURS["accent"] if v > 0 else COLOURS["warning"] for v in signal_vals]
    ax.barh(signal_names, signal_vals, color=colors, alpha=0.8)
    ax.set_xlabel("Signal Value")
    ax.set_title("Composite Alpha Signal Breakdown (CL)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "alpha_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved alpha_signals.png")

    # 6. Summary Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Panel 1: CL forward curve
    cl_tenors = cl_curve.times
    cl_prices = [cl_curve.forward_price(t) for t in cl_tenors]
    axes[0,0].plot(cl_tenors, cl_prices, 'o-', color=COLOURS["CL"], linewidth=2)
    axes[0,0].set_xlabel("Tenor (years)")
    axes[0,0].set_ylabel("Price ($/bbl)")
    axes[0,0].set_title("WTI Crude Oil Forward Curve")
    axes[0,0].grid(True, alpha=0.3)
    # Panel 2: Alpha signals
    axes[0,1].barh(signal_names, signal_vals, color=colors, alpha=0.8)
    axes[0,1].set_xlabel("Signal Value")
    axes[0,1].set_title("Alpha Signal Breakdown")
    axes[0,1].axvline(x=0, color="black", linewidth=0.8)
    axes[0,1].grid(True, alpha=0.3, axis="x")
    # Panel 3: Win probability distribution
    axes[1,0].hist(quotes["win_probability"], bins=20, color=COLOURS["primary"], alpha=0.7, edgecolor="white")
    axes[1,0].set_xlabel("Win Probability")
    axes[1,0].set_ylabel("Count")
    axes[1,0].set_title(f"Quote Win Probability Distribution (avg={quotes['win_probability'].mean():.1%})")
    axes[1,0].grid(True, alpha=0.3, axis="y")
    # Panel 4: Execution trajectory
    axes[1,1].plot(range(len(trajectory)), trajectory, 'o-', color=COLOURS["secondary"], linewidth=2)
    axes[1,1].fill_between(range(len(trajectory)), trajectory, alpha=0.2, color=COLOURS["secondary"])
    axes[1,1].set_xlabel("Time Slice")
    axes[1,1].set_ylabel("Cumulative Fraction")
    axes[1,1].set_title("Optimal Execution Trajectory (CL)")
    axes[1,1].grid(True, alpha=0.3)
    fig.suptitle("Energy Commodities Systematic Trading Platform", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_dashboard.png")

print("\n" + "=" * 72)
print("  DEMO COMPLETE -- Energy Commodities Systematic Trading Platform")
print("=" * 72 + "\n")
