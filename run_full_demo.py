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
from module_b_trading.scenario_engine import STANDARD_SCENARIOS, ScenarioEngine
from module_b_trading.alpha_signals import CompositeAlphaModel
from module_b_trading.rfq_generator import RFQGenerator
from module_b_trading.win_probability import WinProbabilityModel
from module_b_trading.quote_optimizer import QuoteOptimizer
from module_b_trading.carry_rolldown import RollYieldCalculator
from module_b_trading.markout_pnl import MarkoutAnalyzer
from module_b_trading.hedge_selector import HedgeSelector
from module_c_execution.market_impact import AlmgrenChrissModel, ENERGY_FUTURES
from module_c_execution.execution_scheduler import TWAPScheduler, VWAPScheduler, AdaptiveScheduler
from module_c_execution.order_simulator import OrderSimulator, L2Book
from shared.kdb_interface import KDBInterface, KDBConfig
from shared.databento_loader import DatabentoLoader, DatabentoConfig
from shared.plot_style import apply_style, COLOURS

# C++ kernel
import commodities_cpp as _cpp

SHOW_PLOTS = True

start_time = time.time()
valuation_date = date.fromisoformat(DEFAULT_VALUATION_DATE)

print("\n" + "=" * 72)
print("  ENERGY COMMODITIES SYSTEMATIC TRADING PLATFORM")
print("  Full Demo Pipeline -- Steps 0 through 8")
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

bootstrapper = ForwardCurveBootstrapper(interpolation_method="monotone_convex")

# Primary curves are C++; Python curves kept only for seasonal/carry modules
cpp_curves = {}
cpp_pricers = {}
py_curves = {}  # thin Python wrappers for modules needing Python ForwardCurve API

print("--- Forward Curve Construction (C++ monotone convex) ---")
for product in ["CL", "HO", "RB", "NG"]:
    strip = strips.get(product)
    spot = spot_prices.get(product)

    if strip and len(strip) > 0:
        times = [item["time_to_expiry"] for item in strip]
        prices = [item["settlement"] for item in strip]
    else:
        times = [(i + 1) / 12 for i in range(12)]
        _synth = {
            "CL": (72.50, [0.00, 0.80, 1.30, 1.55, 1.65, 1.68, 1.60, 1.45, 1.20, 0.85, 0.40, -0.10]),
            "HO": (2.35,  [0.00, 0.04, 0.06, 0.05, 0.02, -0.02, -0.06, -0.10, -0.13, -0.15, -0.16, -0.15]),
            "RB": (2.45,  [0.00, 0.06, 0.10, 0.11, 0.08, 0.03, -0.04, -0.10, -0.15, -0.18, -0.19, -0.18]),
            "NG": (3.80,  [0.00, -0.12, -0.20, -0.22, -0.15, -0.03, 0.15, 0.38, 0.55, 0.60, 0.48, 0.30]),
        }
        base, offsets = _synth.get(product, (72.50, [0.10 * i for i in range(12)]))
        prices = [base + o for o in offsets]

    # C++ curve: primary pricing engine
    cpp_fc = _cpp.ForwardCurve(times, prices, "monotone_convex")
    cpp_curves[product] = cpp_fc
    cpp_pricers[product] = _cpp.FuturesPricer(cpp_fc)

    # Python curve: needed by SeasonalForwardCurve, RollYieldCalculator, etc.
    spot_val = spot if spot is not None else prices[0]
    py_curve = ForwardCurve(times, prices, product=product,
                            interpolation_method="monotone_convex",
                            spot_price=spot_val,
                            valuation_date=DEFAULT_VALUATION_DATE)
    py_curves[product] = py_curve

    spot_label = f"spot={spot_val:.4f}" if spot else f"spot(proxy)={spot_val:.4f}"
    front_px = cpp_fc.forward_price(times[0])
    contango = cpp_fc.forward_price(times[-1]) > cpp_fc.forward_price(times[0])
    print(f"  {product}: {len(times)} tenors, "
          f"front={front_px:.4f}, {spot_label}, "
          f"contango={'Yes' if contango else 'No'}  [C++]")

# Legacy alias — some downstream modules reference 'curves'
curves = py_curves
cl_curve = curves["CL"]

print("\n--- Convenience Yield Extraction (CL, C++) ---")
for t in [0.25, 0.5, 1.0]:
    cy = cpp_curves["CL"].convenience_yield(t, cl_curve.spot_price, 0.045, 0.02)
    print(f"  CY at {t:.2f}y: {cy*100:.2f}%")

print("\n--- Seasonal Decomposition (All Products) ---")
seasonal_models = {}
for product, curve in curves.items():
    try:
        seasonal = SeasonalForwardCurve(curve, product=product)
        times_arr = np.array(curve.times)
        prices_arr = np.array([curve.forward_price(t) for t in curve.times])
        seasonal.calibrate(times_arr, prices_arr)
        seasonal_models[product] = seasonal

        pattern = seasonal.extract_seasonal_pattern(n_points=12)
        print(f"  {product}: seasonal range = {pattern['seasonal_adjustment'].min():+.3f} to "
              f"{pattern['seasonal_adjustment'].max():+.3f}")
    except Exception as e:
        print(f"  {product}: Seasonal calibration: {e}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Risk Analytics & Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 3: RISK ANALYTICS & SCENARIO ANALYSIS\n{'='*72}\n")

print(f"  C++ curves: {list(cpp_curves.keys())} "
      f"(method={cpp_curves['CL'].interpolation_method()})")

# Python pricers — only needed for _find_tenor() logic
pricers = {product: FuturesPricer(c) for product, c in curves.items()}

print("\n--- Position Setup ---")
portfolio = [
    FuturesPosition("CLK26", "CL", 50, "long", 72.50, "2026-04-20"),
    FuturesPosition("CLN26", "CL", 30, "long", 73.20, "2026-06-22"),
    FuturesPosition("CLZ26", "CL", 20, "short", 74.50, "2026-11-20"),
    FuturesPosition("HOK26", "HO", 15, "long", 2.35, "2026-04-30"),
    FuturesPosition("NGK26", "NG", 25, "long", 3.80, "2026-04-28"),
]
for pos in portfolio:
    print(f"  {pos}")

print("\n--- Portfolio Mark-to-Market (C++ monotone convex) ---")
mtm_rows = []
for pos in portfolio:
    # Use C++ pricer for this product
    cpp_pricer = cpp_pricers.get(pos.product, cpp_pricers["CL"])
    # Build C++ position
    cpp_pos = _cpp.FuturesPosition()
    cpp_pos.ticker = pos.ticker
    cpp_pos.product = pos.product
    cpp_pos.num_contracts = pos.num_contracts
    cpp_pos.direction = pos.direction_sign
    cpp_pos.entry_price = pos.entry_price
    cpp_pos.contract_size = pos.contract_size
    # Compute tenor using Python pricer (has _find_tenor logic)
    py_pricer = pricers.get(pos.product, pricers["CL"])
    tenor = py_pricer._find_tenor(pos)
    # C++ MTM
    cpp_mtm = cpp_pricer.mark_to_market(cpp_pos, tenor)
    current_price = cpp_curves[pos.product].forward_price(tenor)
    spec = CONTRACT_SPECS.get(pos.product, CONTRACT_SPECS["CL"])
    mtm_rows.append({
        "ticker": pos.ticker, "product": pos.product,
        "direction": pos.direction, "contracts": pos.num_contracts,
        "entry_price": pos.entry_price, "current_price": current_price,
        "mtm_usd": cpp_mtm,
        "mtm_per_contract": cpp_mtm / pos.num_contracts,
        "contract_size": spec["contract_size"], "unit": spec["unit"],
    })
mtm_df = pd.DataFrame(mtm_rows)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print(mtm_df.to_string(index=False))
pd.reset_option("display.float_format")
print(f"\n  Total Portfolio MTM: ${mtm_df['mtm_usd'].sum():,.0f}")

cl_positions = [pos for pos in portfolio if pos.product == "CL"]

# ── Hedge Selection & Sizing ────────────────────────────────────────
print("\n--- Hedge Selection & Sizing ---")
hedge_mid_prices = {"CL": 72.50, "HO": 2.35, "RB": 2.45, "NG": 3.80}
for product, curve in curves.items():
    if curve is not None and len(curve.times) > 0:
        hedge_mid_prices[product] = curve.forward_price(curve.times[0])

hedge_selector = HedgeSelector(mid_prices=hedge_mid_prices, hedge_ratio=1.0)
hedge_deltas, hedge_orders = hedge_selector.summary(portfolio)

# A-C cost estimate and execution plan for each hedge order
print(f"\n  Hedge execution plan:")
impact_model_hedge = AlmgrenChrissModel()
total_hedge_cost = 0
for ho in hedge_orders:
    est = impact_model_hedge.estimate_impact(
        ho.product, ho.num_contracts,
        price=hedge_mid_prices.get(ho.product, 70.0))
    total_hedge_cost += est.total_cost_usd

    # Show Adaptive scheduler slice plan
    adaptive = AdaptiveScheduler(kappa=1.5)
    sched = adaptive.schedule(ho.product, ho.num_contracts, n_slices=10,
                               duration_min=est.optimal_horizon_min)
    targets = [s.target_contracts for s in sched.slices]
    spread_str = f" / {ho.spread_leg2}" if ho.spread_leg2 else ""

    print(f"    {ho.direction.upper():4s} {ho.num_contracts:3d} "
          f"{ho.ticker}{spread_str} [{ho.hedge_type}]: "
          f"A-C cost={est.total_cost_bps:.1f}bps (${est.total_cost_usd:,.0f}), "
          f"horizon={est.optimal_horizon_min:.0f}min")
    print(f"         Adaptive slices: {targets}")

print(f"\n  Total hedge cost estimate: ${total_hedge_cost:,.0f}")

# ── C++ Hybrid Bump-and-Revalue Risk Analytics ──────────────────────
# Python orchestrates settlement bumps; C++ kernel does all curve
# construction and pricing (ForwardCurve + FuturesPricer).

def _cpp_curve_from_setts(setts, bump_map=None, interp="monotone_convex"):
    """Build a C++ ForwardCurve from settlements with optional bumps."""
    times = [s.time_to_expiry for s in setts]
    prices = [s.settlement + (bump_map.get(i, 0.0) if bump_map else 0.0)
              for i, s in enumerate(setts)]
    return _cpp.ForwardCurve(times, prices, interp)

def _cpp_mtm(cpp_curve, pos, tenor):
    """Mark a single position to market via C++ pricer."""
    pricer = _cpp.FuturesPricer(cpp_curve)
    p = _cpp.FuturesPosition()
    p.ticker = pos.ticker
    p.product = pos.product
    p.num_contracts = pos.num_contracts
    p.direction = pos.direction_sign
    p.entry_price = pos.entry_price
    p.contract_size = pos.contract_size
    return pricer.mark_to_market(p, tenor)

# Build per-product settlement lists (shared by delta, KCD, gamma)
product_setts = {}
for product in set(pos.product for pos in portfolio):
    strip = strips.get(product, [])
    if strip:
        product_setts[product] = [
            FuturesSettlement(
                product=s["product"], contract_code=s["contract_code"],
                settlement=s["settlement"], time_to_expiry=s["time_to_expiry"],
            ) for s in strip
        ]
    else:
        c = curves[product]
        product_setts[product] = [
            FuturesSettlement(
                product=product, contract_code=f"{product}{i+1:02d}",
                settlement=c.forward_price(t), time_to_expiry=t,
            ) for i, t in enumerate(c.times)
        ]

cl_setts = product_setts.get("CL", [])
scenario_engine = ScenarioEngine(bootstrapper, cl_setts, valuation_date=DEFAULT_VALUATION_DATE, product="CL") if cl_setts else None

print("\n--- Parallel Delta (C++ bump-and-revalue) ---")
try:
    total_delta = 0.0
    for pos in portfolio:
        setts = product_setts[pos.product]
        tenor = pricers[pos.product]._find_tenor(pos)
        n = len(setts)
        v_up = _cpp_mtm(_cpp_curve_from_setts(setts, {i: 1.0 for i in range(n)}), pos, tenor)
        v_down = _cpp_mtm(_cpp_curve_from_setts(setts, {i: -1.0 for i in range(n)}), pos, tenor)
        delta = (v_up - v_down) / 2.0
        total_delta += delta
        print(f"  {pos.ticker}: delta = ${delta:,.0f}")
    print(f"  Portfolio parallel delta: ${total_delta:,.0f}")
except Exception as e:
    print(f"  Delta calculation: {e}")

print("\n--- 12-Tenor Key Contract Delta Ladder (C++) ---")
try:
    for pos in portfolio:
        setts = product_setts[pos.product]
        tenor = pricers[pos.product]._find_tenor(pos)
        base_curve = _cpp_curve_from_setts(setts)
        v_base = _cpp_mtm(base_curve, pos, tenor)
        print(f"\n  {pos.ticker} ({pos.product}):")
        for i, s in enumerate(setts):
            v_bumped = _cpp_mtm(_cpp_curve_from_setts(setts, {i: 1.0}), pos, tenor)
            kcd_val = v_bumped - v_base
            bar = "#" * min(40, max(0, int(abs(kcd_val) / 500)))
            print(f"    T={s.time_to_expiry:.3f}y: ${kcd_val:>12,.0f}  {bar}")
except Exception as e:
    print(f"  Key contract delta: {e}")

print("\n--- Gamma (C++ Second-Order Sensitivity) ---")
try:
    total_gamma = 0.0
    bump_sz = 5.0
    for pos in portfolio:
        setts = product_setts[pos.product]
        tenor = pricers[pos.product]._find_tenor(pos)
        n = len(setts)
        v_base = _cpp_mtm(_cpp_curve_from_setts(setts), pos, tenor)
        v_up = _cpp_mtm(_cpp_curve_from_setts(setts, {i: bump_sz for i in range(n)}), pos, tenor)
        v_down = _cpp_mtm(_cpp_curve_from_setts(setts, {i: -bump_sz for i in range(n)}), pos, tenor)
        g = (v_up + v_down - 2.0 * v_base) / (bump_sz ** 2)
        total_gamma += g
        print(f"  {pos.ticker}: gamma = ${g:,.2f} per ($1)^2")
    print(f"  Portfolio gamma: ${total_gamma:,.2f} per ($1)^2")
except Exception as e:
    print(f"  Gamma calculation: {e}")

print("\n--- Scenario Analysis (C++ bump-and-revalue, all products) ---")

def _scenario_bumps(setts, scenario):
    """Compute per-settlement bump map for a scenario (parallel/twist/custom)."""
    n = len(setts)
    tenors = np.array([s.time_to_expiry for s in setts])
    bumps = np.zeros(n)
    stype = scenario["type"]
    if stype == "parallel":
        bumps[:] = scenario["shift_usd"]
    elif stype == "twist":
        front_usd, back_usd = scenario["front_usd"], scenario["back_usd"]
        t_min, t_max = tenors.min(), tenors.max()
        if t_max > t_min:
            for i, t in enumerate(tenors):
                w = (t - t_min) / (t_max - t_min)
                bumps[i] = front_usd * (1 - w) + back_usd * w
        else:
            bumps[:] = (front_usd + back_usd) / 2
    elif stype == "custom":
        bbt = scenario.get("bumps_by_tenor", {})
        if bbt:
            ct = sorted(bbt.keys())
            cb = [bbt[t] for t in ct]
            for i, t in enumerate(tenors):
                bumps[i] = np.interp(t, ct, cb)
    return {i: float(b) for i, b in enumerate(bumps)}

for name, scenario in list(STANDARD_SCENARIOS.items())[:4]:
    try:
        total_pnl = 0.0
        for pos in portfolio:
            setts = product_setts.get(pos.product)
            if not setts:
                continue
            tenor = pricers[pos.product]._find_tenor(pos)
            bump_map = _scenario_bumps(setts, scenario)
            v_base = _cpp_mtm(_cpp_curve_from_setts(setts), pos, tenor)
            v_scen = _cpp_mtm(_cpp_curve_from_setts(setts, bump_map), pos, tenor)
            total_pnl += v_scen - v_base
        print(f"  {name:25s} -> P&L: ${total_pnl:>12,.0f}")
    except Exception as e:
        print(f"  {name:25s} -> Error: {e}")

print("\n--- Roll Yield Analysis (All Products) ---")
for product, product_name in ENERGY_PRODUCTS.items():
    if product not in curves:
        continue
    curve = curves[product]
    roll_calc = RollYieldCalculator(curve)
    matrix = roll_calc.roll_yield_matrix(product)
    if len(matrix) == 0:
        print(f"\n  {product} ({product_name}): no adjacent pairs")
        continue
    print(f"\n  {product} ({product_name}):")
    print(matrix[["front_tenor", "back_tenor", "roll_yield_ann", "regime"]].to_string(index=False))
    best = roll_calc.best_roll_trades(product, top_n=3)
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

print("\n--- Carry Analysis (All Products) ---")
for product, product_name in ENERGY_PRODUCTS.items():
    if product not in curves:
        continue
    curve = curves[product]
    roll_calc = RollYieldCalculator(curve)
    print(f"\n  {product} ({product_name}):")

    # Convenience yield curve
    cy_df = roll_calc.convenience_yield_curve()
    if len(cy_df) > 0:
        print("  Convenience Yield Curve:")
        for _, row in cy_df.iterrows():
            print(f"    {row['tenor']:.2f}y  fwd=${row['forward_price']:.4f}  "
                  f"conv_yield={row['convenience_yield']*100:+.2f}%")

    # Total carry for adjacent pairs
    tenors = [t for t in curve.times if t > 0]
    if len(tenors) >= 2:
        print("  Total Carry (roll yield + convenience yield):")
        for i in range(len(tenors) - 1):
            t1, t2 = tenors[i], tenors[i + 1]
            carry = roll_calc.total_carry(t1, t2)
            ry = roll_calc.roll_yield(t1, t2)
            cy = roll_calc.convenience_yield(t1)
            print(f"    {t1:.2f}y -> {t2:.2f}y: "
                  f"carry={carry*100:+.2f}%  "
                  f"(roll={ry.roll_yield_ann*100:+.2f}% + conv={cy*100:+.2f}%)")


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

# ── Win Probability Model: Training Pipeline ──────────────────────────
print("\n--- Win Probability Model: Training ---")
training_data = WinProbabilityModel.generate_training_data(n_samples=5000, seed=42)
print(f"  Generated {len(training_data)} synthetic RFQs for training")
print(f"  Features: spread_bps, log_notional, volatility, session_hour, client_segment, product")
print(f"  Overall win rate: {training_data['won'].mean():.1%}")

# Temporal train-test split
train_df, test_df = WinProbabilityModel.temporal_train_test_split(training_data, train_frac=0.8)
print(f"  Temporal split: {len(train_df)} train "
      f"({train_df['timestamp'].iloc[0].strftime('%b %d')}–"
      f"{train_df['timestamp'].iloc[-1].strftime('%b %d')}) / "
      f"{len(test_df)} test "
      f"({test_df['timestamp'].iloc[0].strftime('%b %d')}–"
      f"{test_df['timestamp'].iloc[-1].strftime('%b %d')})")

# Train logistic regression
win_model = WinProbabilityModel()
X_train = win_model.prepare_features(train_df)
y_train = train_df["won"].values
X_test = win_model.prepare_features(test_df)
y_test = test_df["won"].values

print(f"  Training logistic regression (2000 iterations)...")
losses = win_model.calibrate(X_train, y_train, lr=0.01, n_iter=2000)

# Evaluate on train and test
train_pred = win_model._sigmoid(X_train @ win_model.beta)
test_pred = win_model._sigmoid(X_test @ win_model.beta)
train_loss = -np.mean(y_train * np.log(train_pred + 1e-12) + (1 - y_train) * np.log(1 - train_pred + 1e-12))
test_loss = -np.mean(y_test * np.log(test_pred + 1e-12) + (1 - y_test) * np.log(1 - test_pred + 1e-12))
print(f"  Log-loss: train={train_loss:.3f}, test={test_loss:.3f}")

# Trained coefficients
print(f"\n  Trained coefficients:")
feat_names = WinProbabilityModel.feature_names()
coef_desc = {
    "intercept": "base log-odds",
    "spread_bps": "wider -> lower win",
    "log_notional": "larger trades harder",
    "volatility": "high vol -> less eager",
    "session_hour": "midday slightly easier",
}
for name, coef in zip(feat_names, win_model.beta):
    desc = coef_desc.get(name, "")
    desc_str = f"  ({desc})" if desc else ""
    print(f"    {name:18s}: {coef:+.3f}{desc_str}")

# Decile calibration
print(f"\n  Decile Calibration (test set):")
cal = WinProbabilityModel.decile_calibration(y_test, test_pred)
print(f"    {'Decile':>6s}  {'Predicted':>9s}  {'Actual':>7s}  {'Count':>5s}")
for row in cal:
    print(f"    {row['decile']:6d}  {row['pred_mean']:9.3f}  {row['actual_mean']:7.3f}  {row['count']:5d}")

# Drift detection (PSI)
print(f"\n  Drift Detection (PSI, train vs test):")
drift = WinProbabilityModel.detect_drift(X_train, X_test, feature_names=feat_names)
for row in drift:
    if row["feature"] == "intercept":
        continue
    print(f"    {row['feature']:18s}: PSI={row['psi']:.3f} -- {row['status']}")

# Sample predictions on live RFQs
sample = rfqs.head(5).copy()
sample["spread_bps"] = 0.5
sample["volatility"] = 0.25
probs = win_model.predict_proba(sample)
print(f"\n  Live RFQ predictions (at 0.5bps): {[f'{p:.1%}' for p in probs]}")

# ── Alpha-Informed Directional Skew ──────────────────────────────────
print("\n--- Alpha-Informed Directional Skew ---")
alpha_skews = {
    "CL": composite,
    "HO": composite * 0.7,  # crude-correlated product
    "RB": composite * 0.7,  # crude-correlated product
    "NG": ng_composite,
}
for prod_code, skew in alpha_skews.items():
    pct = skew * 10
    if skew > 0.05:
        action = "tighten bid, widen ask (favour longs)"
    elif skew < -0.05:
        action = "widen bid, tighten ask (favour shorts)"
    else:
        action = "symmetric"
    print(f"  {prod_code}: alpha={skew:+.2f} -> skew={pct:+.1f}% ({action})")

# ── Quote Optimisation (alpha skew + inventory penalty) ──────────────
print("\n--- Quote Optimisation (alpha skew + inventory risk) ---")
optimizer = QuoteOptimizer(
    win_model=win_model,
    alpha_skews=alpha_skews,
    current_delta={"CL": 0, "HO": 0, "RB": 0, "NG": 0},
    risk_lambda=5e-12,
)
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
              f"avg bid/ask={sub['bid_spread'].mean():.4f}/{sub['ask_spread'].mean():.4f}, "
              f"E[PnL]=${sub['expected_pnl'].sum():,.0f}")

# ── Optimizer vs Flat Pricing Comparison ─────────────────────────────
print("\n--- Optimizer vs Flat Pricing ---")
# Flat baseline uses same win model for fair comparison
flat_optimizer = QuoteOptimizer(win_model=win_model)  # same model, no alpha, no inventory
flat_quotes = flat_optimizer.optimize_batch(rfqs, mid_prices)
print(f"  {'Metric':20s} {'Optimizer':>12s} {'Flat':>12s} {'Edge':>12s}")
print(f"  {'-'*56}")
opt_epnl = quotes['expected_pnl'].sum()
flat_epnl = flat_quotes['expected_pnl'].sum()
print(f"  {'Total E[PnL]':20s} ${opt_epnl:>11,.0f} ${flat_epnl:>11,.0f} ${opt_epnl - flat_epnl:>+11,.0f}")
opt_wp = quotes['win_probability'].mean()
flat_wp = flat_quotes['win_probability'].mean()
print(f"  {'Avg Win Prob':20s} {opt_wp:>11.1%} {flat_wp:>11.1%} {opt_wp - flat_wp:>+11.1%}")
opt_spread = quotes['spread'].mean()
flat_spread = flat_quotes['spread'].mean()
print(f"  {'Avg Spread':20s} {opt_spread:>11.4f} {flat_spread:>11.4f} {opt_spread - flat_spread:>+11.4f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 6: KDB+ Storage & Databento L2 Data
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 6: KDB+ STORAGE & DATABENTO L2 DATA\n{'='*72}\n")

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

kdb = None
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

print("\n--- Databento L2 Data (All Products) ---")
from config import get_databento_api_key
from shared.databento_loader import front_month_symbol
l2_books = {}  # product -> DataFrame of MBP-10 snapshots
db_key = get_databento_api_key()
if db_key:
    try:
        db_loader = DatabentoLoader(DatabentoConfig(api_key=db_key))
        db_date = DEFAULT_VALUATION_DATE
        for _prod in ["CL", "HO", "RB", "NG"]:
            _sym = front_month_symbol(_prod, db_date)
            try:
                _snaps = db_loader.load_book_snapshots(_sym, db_date, n_snapshots=10000)
                if len(_snaps) > 0:
                    l2_books[_prod] = _snaps
                    print(f"  {_prod} ({_sym}): {len(_snaps)} snapshots, "
                          f"bid={_snaps['bid_px_00'].iloc[0]:.4f}, "
                          f"ask={_snaps['ask_px_00'].iloc[0]:.4f}")
            except Exception as e:
                print(f"  {_prod}: L2 load failed ({e})")
        print(f"  Loaded L2 data for {len(l2_books)} products: {list(l2_books.keys())}")
    except Exception as e:
        print(f"  Databento API error: {e}")
        print(f"  (Requires valid Databento subscription and available data)")
else:
    print("  Databento API key not set; skipping L2 data load")

# Store L2 book data in KDB+ if available
_l2_schema_cols = (["ts_event"]
                   + [f"bid_px_{i:02d}" for i in range(10)]
                   + [f"ask_px_{i:02d}" for i in range(10)]
                   + [f"bid_sz_{i:02d}" for i in range(10)]
                   + [f"ask_sz_{i:02d}" for i in range(10)])
if len(l2_books) > 0 and kdb is not None:
    for _prod, _snaps in l2_books.items():
        try:
            l2_df = _snaps[[c for c in _l2_schema_cols if c in _snaps.columns]].copy()
            l2_df["product"] = _prod
            l2_df = l2_df[["ts_event", "product"]
                          + [c for c in _l2_schema_cols if c != "ts_event"]]
            kdb.insert_l2_books(l2_df)
            print(f"  Stored {len(l2_df)} {_prod} L2 snapshots in KDB+")
        except Exception as e:
            print(f"  {_prod} L2 KDB+ storage failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 7: Execution Planning & Market Impact
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 7: EXECUTION PLANNING & MARKET IMPACT ANALYSIS\n{'='*72}\n")

print("--- Almgren-Chriss Impact Model ---")
impact_model = AlmgrenChrissModel()
for product, size in [("CL", 100), ("HO", 50), ("RB", 50), ("NG", 75)]:
    est = impact_model.estimate_impact(product, size)
    print(f"  {product} {size} lots: cost=${est.total_cost_usd:,.0f} "
          f"({est.total_cost_bps:.1f}bps), participation={est.participation_rate:.1%}, "
          f"optimal horizon={est.optimal_horizon_min:.0f}min")

print("\n--- Strategy Comparison (CL 200 lots) ---")
comparison = impact_model.compare_strategies("CL", 200, price=72.50)
print(comparison.to_string(index=False))

print("\n--- Optimal Execution Trajectory ---")
trajectory = impact_model.optimal_trajectory("CL", 200, n_slices=10)
for i, frac in enumerate(trajectory):
    bar = "#" * int(frac * 40)
    print(f"  Slice {i:2d}: {frac:5.1%} {bar}")

# ── Multi-Product Execution Backtest (TWAP / VWAP / Adaptive) ────────
print("\n--- Multi-Product Execution Backtest (Databento L2) ---")
sim = OrderSimulator(seed=42)
_CS = {"CL": 1000, "HO": 42000, "RB": 42000, "NG": 10000}
exec_sizes = {"CL": 100, "HO": 50, "RB": 50, "NG": 75}
all_exec_fills = []  # collect fills for KDB storage
is_results = []      # implementation shortfall records

if len(l2_books) == 0:
    print("  No Databento L2 data available -- execution backtest skipped")
    print("  (Set DATABENTO_API_KEY to enable real L2 execution)")
else:
    for product, qty in exec_sizes.items():
        if product not in l2_books:
            print(f"\n  {product}: no L2 data -- skipped")
            continue

        prod_books = l2_books[product]
        cs = _CS[product]

        # Arrival mid from first L2 snapshot
        arrival_book = L2Book.from_databento_row(product, prod_books.iloc[0])
        arrival_mid = arrival_book.mid_price

        print(f"\n  {product} ({qty} lots, Databento L2, "
              f"mid={arrival_mid:.4f}, spread={arrival_book.spread:.6f}):")
        print(f"    Top of book: bid={arrival_book.bids[0].price:.4f}"
              f"x{arrival_book.bids[0].size} / "
              f"ask={arrival_book.asks[0].price:.4f}"
              f"x{arrival_book.asks[0].size}")

        # Compute optimal horizon from Almgren-Chriss for this product/qty
        adaptive_sched = AdaptiveScheduler(kappa=1.5)
        _opt_sched = adaptive_sched.schedule(product, qty, n_slices=10)
        optimal_horizon = _opt_sched.total_duration_min
        print(f"    Optimal horizon (A-C): {optimal_horizon:.0f} min")

        for SchedulerClass, sname in [(TWAPScheduler, "TWAP"),
                                       (VWAPScheduler, "VWAP"),
                                       (AdaptiveScheduler, "Adaptive")]:
            if sname == "Adaptive":
                scheduler = SchedulerClass(kappa=1.5)
            else:
                scheduler = SchedulerClass()

            # TWAP/VWAP use full session; Adaptive uses optimal horizon
            n_sl = 10
            sched_horizon = optimal_horizon if sname == "Adaptive" else 330.0
            sched = scheduler.schedule(product, qty, n_slices=n_sl,
                                       duration_min=sched_horizon)
            slice_targets = [s.target_contracts for s in sched.slices]

            # Execute against real L2 snapshots using schedule targets
            exec_df = sim.simulate_execution_real(
                product, "buy", qty, prod_books,
                slice_targets=slice_targets,
                duration_min=sched_horizon)

            if len(exec_df) == 0:
                continue

            exec_vwap = ((exec_df["price"] * exec_df["size"]).sum()
                         / exec_df["size"].sum())
            avg_slip = ((exec_df["slippage"] * exec_df["size"]).sum()
                        / exec_df["size"].sum())
            total_filled = int(exec_df["size"].sum())

            # Implementation Shortfall = |exec_VWAP - arrival| * qty * cs
            is_dollar = abs(exec_vwap - arrival_mid) * total_filled * cs
            is_bps = abs(exec_vwap - arrival_mid) / arrival_mid * 10000

            # Almgren-Chriss estimated cost at this horizon
            ac_est = impact_model.estimate_impact(
                product, qty, optimal_horizon, price=arrival_mid)
            print(f"    {sname:8s}: VWAP={exec_vwap:.4f}, "
                  f"slip={avg_slip:.6f}, "
                  f"IS=${is_dollar:,.0f} ({is_bps:.2f}bps), "
                  f"A-C cost={ac_est.total_cost_bps:.1f}bps, "
                  f"risk={ac_est.timing_risk_bps:.1f}bps")

            is_results.append({
                "product": product, "strategy": sname,
                "qty": total_filled, "arrival_mid": arrival_mid,
                "exec_vwap": exec_vwap, "slippage_avg": avg_slip,
                "is_dollar": is_dollar, "is_bps": is_bps,
                "ac_cost_bps": ac_est.total_cost_bps,
                "ac_risk_bps": ac_est.timing_risk_bps,
                "ac_cost_usd": ac_est.total_cost_usd,
                "horizon_min": optimal_horizon,
            })

            # Collect fills for KDB storage
            for _, frow in exec_df.iterrows():
                all_exec_fills.append({
                    "fill_id": len(all_exec_fills) + 1,
                    "product": product, "trd_side": "buy",
                    "price": frow["price"], "qty": int(frow["size"]),
                    "slippage": frow["slippage"],
                })

    # ── Implementation Shortfall Summary ─────────────────────────────
    if is_results:
        print("\n--- Implementation Shortfall Summary ---")
        is_df = pd.DataFrame(is_results)
        print(f"  {'Product':6s} {'Strategy':10s} {'Horizon':>7s} "
              f"{'IS ($)':>10s} {'IS(bps)':>8s} "
              f"{'A-C cost':>9s} {'Risk':>9s}")
        print(f"  {'-'*65}")
        for _, r in is_df.iterrows():
            print(f"  {r['product']:6s} {r['strategy']:10s} "
                  f"{r['horizon_min']:6.0f}m "
                  f"${r['is_dollar']:>9,.0f} "
                  f"{r['is_bps']:>7.2f} "
                  f"{r['ac_cost_bps']:>8.1f}bps "
                  f"{r['ac_risk_bps']:>8.1f}bps")

        print("\n  Best strategy per product (lowest |IS|):")
        for product in exec_sizes:
            prod_is = is_df[is_df["product"] == product]
            if len(prod_is) > 0:
                best = prod_is.loc[prod_is["is_dollar"].abs().idxmin()]
                print(f"    {product}: {best['strategy']} "
                      f"(IS=${best['is_dollar']:+,.0f})")

    # ── Volatile Day Comparison (Adaptive vs TWAP) ──────────────────
    # Load a volatile day to demonstrate Adaptive's risk advantage
    if db_key:
        vol_date = "2026-02-20"
        vol_sym = front_month_symbol("CL", vol_date)
        try:
            vol_snaps = db_loader.load_book_snapshots(vol_sym, vol_date, n_snapshots=500)
            if len(vol_snaps) > 0:
                vol_mids = (vol_snaps["bid_px_00"] + vol_snaps["ask_px_00"]) / 2
                print(f"\n--- Volatile Day Backtest: CL {vol_date} ---")
                print(f"  {vol_sym}: {len(vol_snaps)} snapshots, "
                      f"mid {vol_mids.iloc[0]:.2f} -> {vol_mids.iloc[-1]:.2f} "
                      f"(drift={(vol_mids.iloc[-1]-vol_mids.iloc[0])/vol_mids.iloc[0]*10000:+.0f}bps)")

                vol_qty = 100
                vol_cs = 1000
                vol_arrival = L2Book.from_databento_row("CL", vol_snaps.iloc[0])
                vol_mid = vol_arrival.mid_price

                # Sell execution: price dropped, so selling late = worse fills
                adaptive_s = AdaptiveScheduler(kappa=1.5)
                opt_sched = adaptive_s.schedule("CL", vol_qty, n_slices=10)
                opt_h = opt_sched.total_duration_min

                for Cls, sname in [(TWAPScheduler, "TWAP"),
                                    (VWAPScheduler, "VWAP"),
                                    (AdaptiveScheduler, "Adaptive")]:
                    if sname == "Adaptive":
                        sched = Cls(kappa=1.5).schedule("CL", vol_qty, n_slices=10, duration_min=opt_h)
                        h = opt_h
                    else:
                        sched = Cls().schedule("CL", vol_qty, n_slices=10, duration_min=330.0)
                        h = 330.0
                    targets = [s.target_contracts for s in sched.slices]
                    edf = sim.simulate_execution_real(
                        "CL", "sell", vol_qty, vol_snaps,
                        slice_targets=targets, duration_min=h)
                    if len(edf) == 0:
                        continue
                    evwap = (edf["price"] * edf["size"]).sum() / edf["size"].sum()
                    filled = int(edf["size"].sum())
                    is_d = abs(vol_mid - evwap) * filled * vol_cs
                    is_b = abs(vol_mid - evwap) / vol_mid * 10000
                    print(f"  {sname:8s} [{h:.0f}min]: sell VWAP={evwap:.2f}, "
                          f"IS=${is_d:+,.0f} ({is_b:+.1f}bps)")
        except Exception as e:
            print(f"\n  Volatile day backtest failed: {e}")

    # ── Store Execution Fills in KDB+ ────────────────────────────────
    if kdb is not None and len(all_exec_fills) > 0:
        try:
            fills_df = pd.DataFrame(all_exec_fills)
            kdb.insert_fills(fills_df)
            print(f"\n  Stored {len(fills_df)} execution fills in KDB+ "
                  f"trd_fills table")
            print(f"  Table counts: {kdb.table_counts()}")
        except Exception as e:
            print(f"\n  KDB+ fill storage failed: {e}")

    # ── Post-Trade Execution Markout P&L Decomposition ───────────────
    # Reuse Adaptive results from the backtest above for a proper
    # 3-way cost decomposition: spread + market impact + timing/residual
    print("\n--- Execution Markout P&L Decomposition ---")
    if is_results:
        is_df_all = pd.DataFrame(is_results)
        for product, qty in exec_sizes.items():
            if product not in l2_books:
                continue
            prod_rows = is_df_all[
                (is_df_all["product"] == product)
                & (is_df_all["strategy"] == "Adaptive")
            ]
            if len(prod_rows) == 0:
                continue
            r = prod_rows.iloc[0]

            cs = _CS[product]
            arrival_mid = r["arrival_mid"]
            exec_vwap = r["exec_vwap"]
            total_filled = int(r["qty"])
            is_abs = abs(exec_vwap - arrival_mid)
            is_bps = is_abs / arrival_mid * 10000

            # Component 1: half-spread crossing
            book = L2Book.from_databento_row(product, l2_books[product].iloc[0])
            spread_cost = book.spread / 2 * total_filled * cs

            # Component 2: market impact (Almgren-Chriss model)
            impact_cost = r["ac_cost_usd"]

            # Component 3: timing/residual = total IS - spread - impact
            total_exec_cost = is_abs * total_filled * cs
            timing_cost = total_exec_cost - spread_cost - impact_cost

            print(f"\n  {product} ({qty} lots, Adaptive {r['horizon_min']:.0f}min):")
            print(f"    Arrival mid    : {arrival_mid:.4f}")
            print(f"    Exec VWAP      : {exec_vwap:.4f}")
            print(f"    IS             : {is_abs:.6f} ({is_bps:.2f} bps)")
            print(f"    Spread crossing: ${spread_cost:>+,.2f}")
            print(f"    Market impact  : ${impact_cost:>+,.2f}")
            print(f"    Timing/residual: ${timing_cost:>+,.2f}")
            print(f"    Total exec cost: ${total_exec_cost:>+,.2f}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 8: Portfolio Summary & Markout Analysis
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}\n  STEP 8: PORTFOLIO SUMMARY & MARKOUT ANALYSIS\n{'='*72}\n")

print("--- Markout P&L Analysis ---")
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
            if scenario_engine is None:
                raise RuntimeError("scenario_engine not initialised")
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
    axes[1].plot(comparison["horizon_min"], comparison["impact_bps"], 's-', color=COLOURS["secondary"], linewidth=2, label="Expected Cost")
    axes[1].plot(comparison["horizon_min"], comparison["risk_bps"], 'o--', color=COLOURS["accent"], linewidth=2, label="Timing Risk (1σ)")
    axes[1].legend()
    axes[1].set_xlabel("Execution Horizon (minutes)")
    axes[1].set_ylabel("Cost (bps)")
    axes[1].set_title("Impact vs Risk Tradeoff")
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
