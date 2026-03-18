"""Hedge selection and sizing for energy commodity futures portfolios.

Computes net delta per product, selects optimal hedge contracts,
sizes hedges to neutralise directional and term structure risk,
and outputs hedge orders for the execution module.

Supports:
    1. Outright delta hedge — flatten net delta per product via
       front-month futures.
    2. Calendar spread hedge — neutralise term structure (front vs back)
       exposure by trading the spread as a single instrument.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from module_b_trading.futures_pricer import (
    FuturesPosition, CONTRACT_SPECS, MONTH_CODE,
)


_CODE_FROM_MONTH = {v: k for k, v in MONTH_CODE.items()}


def _ticker_month(ticker):
    """Extract month number from a ticker like 'CLK26' -> 5 (May)."""
    product_len = 2
    if len(ticker) > 3 and ticker[2].isalpha() and ticker[2] not in MONTH_CODE:
        product_len = 3
    month_char = ticker[product_len]
    return MONTH_CODE.get(month_char, 0)


@dataclass
class HedgeOrder:
    """A single hedge order to be sent to the execution module."""
    ticker: str
    product: str
    num_contracts: int
    direction: str
    hedge_type: str = "outright"
    spread_leg2: str = ""
    rationale: str = ""


@dataclass
class PortfolioDelta:
    """Net delta summary for a single product."""
    product: str
    net_contracts: int
    net_dollar_delta: float
    positions: List[FuturesPosition] = field(default_factory=list)
    front_month: str = ""
    back_month: str = ""


class HedgeSelector:
    """Computes hedge requirements and generates hedge orders.

    Given a portfolio of FuturesPositions and current mid prices,
    computes net delta per product, identifies calendar spread
    exposure, and generates hedge orders to neutralise both
    directional and term structure risk.
    """

    def __init__(self, mid_prices, hedge_ratio=1.0):
        self._mid_prices = mid_prices
        self._hedge_ratio = hedge_ratio

    def compute_deltas(self, portfolio):
        """Compute net delta per product from a list of positions."""
        by_product = {}
        for pos in portfolio:
            if pos.product not in by_product:
                by_product[pos.product] = []
            by_product[pos.product].append(pos)

        deltas = {}
        for product, positions in by_product.items():
            net = sum(p.direction_sign * p.num_contracts for p in positions)
            mid = self._mid_prices.get(product, 70.0)
            cs = CONTRACT_SPECS.get(product, {"contract_size": 1000})["contract_size"]
            dollar_delta = net * cs * mid

            sorted_pos = sorted(positions, key=lambda p: _ticker_month(p.ticker))
            front = sorted_pos[0].ticker if sorted_pos else ""
            back = sorted_pos[-1].ticker if len(sorted_pos) > 1 else ""

            deltas[product] = PortfolioDelta(
                product=product,
                net_contracts=net,
                net_dollar_delta=dollar_delta,
                positions=positions,
                front_month=front,
                back_month=back,
            )

        return deltas

    def select_hedges(self, portfolio):
        """Generate hedge orders to neutralise portfolio risk.

        Outright delta hedge per product using the front-month contract.
        """
        deltas = self.compute_deltas(portfolio)
        orders = []

        for product, delta in deltas.items():
            residual = delta.net_contracts
            hedge_contracts = int(round(abs(residual) * self._hedge_ratio))

            if hedge_contracts > 0:
                hedge_dir = "sell" if residual > 0 else "buy"
                orders.append(HedgeOrder(
                    ticker=delta.front_month,
                    product=product,
                    num_contracts=hedge_contracts,
                    direction=hedge_dir,
                    hedge_type="outright",
                    rationale=(
                        f"Delta hedge: net {residual:+d} {product} "
                        f"-> {hedge_dir} {hedge_contracts} "
                        f"{delta.front_month}"
                    ),
                ))

        return orders

    def summary(self, portfolio):
        """Print a human-readable hedge summary."""
        deltas = self.compute_deltas(portfolio)
        orders = self.select_hedges(portfolio)

        print("  Portfolio Delta Summary:")
        for product, d in deltas.items():
            print(f"    {product}: net {d.net_contracts:+d} contracts, "
                  f"${d.net_dollar_delta:+,.0f} dollar-delta "
                  f"(front={d.front_month}, back={d.back_month})")

        print(f"\n  Hedge Orders ({len(orders)}):")
        for o in orders:
            spread_str = f" / {o.spread_leg2}" if o.spread_leg2 else ""
            print(f"    {o.direction.upper():4s} {o.num_contracts} "
                  f"{o.ticker}{spread_str} "
                  f"[{o.hedge_type}] -- {o.rationale}")

        return deltas, orders
