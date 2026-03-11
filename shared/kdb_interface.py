"""KDB+/q interface for commodity tick-data storage.

Provides schema definitions and query helpers for energy commodity
market data stored in a KDB+ tick database. Connects via qpython to
a running KDB+ instance and executes q expressions for table creation,
row insertion, and queries.

Tables:
    forwards: Forward curve snapshots keyed by (date, product, tenor).
    inventory: EIA inventory releases keyed by (date, series).
    fills: Trade execution fills keyed by fill_id.
    tick_data: L1/L2 tick data keyed by (timestamp, product).
"""

import numpy as np
import pandas as pd

import qpython.qconnection as qconn
from qpython.qtype import QSYMBOL_LIST, QFLOAT_LIST, QLONG_LIST, QINT_LIST


_SCHEMAS = {
    "forwards": (
        "forwards:([] dt:(); product:`symbol$(); tenor:`float$();"
        " price:`float$())"
    ),
    "inventory": (
        "inventory:([] dt:(); series:`symbol$(); val:`float$();"
        " unt:`symbol$())"
    ),
    "trd_fills": (
        "trd_fills:([] fill_id:`long$();"
        " product:`symbol$(); trd_side:`symbol$();"
        " price:`float$(); qty:`int$(); slippage:`float$())"
    ),
    "tick_data": (
        "tick_data:([] product:`symbol$();"
        " bid_px:`float$(); ask_px:`float$();"
        " bid_sz:`int$(); ask_sz:`int$();"
        " last_px:`float$(); last_sz:`int$())"
    ),
}


class KDBConfig:
    """KDB+ connection configuration.

    Attributes:
        host: Hostname or IP address of the KDB+ server.
        port: TCP port the KDB+ server listens on.
        user: Username for authenticated connections.
        password: Password for authenticated connections.
    """

    def __init__(self, host="localhost", port=5000, user="", password=""):
        self.host = host
        self.port = port
        self.user = user
        self.password = password


class KDBInterface:
    """Interface to KDB+ for commodity market data.

    Connects via qpython to a running KDB+ instance and provides
    methods for creating tables, inserting rows, and querying data.
    Rows are inserted using q-lang string expressions built from
    DataFrame values.

    Attributes:
        _config: KDBConfig with connection parameters.
        _conn: qpython.QConnection to the KDB+ server.
    """

    def __init__(self, config):
        """Open a connection to KDB+.

        Args:
            config: KDBConfig with host, port, and optional
                credentials.

        Raises:
            ConnectionRefusedError: If no KDB+ server is listening
                on the specified host and port.
        """
        self._config = config
        self._conn = qconn.QConnection(
            host=config.host, port=config.port,
            username=config.user, password=config.password,
        )
        self._conn.open()

    def create_tables(self):
        """Create all schema tables.

        Executes the q DDL for each table defined in ``_SCHEMAS``.
        If a table already exists with the same schema, the DDL
        re-initialises it as an empty table.
        """
        for name, ddl in _SCHEMAS.items():
            self._conn.sendSync(ddl)

    def _insert_rows(self, table, df):
        """Insert DataFrame rows into a KDB+ table using backtick insert.

        Builds a q expression for each row and sends it synchronously.
        Handles type conversion: strings become q symbols via backtick,
        dates use ``\`$"..."`` cast, and numerics are passed directly.

        Args:
            table: Name of the target KDB+ table.
            df: pandas.DataFrame whose columns match the table schema.

        Returns:
            Number of rows inserted.
        """
        sym_cols = {"product", "series", "trd_side", "unt"}
        for _, row in df.iterrows():
            vals = []
            for col in df.columns:
                v = row[col]
                if col in sym_cols:
                    vals.append(f"`{v}")
                elif col in ("dt",):
                    vals.append(f'`$"{v}"')
                elif isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                else:
                    vals.append(str(float(v)))
            q_expr = f"`{table} insert ({';'.join(vals)})"
            self._conn.sendSync(q_expr)
        return len(df)

    def insert_forwards(self, df):
        """Insert forward curve data into the forwards table.

        Args:
            df: DataFrame with columns: date, product, tenor, price,
                timestamp.

        Returns:
            Number of rows inserted.
        """
        return self._insert_rows("forwards", df)

    def insert_inventory(self, df):
        """Insert inventory data into the inventory table.

        Args:
            df: DataFrame with columns: date, series, value, unit.

        Returns:
            Number of rows inserted.
        """
        return self._insert_rows("inventory", df)

    def insert_fills(self, df):
        """Insert execution fills into the trd_fills table.

        Args:
            df: DataFrame with columns: fill_id, product,
                trd_side, prc, qty, slippage.

        Returns:
            Number of rows inserted.
        """
        return self._insert_rows("trd_fills", df)

    def query_forwards(self, product):
        """Query forward curve data for a product.

        Args:
            product: Product code string (e.g. ``'CL'``).

        Returns:
            pandas.DataFrame with columns dt, product, tenor, price.
        """
        result = self._conn.sendSync(
            f'select from forwards where product=`{product}',
        )
        return pd.DataFrame(result)

    def query_inventory(self, series):
        """Query inventory time series.

        Args:
            series: Inventory series name (e.g. ``'crude_stocks'``).

        Returns:
            pandas.DataFrame with columns dt, series, val, unt.
        """
        result = self._conn.sendSync(
            f'select from inventory where series=`{series}',
        )
        return pd.DataFrame(result)

    def query_fills(self, product=None):
        """Query execution fills.

        Args:
            product: Optional product code to filter by. If None,
                returns all fills.

        Returns:
            pandas.DataFrame with columns fill_id, product,
            trd_side, price, qty, slippage.
        """
        if product:
            result = self._conn.sendSync(
                f'select from trd_fills where product=`{product}',
            )
        else:
            result = self._conn.sendSync('select from trd_fills')
        return pd.DataFrame(result)

    def table_counts(self):
        """Get row counts for all tables.

        Returns:
            Dictionary mapping table name strings to integer row
            counts.
        """
        counts = {}
        for name in _SCHEMAS:
            result = self._conn.sendSync(f'count {name}')
            counts[name] = int(result)
        return counts

    def close(self):
        """Close the connection to KDB+."""
        self._conn.close()

    def __repr__(self):
        """Return a string representation showing the connection target.

        Returns:
            String of the form ``KDBInterface(kdb+://host:port)``.
        """
        return f"KDBInterface(kdb+://{self._config.host}:{self._config.port})"
