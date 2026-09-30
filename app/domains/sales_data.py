"""In-memory read-only analytical database for the data-analysis domain (project B)."""
from __future__ import annotations

import sqlite3

_CUSTOMERS = [
    ("C1", "张三", "上海"),
    ("C2", "李四", "北京"),
    ("C3", "王五", "上海"),
    ("C4", "赵六", "广州"),
    ("C5", "钱七", "北京"),
]

_PRODUCTS = [
    ("P1", "手机", "电子产品", 3000.0),
    ("P2", "耳机", "电子产品", 300.0),
    ("P3", "沙发", "家具", 2000.0),
    ("P4", "台灯", "家具", 150.0),
    ("P5", "咖啡豆", "食品", 80.0),
]

_ORDERS = [
    ("O1", "C1", "P1", 1, 3000.0, "2026-01-05", "上海"),
    ("O2", "C1", "P2", 2, 600.0, "2026-01-06", "上海"),
    ("O3", "C2", "P3", 1, 2000.0, "2026-01-07", "北京"),
    ("O4", "C3", "P1", 1, 3000.0, "2026-02-01", "上海"),
    ("O5", "C3", "P5", 3, 240.0, "2026-02-02", "上海"),
    ("O6", "C4", "P4", 2, 300.0, "2026-02-03", "广州"),
    ("O7", "C2", "P2", 1, 300.0, "2026-02-04", "北京"),
    ("O8", "C5", "P3", 1, 2000.0, "2026-02-05", "北京"),
    ("O9", "C1", "P5", 1, 80.0, "2026-03-01", "上海"),
    ("O10", "C2", "P1", 1, 3000.0, "2026-03-02", "北京"),
    ("O11", "C4", "P1", 1, 3000.0, "2026-03-03", "广州"),
    ("O12", "C5", "P5", 2, 160.0, "2026-03-04", "北京"),
]

_SCHEMA = """
CREATE TABLE customers(
    customer_id TEXT PRIMARY KEY, name TEXT, city TEXT
);
CREATE TABLE products(
    product_id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL
);
CREATE TABLE orders(
    order_id TEXT PRIMARY KEY, customer_id TEXT, product_id TEXT,
    quantity INTEGER, amount REAL, order_date TEXT, region TEXT
);
"""

_connection: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(_SCHEMA)
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", _CUSTOMERS)
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", _PRODUCTS)
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", _ORDERS)
    conn.commit()
    conn.execute("PRAGMA query_only=ON")  # defense in depth: writes are rejected
    return conn


def get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = _connect()
    return _connection


def schema() -> list[dict]:
    conn = get_connection()
    tables: list[dict] = []
    for (table_name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        columns = [
            {"name": row[1], "type": row[2]}
            for row in conn.execute(f"PRAGMA table_info({table_name})")
        ]
        tables.append({"table": table_name, "columns": columns})
    return tables
