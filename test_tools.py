"""Tests for the database layer and business tools.

Runs with either `pytest` or `python -m unittest`.

The fixture dataset is small enough to verify by hand. "Today" is fixed at 2026-03-15.

Products (cost / price / stock / reorder):
  P1 Cable    4 / 10 /   5 / 10
  P2 Charger  8 / 15 /  50 / 10
  P3 Earbuds 20 / 35 /   0 /  5
  P4 Case     2 /  6 / 100 / 20   (never sold)

Sales (date, customer, items, paid):
  S1 03-15 Alice : 10 x P1 @10, 2 x P2 @15  total 130, cogs 56, paid 50
  S2 03-15 walk-in: 1 x P3 @35               total  35, cogs 20, paid 35
  S3 03-14 Bob   : 4 x P2 @15               total  60, cogs 32, paid  0
  S4 03-01 Alice : 3 x P3 @35               total 105, cogs 60, paid 105
  S5 02-10 Bob   : 5 x P1 @10               total  50, cogs 20, paid 20
Payments: Bob 25 on 03-10
Expenses: 03-15 Rent 100, 03-14 Transport 20, 03-02 Utilities 30, 02-10 Rent 100
"""
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date

import business_tools as bt
import database

TODAY = date(2026, 3, 15)


def build_fixture(conn):
    database.init_db(conn)
    conn.executemany(
        "INSERT INTO products (id, sku, name, category, cost_price, selling_price, stock_qty, reorder_level) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, "P1", "Cable", "Accessories", 4, 10, 5, 10),
            (2, "P2", "Charger", "Accessories", 8, 15, 50, 10),
            (3, "P3", "Earbuds", "Audio", 20, 35, 0, 5),
            (4, "P4", "Case", "Accessories", 2, 6, 100, 20),
        ],
    )
    conn.executemany("INSERT INTO customers (id, name, phone) VALUES (?,?,?)",
                     [(1, "Alice Rahman", "111"), (2, "Bob Karim", "222")])
    conn.executemany(
        "INSERT INTO sales (id, sale_date, customer_id, amount_paid) VALUES (?,?,?,?)",
        [(1, "2026-03-15", 1, 50), (2, "2026-03-15", None, 35), (3, "2026-03-14", 2, 0),
         (4, "2026-03-01", 1, 105), (5, "2026-02-10", 2, 20)],
    )
    conn.executemany(
        "INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost) VALUES (?,?,?,?,?)",
        [(1, 1, 10, 10, 4), (1, 2, 2, 15, 8), (2, 3, 1, 35, 20), (3, 2, 4, 15, 8),
         (4, 3, 3, 35, 20), (5, 1, 5, 10, 4)],
    )
    conn.execute("INSERT INTO customer_payments (customer_id, payment_date, amount) VALUES (2, '2026-03-10', 25)")
    conn.executemany(
        "INSERT INTO expenses (expense_date, category, description, amount) VALUES (?,?,?,?)",
        [("2026-03-15", "Rent", "March rent", 100), ("2026-03-14", "Transport", "Delivery", 20),
         ("2026-03-02", "Utilities", "Electricity", 30), ("2026-02-10", "Rent", "Feb rent", 100)],
    )
    conn.commit()


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self.conn = database.get_connection(":memory:")
        build_fixture(self.conn)

    def tearDown(self):
        self.conn.close()


class EmptyDbTestCase(unittest.TestCase):
    def setUp(self):
        self.conn = database.get_connection(":memory:")
        database.init_db(self.conn)

    def tearDown(self):
        self.conn.close()


# --------------------------------------------------------------------------- #
class TestPeriods(unittest.TestCase):
    def test_named_periods(self):
        r = bt.resolve_period
        self.assertEqual((r("today", today=TODAY).start, r("today", today=TODAY).end), (TODAY, TODAY))
        self.assertEqual(r("yesterday", today=TODAY).start, date(2026, 3, 14))
        self.assertEqual(r("this_week", today=TODAY).start, date(2026, 3, 9))  # Monday
        self.assertEqual(r("last_7_days", today=TODAY).start, date(2026, 3, 9))
        self.assertEqual(r("this_month", today=TODAY).start, date(2026, 3, 1))
        self.assertEqual(r("last_30_days", today=TODAY).start, date(2026, 2, 14))
        lm = r("last_month", today=TODAY)
        self.assertEqual((lm.start, lm.end), (date(2026, 2, 1), date(2026, 2, 28)))
        self.assertEqual(r("this_year", today=TODAY).start, date(2026, 1, 1))
        self.assertEqual(r(None, today=TODAY).label, "last_30_days")

    def test_last_month_across_year_boundary_and_leap_year(self):
        lm = bt.resolve_period("last_month", today=date(2026, 1, 10))
        self.assertEqual((lm.start, lm.end), (date(2025, 12, 1), date(2025, 12, 31)))
        lm = bt.resolve_period("last_month", today=date(2024, 3, 5))
        self.assertEqual(lm.end, date(2024, 2, 29))

    def test_custom_range(self):
        p = bt.resolve_period(start_date="2026-03-01", end_date="2026-03-10", today=TODAY)
        self.assertEqual((p.label, p.days), ("custom", 10))
        p = bt.resolve_period(start_date="2026-03-01", today=TODAY)
        self.assertEqual(p.end, TODAY)

    def test_invalid_periods(self):
        with self.assertRaises(bt.ToolInputError):
            bt.resolve_period("next_week", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.resolve_period(start_date="03/01/2026", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.resolve_period(start_date="2026-02-30", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.resolve_period(start_date="2026-03-10", end_date="2026-03-01", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.resolve_period(end_date="2026-03-01", today=TODAY)

    def test_previous_period(self):
        p = bt.resolve_period("this_month", today=TODAY)  # 03-01..03-15 = 15 days
        prev = bt.previous_period(p)
        self.assertEqual((prev.start, prev.end), (date(2026, 2, 14), date(2026, 2, 28)))
        self.assertIsNone(bt.previous_period(bt.resolve_period("all_time", today=TODAY)))


# --------------------------------------------------------------------------- #
class TestSalesSummary(ToolTestCase):
    def test_today(self):
        r = bt.get_sales_summary(self.conn, "today", today=TODAY)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["sale_count"], 2)
        self.assertEqual(r["units_sold"], 13)
        self.assertEqual(r["total_sales"], 165.0)
        self.assertEqual(r["cost_of_goods_sold"], 76.0)
        self.assertEqual(r["gross_profit"], 89.0)
        self.assertEqual(r["gross_margin_pct"], 53.94)
        self.assertEqual(r["total_recorded_expenses"], 100.0)
        self.assertEqual(r["profit_after_recorded_expenses"], -11.0)
        # cash: 50 + 35 collected at sale, nothing later, minus 100 expenses
        self.assertEqual(r["cash_collected_on_sales"], 85.0)
        self.assertEqual(r["total_cash_received"], 85.0)
        self.assertEqual(r["net_recorded_cash_flow"], -15.0)

    def test_yesterday(self):
        r = bt.get_sales_summary(self.conn, "yesterday", today=TODAY)
        self.assertEqual((r["total_sales"], r["cost_of_goods_sold"], r["gross_profit"]), (60.0, 32.0, 28.0))
        self.assertEqual(r["profit_after_recorded_expenses"], 8.0)
        self.assertEqual(r["net_recorded_cash_flow"], -20.0)

    def test_this_month_includes_customer_payment_in_cash_flow(self):
        r = bt.get_sales_summary(self.conn, "this_month", today=TODAY)
        self.assertEqual(r["sale_count"], 4)
        self.assertEqual(r["total_sales"], 330.0)
        self.assertEqual(r["cost_of_goods_sold"], 168.0)
        self.assertEqual(r["gross_profit"], 162.0)
        self.assertEqual(r["total_recorded_expenses"], 150.0)
        self.assertEqual(r["profit_after_recorded_expenses"], 12.0)
        self.assertEqual(r["cash_collected_on_sales"], 190.0)
        self.assertEqual(r["customer_payments_received"], 25.0)
        self.assertEqual(r["net_recorded_cash_flow"], 65.0)
        # profit and cash flow are different numbers
        self.assertNotEqual(r["profit_after_recorded_expenses"], r["net_recorded_cash_flow"])

    def test_date_filter_boundaries_are_inclusive(self):
        r = bt.get_sales_summary(self.conn, start_date="2026-03-14", end_date="2026-03-14", today=TODAY)
        self.assertEqual(r["total_sales"], 60.0)
        r = bt.get_sales_summary(self.conn, start_date="2026-03-01", end_date="2026-03-01", today=TODAY)
        self.assertEqual(r["total_sales"], 105.0)
        r = bt.get_sales_summary(self.conn, start_date="2026-02-11", end_date="2026-02-28", today=TODAY)
        self.assertEqual(r["status"], "no_sales")

    def test_last_30_days_excludes_feb_10(self):
        r = bt.get_sales_summary(self.conn, "last_30_days", today=TODAY)
        self.assertEqual(r["total_sales"], 330.0)
        self.assertEqual(r["total_recorded_expenses"], 150.0)

    def test_last_month_and_all_time(self):
        r = bt.get_sales_summary(self.conn, "last_month", today=TODAY)
        self.assertEqual((r["total_sales"], r["gross_profit"]), (50.0, 30.0))
        self.assertEqual(r["profit_after_recorded_expenses"], -70.0)
        r = bt.get_sales_summary(self.conn, "all_time", today=TODAY)
        self.assertEqual((r["total_sales"], r["cost_of_goods_sold"], r["gross_profit"]), (380.0, 188.0, 192.0))
        self.assertEqual(r["total_recorded_expenses"], 250.0)
        self.assertEqual(r["profit_after_recorded_expenses"], -58.0)
        self.assertEqual(r["net_recorded_cash_flow"], -15.0)
        self.assertNotIn("daily_breakdown", r)

    def test_daily_breakdown_sums_match_totals(self):
        r = bt.get_sales_summary(self.conn, "this_month", today=TODAY)
        days = r["daily_breakdown"]
        self.assertEqual(sum(d["total_sales"] for d in days), r["total_sales"])
        self.assertEqual(sum(d["expenses"] for d in days), r["total_recorded_expenses"])
        self.assertEqual([d["date"] for d in days], sorted(d["date"] for d in days))

    def test_no_sales_period_still_reports_expenses(self):
        r = bt.get_sales_summary(self.conn, start_date="2026-03-02", end_date="2026-03-02", today=TODAY)
        self.assertEqual(r["status"], "no_sales")
        self.assertEqual(r["total_sales"], 0.0)
        self.assertEqual(r["total_recorded_expenses"], 30.0)
        self.assertEqual(r["profit_after_recorded_expenses"], -30.0)
        self.assertIsNone(r["gross_margin_pct"])

    def test_sale_cost_uses_snapshot_not_current_cost(self):
        self.conn.execute("UPDATE products SET cost_price = 999 WHERE id = 1")
        r = bt.get_sales_summary(self.conn, "today", today=TODAY)
        self.assertEqual(r["cost_of_goods_sold"], 76.0)


class TestTopProducts(ToolTestCase):
    def test_ranking_by_revenue(self):
        r = bt.get_top_products(self.conn, "this_month", today=TODAY)
        self.assertEqual([p["name"] for p in r["products"]], ["Earbuds", "Cable", "Charger"])
        earbuds = r["products"][0]
        self.assertEqual((earbuds["units_sold"], earbuds["total_sales"], earbuds["gross_profit"]), (4, 140.0, 60.0))

    def test_limit_and_sort(self):
        r = bt.get_top_products(self.conn, "this_month", limit=1, sort_by="units", today=TODAY)
        self.assertEqual(r["products"][0]["name"], "Cable")  # 10 units
        self.assertEqual(r["products_with_sales"], 3)

    def test_validation(self):
        with self.assertRaises(bt.ToolInputError):
            bt.get_top_products(self.conn, limit=0, today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.get_top_products(self.conn, limit="5", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.get_top_products(self.conn, sort_by="color", today=TODAY)


# --------------------------------------------------------------------------- #
class TestInventory(ToolTestCase):
    def test_inventory_status(self):
        r = bt.get_inventory_status(self.conn)
        self.assertEqual(r["product_count"], 4)
        self.assertEqual(r["out_of_stock_count"], 1)
        self.assertEqual(r["low_stock_count"], 1)
        # 5*4 + 50*8 + 0*20 + 100*2
        self.assertEqual(r["total_stock_value_at_cost"], 620.0)
        status = {p["name"]: p["stock_status"] for p in r["products"]}
        self.assertEqual(status, {"Cable": "low_stock", "Charger": "ok", "Earbuds": "out_of_stock", "Case": "ok"})

    def test_category_filter_is_case_insensitive_and_parameterized(self):
        self.assertEqual(bt.get_inventory_status(self.conn, category="audio")["product_count"], 1)
        r = bt.get_inventory_status(self.conn, category="x'; DROP TABLE products;--")
        self.assertEqual(r["status"], "no_data")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0], 4)

    def test_inactive_products_hidden_by_default(self):
        self.conn.execute("UPDATE products SET is_active = 0 WHERE id = 4")
        self.assertEqual(bt.get_inventory_status(self.conn)["product_count"], 3)
        self.assertEqual(bt.get_inventory_status(self.conn, include_inactive=True)["product_count"], 4)

    def test_low_stock_products_and_priorities(self):
        r = bt.get_low_stock_products(self.conn, today=TODAY)
        self.assertEqual(r["low_stock_count"], 2)
        self.assertEqual(r["out_of_stock_count"], 1)
        names = [p["name"] for p in r["products"]]
        self.assertEqual(names, ["Earbuds", "Cable"])  # critical before medium
        earbuds, cable = r["products"]
        self.assertEqual(earbuds["priority"], "critical")
        self.assertEqual(earbuds["units_sold_last_30_days"], 4)
        # target = max(4 units of cover, 2 x reorder 5 = 10) -> 10 - 0
        self.assertEqual(earbuds["suggested_reorder_qty"], 10)
        self.assertEqual(earbuds["estimated_restock_cost"], 200.0)
        self.assertEqual(cable["priority"], "medium")
        self.assertEqual(cable["days_of_cover"], 15.0)  # 5 in stock, 10 sold / 30 days
        # target = max(10, 2 x 10) = 20 -> 20 - 5
        self.assertEqual(cable["suggested_reorder_qty"], 15)
        self.assertEqual(cable["estimated_restock_cost"], 60.0)

    def test_restock_recommendations(self):
        r = bt.get_restock_recommendations(self.conn, today=TODAY)
        self.assertEqual([x["name"] for x in r["recommendations"]], ["Earbuds", "Cable"])
        self.assertEqual(r["total_estimated_restock_cost_all_candidates"], 260.0)
        r = bt.get_restock_recommendations(self.conn, limit=1, today=TODAY)
        self.assertEqual(len(r["recommendations"]), 1)
        self.assertEqual(r["candidate_count"], 2)

    def test_fast_seller_above_reorder_level_is_flagged_as_high(self):
        # 100 units sold in 30 days (3.33/day) with 20 in stock = 6 days of cover, reorder level only 5.
        self.conn.execute("UPDATE products SET stock_qty = 20, reorder_level = 5 WHERE id = 4")
        self.conn.execute("INSERT INTO sales (id, sale_date, customer_id, amount_paid) VALUES (9,'2026-03-10',NULL,600)")
        self.conn.execute("INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost) "
                          "VALUES (9, 4, 100, 6, 2)")
        low = bt.get_low_stock_products(self.conn, today=TODAY)
        self.assertNotIn("Case", [p["name"] for p in low["products"]])  # not below reorder level
        rec = bt.get_restock_recommendations(self.conn, today=TODAY)
        case = next(x for x in rec["recommendations"] if x["name"] == "Case")
        self.assertEqual(case["priority"], "high")
        self.assertEqual(case["days_of_cover"], 6.0)
        self.assertEqual(case["suggested_reorder_qty"], 80)  # target 100 - 20

    def test_stock_classification_rules(self):
        c = bt._classify_stock
        self.assertEqual(c(0, 5, 4)[0], "critical")
        self.assertEqual(c(1, 5, 30)[0], "critical")   # 1 day of cover
        self.assertEqual(c(7, 50, 30)[0], "high")      # 7 days of cover
        self.assertEqual(c(8, 50, 30)[0], "medium")    # 8 days, below reorder level
        self.assertEqual(c(3, 5, 0)[0], "low")         # low but no sales
        self.assertIsNone(c(50, 5, 3)[0])
        self.assertEqual(c(0, 0, 0)[0], "low")         # out of stock, never sold


# --------------------------------------------------------------------------- #
class TestCustomerDues(ToolTestCase):
    def test_total_and_ordering(self):
        r = bt.get_customer_dues(self.conn, today=TODAY)
        self.assertEqual(r["total_outstanding"], 145.0)
        self.assertEqual(r["customers_owing_count"], 2)
        self.assertEqual([(c["name"], c["outstanding_balance"]) for c in r["customers"]],
                         [("Alice Rahman", 80.0), ("Bob Karim", 65.0)])

    def test_bob_details_and_fifo_aging(self):
        r = bt.get_customer_dues(self.conn, customer_name="bob", today=TODAY)
        bob = r["customers"][0]
        self.assertEqual(bob["total_billed"], 110.0)
        self.assertEqual(bob["paid_at_time_of_sale"], 20.0)
        self.assertEqual(bob["later_payments_received"], 25.0)
        self.assertEqual(bob["last_payment_date"], "2026-03-10")
        # the 25 payment is applied to the Feb 10 sale (30 unpaid) first -> 5 still unpaid from Feb 10
        self.assertEqual(bob["oldest_unpaid_sale_date"], "2026-02-10")
        self.assertEqual(bob["days_since_oldest_unpaid_sale"], 33)
        aging = bt.get_customer_dues(self.conn, today=TODAY)["aging_of_outstanding_by_sale_age"]
        self.assertEqual(aging, {"0_30_days": 140.0, "31_60_days": 5.0, "61_90_days": 0.0, "over_90_days": 0.0})

    def test_unknown_customer(self):
        self.assertEqual(bt.get_customer_dues(self.conn, customer_name="Zed", today=TODAY)["status"], "not_found")

    def test_fully_paid_customer_and_credit_balance(self):
        self.conn.execute("INSERT INTO customer_payments (customer_id, payment_date, amount) VALUES (2,'2026-03-12',65)")
        self.conn.execute("INSERT INTO customer_payments (customer_id, payment_date, amount) VALUES (2,'2026-03-13',10)")
        r = bt.get_customer_dues(self.conn, today=TODAY)
        self.assertEqual(r["total_outstanding"], 80.0)
        self.assertEqual(r["customers_with_credit_count"], 1)
        bob = bt.get_customer_dues(self.conn, customer_name="Bob", today=TODAY)["customers"][0]
        self.assertEqual(bob["outstanding_balance"], -10.0)
        self.assertIsNone(bob["oldest_unpaid_sale_date"])

    def test_no_dues(self):
        self.conn.execute("UPDATE sales SET amount_paid = 1000 WHERE customer_id IS NOT NULL")
        r = bt.get_customer_dues(self.conn, today=TODAY)
        self.assertEqual(r["total_outstanding"], 0.0)
        self.assertEqual(r["customers"], [])
        self.assertIn("message", r)

    def test_walk_in_sales_never_create_dues(self):
        self.conn.execute("UPDATE sales SET amount_paid = 0 WHERE id = 2")  # walk-in, unpaid
        self.assertEqual(bt.get_customer_dues(self.conn, today=TODAY)["total_outstanding"], 145.0)


class TestExpenses(ToolTestCase):
    def test_this_month(self):
        r = bt.get_expense_summary(self.conn, "this_month", today=TODAY)
        self.assertEqual(r["total_recorded_expenses"], 150.0)
        self.assertEqual(r["expense_entries"], 3)
        cats = [(c["category"], c["total"], c["pct_of_expenses"]) for c in r["by_category"]]
        self.assertEqual(cats, [("Rent", 100.0, 66.67), ("Utilities", 30.0, 20.0), ("Transport", 20.0, 13.33)])
        self.assertEqual(r["largest_expenses"][0]["amount"], 100.0)
        impact = r["impact_on_profit"]
        self.assertEqual(impact["gross_profit"], 162.0)
        self.assertEqual(impact["profit_after_recorded_expenses"], 12.0)
        self.assertEqual(impact["expenses_pct_of_sales"], 45.45)
        self.assertEqual(impact["expenses_pct_of_gross_profit"], 92.59)
        self.assertTrue(impact["gross_profit_covers_expenses"])

    def test_top_n(self):
        r = bt.get_expense_summary(self.conn, "this_month", top_n=1, today=TODAY)
        self.assertEqual(len(r["by_category"]), 1)
        self.assertEqual(len(r["largest_expenses"]), 1)

    def test_no_expenses(self):
        r = bt.get_expense_summary(self.conn, start_date="2026-03-03", end_date="2026-03-13", today=TODAY)
        self.assertEqual(r["status"], "no_expenses")
        self.assertEqual(r["total_recorded_expenses"], 0.0)


class TestRecommendations(ToolTestCase):
    def codes(self, **kw):
        r = bt.get_business_recommendations(self.conn, today=TODAY, **kw)
        return r, {i["code"]: i for i in r["insights"]}

    def test_expected_insights_for_fixture(self):
        r, codes = self.codes(period="last_30_days")
        self.assertEqual(r["status"], "ok")
        self.assertIn("revenue_up", codes)             # 330 vs 50
        self.assertIn("expense_ratio_high", codes)     # 150 / 162
        self.assertIn("restock_urgent", codes)         # Earbuds out of stock
        self.assertIn("slow_movers", codes)            # Case: 100 units, no sales
        self.assertIn("high_dues", codes)              # 145 owed vs 330 sales = 43.9%
        self.assertIn("top_profit_product", codes)
        self.assertNotIn("loss_after_expenses", codes)
        self.assertNotIn("sold_below_cost", codes)
        self.assertEqual(codes["slow_movers"]["evidence"]["stock_value_at_cost"], 200.0)
        self.assertEqual(codes["high_dues"]["evidence"]["total_outstanding"], 145.0)
        self.assertEqual(codes["high_dues"]["severity"], "medium")
        # high severity items come first
        ranks = [{"high": 0, "medium": 1, "low": 2}[i["severity"]] for i in r["insights"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_loss_and_revenue_drop(self):
        # today vs yesterday: revenue 165 vs 60 (up), but a loss after the 100 rent
        r, codes = self.codes(period="today")
        self.assertIn("loss_after_expenses", codes)
        self.assertEqual(codes["loss_after_expenses"]["severity"], "high")
        self.assertEqual(codes["loss_after_expenses"]["evidence"]["profit_after_recorded_expenses"], -11.0)
        # yesterday vs the day before (no sales then): no trend insight, cannot compare to zero
        r, codes = self.codes(period="yesterday")
        self.assertNotIn("revenue_up", codes)
        self.assertNotIn("revenue_down", codes)

    def test_revenue_down_detected(self):
        r, codes = self.codes(start_date="2026-03-15", end_date="2026-03-15")
        self.assertIn("revenue_up", codes)
        self.conn.execute("UPDATE sale_items SET unit_price = 1 WHERE sale_id IN (1, 2)")  # today collapses
        _, codes = self.codes(start_date="2026-03-15", end_date="2026-03-15")
        self.assertEqual(codes["revenue_down"]["severity"], "high")

    def test_sold_below_cost_and_low_margin(self):
        self.conn.execute("UPDATE sale_items SET unit_price = 3 WHERE sale_id = 3")  # 4 x P2 cost 8 sold at 3
        _, codes = self.codes(period="yesterday")
        self.assertEqual(codes["sold_below_cost"]["evidence"]["total_loss"], 20.0)  # (8-3)*4
        self.assertIn("low_margin", codes)

    def test_expense_category_jump(self):
        self.conn.execute("INSERT INTO expenses (expense_date, category, amount) VALUES ('2026-03-05','Rent',500)")
        _, codes = self.codes(start_date="2026-03-01", end_date="2026-03-15")
        # previous window is 02-14..02-28 which has no expenses -> no jump possible
        self.assertNotIn("expense_category_up", codes)
        self.conn.execute("INSERT INTO expenses (expense_date, category, amount) VALUES ('2026-02-20','Rent',100)")
        _, codes = self.codes(start_date="2026-03-01", end_date="2026-03-15")
        self.assertIn("expense_category_up", codes)
        self.assertEqual(codes["expense_category_up"]["evidence"], {"category": "Rent", "current": 600.0, "previous": 100.0})

    def test_overdue_dues(self):
        self.conn.execute("INSERT INTO sales (id, sale_date, customer_id, amount_paid) VALUES (20,'2025-12-01',1,0)")
        self.conn.execute("INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost) "
                          "VALUES (20, 2, 2, 15, 8)")
        _, codes = self.codes(period="this_month")
        self.assertEqual(codes["overdue_dues"]["evidence"]["overdue_amount"], 30.0)


# --------------------------------------------------------------------------- #
class TestReports(ToolTestCase):
    def test_daily_report(self):
        r = bt.get_business_report(self.conn, "daily", today=TODAY)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["period"]["start_date"], "2026-03-15")
        self.assertEqual(r["previous_period"]["start_date"], "2026-03-14")
        self.assertEqual(r["summary"]["total_sales"], 165.0)
        cmp_sales = r["comparison_with_previous_period"]["total_sales"]
        self.assertEqual((cmp_sales["current"], cmp_sales["previous"], cmp_sales["change_pct"]), (165.0, 60.0, 175.0))
        self.assertIsNone(r["daily_breakdown"])
        self.assertEqual(r["customer_dues"]["total_outstanding"], 145.0)
        self.assertEqual(r["inventory"]["low_stock_count"], 2)

    def test_weekly_report_has_none_change_when_previous_is_zero(self):
        r = bt.get_business_report(self.conn, "weekly", today=TODAY)
        self.assertEqual(r["period"]["start_date"], "2026-03-09")
        self.assertEqual(r["summary"]["total_sales"], 225.0)
        self.assertEqual(r["previous_period"]["start_date"], "2026-03-02")
        self.assertIsNone(r["comparison_with_previous_period"]["total_sales"]["change_pct"])
        self.assertEqual(r["summary"]["customer_payments_received"], 25.0)
        self.assertEqual(len(r["daily_breakdown"]), 2)

    def test_monthly_report_and_specific_date(self):
        r = bt.get_business_report(self.conn, "monthly", today=TODAY)
        self.assertEqual(r["period"]["start_date"], "2026-03-01")
        self.assertEqual(r["summary"]["profit_after_recorded_expenses"], 12.0)
        r = bt.get_business_report(self.conn, "monthly", report_date="2026-02-28", today=TODAY)
        self.assertEqual(r["summary"]["total_sales"], 50.0)

    def test_text_rendering(self):
        text = bt.report_to_text(bt.get_business_report(self.conn, "daily", today=TODAY))
        self.assertIn("DAILY BUSINESS REPORT", text)
        self.assertIn("Gross profit:", text)
        self.assertIn("89.00", text)

    def test_invalid_input(self):
        with self.assertRaises(bt.ToolInputError):
            bt.get_business_report(self.conn, "yearly", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.get_business_report(self.conn, "daily", report_date="2026-03-16", today=TODAY)
        with self.assertRaises(bt.ToolInputError):
            bt.get_business_report(self.conn, "daily", report_date="yesterday", today=TODAY)


# --------------------------------------------------------------------------- #
class TestEmptyDatabase(EmptyDbTestCase):
    def test_every_tool_handles_empty_database(self):
        for name in bt.TOOLS:
            args = _smoke_test_arguments(name)
            result = bt.execute_tool(self.conn, name, args, today=TODAY)
            self.assertNotEqual(result["status"], "error", f"{name}: {result}")
            json.dumps(result)  # must be serialisable

    def test_empty_values(self):
        s = bt.get_sales_summary(self.conn, "today", today=TODAY)
        self.assertEqual((s["status"], s["total_sales"], s["gross_profit"]), ("no_sales", 0.0, 0.0))
        self.assertIsNone(s["gross_margin_pct"])
        self.assertEqual(bt.get_low_stock_products(self.conn, today=TODAY)["status"], "no_data")
        self.assertEqual(bt.get_inventory_status(self.conn)["status"], "no_data")
        self.assertEqual(bt.get_customer_dues(self.conn, today=TODAY)["status"], "no_data")
        self.assertEqual(bt.get_business_recommendations(self.conn, today=TODAY)["status"], "no_data")
        self.assertEqual(bt.get_business_report(self.conn, "monthly", today=TODAY)["status"], "no_data")
        self.assertEqual(bt.get_expense_summary(self.conn, today=TODAY)["status"], "no_expenses")


class TestExecuteTool(ToolTestCase):
    def test_runs_tool_with_json_string_arguments(self):
        r = bt.execute_tool(self.conn, "get_sales_summary", '{"period": "today"}', today=TODAY)
        self.assertEqual(r["total_sales"], 165.0)

    def test_error_results(self):
        e = bt.execute_tool
        self.assertEqual(e(self.conn, "drop_tables", {}, TODAY)["error_type"], "unknown_tool")
        self.assertEqual(e(self.conn, "get_sales_summary", "{not json", TODAY)["error_type"], "invalid_input")
        self.assertEqual(e(self.conn, "get_sales_summary", "[1]", TODAY)["error_type"], "invalid_input")
        self.assertEqual(e(self.conn, "get_sales_summary", {"period": "forever"}, TODAY)["error_type"], "invalid_input")
        self.assertEqual(e(self.conn, "get_sales_summary", {"sql": "DELETE FROM sales"}, TODAY)["error_type"],
                         "invalid_input")
        self.assertEqual(e(self.conn, "get_sales_summary", {"today": "2020-01-01"}, TODAY)["error_type"],
                         "invalid_input")
        self.assertEqual(e(self.conn, "get_business_report", {}, TODAY)["error_type"], "invalid_input")

    def test_database_errors_are_reported_not_raised(self):
        self.conn.execute("DROP TABLE expenses")
        r = bt.execute_tool(self.conn, "get_expense_summary", {}, today=TODAY)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["error_type"], "database_error")
        self.assertNotIn("Traceback", r["message"])

    def test_tool_definitions_match_registry(self):
        defs = bt.get_tool_definitions()
        self.assertEqual({d["function"]["name"] for d in defs}, set(bt.TOOLS))
        for d in defs:
            self.assertEqual(d["type"], "function")
            self.assertEqual(d["function"]["parameters"]["type"], "object")
            self.assertTrue(d["function"]["description"])
        json.dumps(defs)

    def test_tools_never_modify_data(self):
        def snapshot():
            return [tuple(r) for r in self.conn.execute(
                "SELECT (SELECT COUNT(*) FROM products), (SELECT COUNT(*) FROM sales), "
                "(SELECT COUNT(*) FROM expenses), (SELECT SUM(stock_qty) FROM products), "
                "(SELECT SUM(amount) FROM expenses)")]
        before = snapshot()
        for name in bt.TOOLS:
            args = {"report_type": "monthly"} if name == "get_business_report" else {}
            bt.execute_tool(self.conn, name, args, today=TODAY)
        self.assertEqual(snapshot(), before)


class TestReadOnlyConnection(unittest.TestCase):
    def test_read_only_connection_blocks_writes_but_allows_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.db")
            rw = database.get_connection(path)
            build_fixture(rw)
            rw.close()

            ro = database.get_connection(path, read_only=True)
            try:
                self.assertEqual(bt.get_sales_summary(ro, "today", today=TODAY)["total_sales"], 165.0)
                with self.assertRaises(sqlite3.OperationalError):
                    ro.execute("DELETE FROM sales")
                with self.assertRaises(sqlite3.OperationalError):
                    ro.execute("INSERT INTO expenses (expense_date, category, amount) VALUES ('2026-03-15','x',1)")
            finally:
                ro.close()

    def test_schema_rejects_bad_data(self):
        conn = database.get_connection(":memory:")
        build_fixture(conn)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO sales (sale_date, amount_paid) VALUES ('15/03/2026', 0)")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO expenses (expense_date, category, amount) VALUES ('2026-03-15','x',-5)")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO sale_items (sale_id, product_id, quantity, unit_price, unit_cost) "
                         "VALUES (1, 1, 0, 1, 1)")
        conn.close()


class TestSeedData(unittest.TestCase):
    """Cross-check tool output against independent plain-Python arithmetic on the seeded rows."""

    def setUp(self):
        import seed_data
        self.seed_data = seed_data
        self.conn = database.get_connection(":memory:")
        self.counts = seed_data.seed_database(self.conn, today=TODAY, seed=7)

    def tearDown(self):
        self.conn.close()

    def test_seed_refuses_to_overwrite_without_reset(self):
        with self.assertRaises(RuntimeError):
            self.seed_data.seed_database(self.conn, today=TODAY)
        self.seed_data.seed_database(self.conn, today=TODAY, seed=7, reset=True)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0], len(self.seed_data.PRODUCTS))

    def test_seed_is_deterministic(self):
        other = database.get_connection(":memory:")
        self.seed_data.seed_database(other, today=TODAY, seed=7)
        q = "SELECT COUNT(*), SUM(quantity * unit_price) FROM sale_items"
        self.assertEqual(tuple(self.conn.execute(q).fetchone()), tuple(other.execute(q).fetchone()))
        other.close()

    def test_totals_match_independent_calculation(self):
        rows = self.conn.execute(
            "SELECT s.sale_date, si.quantity, si.unit_price, si.unit_cost "
            "FROM sales s JOIN sale_items si ON si.sale_id = s.id").fetchall()
        start = "2026-02-14"  # last_30_days
        in_range = [r for r in rows if start <= r["sale_date"] <= "2026-03-15"]
        revenue = sum(r["quantity"] * r["unit_price"] for r in in_range)
        cogs = sum(r["quantity"] * r["unit_cost"] for r in in_range)
        expenses = sum(r["amount"] for r in self.conn.execute(
            "SELECT amount FROM expenses WHERE expense_date BETWEEN ? AND ?", (start, "2026-03-15")))
        result = bt.get_sales_summary(self.conn, "last_30_days", today=TODAY)
        self.assertAlmostEqual(result["total_sales"], revenue, places=2)
        self.assertAlmostEqual(result["cost_of_goods_sold"], cogs, places=2)
        self.assertAlmostEqual(result["gross_profit"], revenue - cogs, places=2)
        self.assertAlmostEqual(result["total_recorded_expenses"], expenses, places=2)
        self.assertAlmostEqual(result["profit_after_recorded_expenses"], revenue - cogs - expenses, places=2)

    def test_dues_match_independent_calculation(self):
        expected = 0.0
        for c in self.conn.execute("SELECT id FROM customers"):
            billed = self.conn.execute(
                "SELECT COALESCE(SUM(si.quantity*si.unit_price),0) FROM sales s JOIN sale_items si ON si.sale_id=s.id "
                "WHERE s.customer_id=?", (c["id"],)).fetchone()[0]
            paid = self.conn.execute("SELECT COALESCE(SUM(amount_paid),0) FROM sales WHERE customer_id=?",
                                     (c["id"],)).fetchone()[0]
            later = self.conn.execute("SELECT COALESCE(SUM(amount),0) FROM customer_payments WHERE customer_id=?",
                                      (c["id"],)).fetchone()[0]
            expected += max(0.0, billed - paid - later)
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(bt.get_customer_dues(self.conn, today=TODAY)["total_outstanding"], expected, places=2)

    def test_no_sale_is_overpaid_or_future_dated(self):
        over = self.conn.execute(
            "SELECT COUNT(*) FROM sales s WHERE s.amount_paid > "
            "(SELECT COALESCE(SUM(quantity*unit_price),0) FROM sale_items WHERE sale_id = s.id) + 0.005").fetchone()[0]
        self.assertEqual(over, 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM sales WHERE sale_date > ?", ("2026-03-15",)).fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM customer_payments WHERE payment_date > ?",
                                           ("2026-03-15",)).fetchone()[0], 0)

    def test_every_tool_runs_on_seed_data_and_sample_questions_have_data(self):
        for name in bt.TOOLS:
            args = _smoke_test_arguments(name)
            result = bt.execute_tool(self.conn, name, args, today=TODAY)
            self.assertNotEqual(result["status"], "error", f"{name}: {result}")
        self.assertEqual(bt.get_sales_summary(self.conn, "today", today=TODAY)["status"], "ok")
        self.assertGreater(bt.get_low_stock_products(self.conn, today=TODAY)["low_stock_count"], 0)
        self.assertGreater(bt.get_customer_dues(self.conn, today=TODAY)["total_outstanding"], 0)
        self.assertGreater(len(bt.get_expense_summary(self.conn, "this_month", today=TODAY)["by_category"]), 0)


def _smoke_test_arguments(name):
    if name in ("get_business_report", "generate_business_report"):
        return {"report_type": "daily"}
    if name == "get_sales_by_date":
        return {"sale_date": TODAY.isoformat()}
    if name == "compare_business_periods":
        return {"current_period": "this_month", "comparison_period": "last_month"}
    return {}


if __name__ == "__main__":
    unittest.main()
