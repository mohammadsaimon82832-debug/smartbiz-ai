"""API tests. They need fastapi + httpx (pip install -r requirements.txt) and are skipped otherwise."""
import os
import tempfile
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:  # pragma: no cover
    HAVE_FASTAPI = False

import database
from test_tools import build_fixture


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "api.db")
        conn = database.get_connection(self.db_path)
        build_fixture(conn)
        conn.close()
        self.env = mock.patch.dict(os.environ, {"SMARTBIZ_DB": self.db_path, "OLLAMA_MODEL": "qwen-test",
                                                "SMARTBIZ_API_TOKEN": ""})
        self.env.start()
        import main
        self.main = main
        main._agent = None
        self.client_cm = TestClient(main.app)
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)
        self.env.stop()
        self.tmp.cleanup()

    def test_health_and_index(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "ai_configured": True})
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_sales_summary_uses_real_data(self):
        r = self.client.get("/api/sales/summary", params={"start_date": "2026-03-15", "end_date": "2026-03-15"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total_sales"], 165.0)
        self.assertEqual(r.json()["gross_profit"], 89.0)

    def test_dashboard_uses_real_business_facts(self):
        result = self.client.get("/api/dashboard?period=all_time")
        self.assertEqual(result.status_code, 200)
        data = result.json()
        self.assertEqual(data["kpis"]["total_sales"], 380.0)
        self.assertEqual(data["kpis"]["gross_profit"], 192.0)
        self.assertEqual(data["kpis"]["total_expenses"], 250.0)
        self.assertEqual(data["kpis"]["outstanding_sales_balances"], 145.0)
        self.assertEqual(data["currency"], "USD")
        self.assertEqual(len(data["daily_sales"]), 14)
        self.assertTrue(data["recent_transactions"])

    def test_product_customer_expense_and_sale_workflows(self):
        created_product = self.client.post("/api/products", json={
            "sku": "NEW-1", "name": "New item", "category": "Test",
            "cost_price": 2, "selling_price": 5, "stock_qty": 10, "reorder_level": 2,
        })
        self.assertEqual(created_product.status_code, 201)
        product_id = created_product.json()["product_id"]
        customer_response = self.client.post("/api/customers", json={"name": "Test customer", "phone": "123"})
        self.assertEqual(customer_response.status_code, 201)
        customer_id = customer_response.json()["customer_id"]
        sale_response = self.client.post("/api/sales", json={
            "sale_date": "2026-03-15", "customer_id": customer_id, "amount_paid": 5,
            "request_id": "74e19f7a-21d2-4f55-a5d2-5618092ce720",
            "items": [{"product_id": product_id, "quantity": 2}],
        })
        self.assertEqual(sale_response.status_code, 201)
        self.assertEqual(sale_response.json()["total"], 10)
        duplicate = self.client.post("/api/sales", json={
            "sale_date": "2026-03-15", "customer_id": customer_id, "amount_paid": 5,
            "request_id": "74e19f7a-21d2-4f55-a5d2-5618092ce720",
            "items": [{"product_id": product_id, "quantity": 2}],
        })
        self.assertTrue(duplicate.json()["duplicate"])
        listed = self.client.get("/api/products").json()["products"]
        self.assertEqual(next(product["stock_qty"] for product in listed if product["id"] == product_id), 8)
        payment = self.client.post(f"/api/customers/{customer_id}/payments", json={
            "payment_date": "2026-03-15", "amount": 5,
        })
        self.assertEqual(payment.status_code, 201)
        expense = self.client.post("/api/expenses", json={
            "expense_date": "2026-03-15", "category": "Test", "amount": 4,
        })
        self.assertEqual(expense.status_code, 201)
        self.assertEqual(self.client.get("/api/sales/summary?period=today").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sales/{sale_response.json()['sale_id']}").status_code, 200)

    def test_sale_validation_does_not_change_stock(self):
        before = self.client.get("/api/products").json()["products"][0]["stock_qty"]
        response = self.client.post("/api/sales", json={
            "sale_date": "2026-03-15", "items": [{"product_id": 1, "quantity": before + 1}],
        })
        self.assertEqual(response.status_code, 422)
        after = self.client.get("/api/products").json()["products"][0]["stock_qty"]
        self.assertEqual(after, before)

    def test_validation_errors(self):
        self.assertEqual(self.client.get("/api/sales/summary", params={"period": "forever"}).status_code, 422)
        self.assertEqual(self.client.get("/api/sales/summary", params={"start_date": "bad"}).status_code, 422)
        self.assertEqual(self.client.get("/api/sales/summary",
                                         params={"start_date": "2026-03-10", "end_date": "2026-03-01"}).status_code, 422)
        self.assertEqual(self.client.get("/api/reports/yearly").status_code, 422)
        self.assertEqual(self.client.get("/api/sales/top-products", params={"limit": 0}).status_code, 422)

    def test_other_endpoints(self):
        for path in ("/api/inventory", "/api/inventory/low-stock", "/api/inventory/restock", "/api/customers/dues",
                     "/api/expenses/summary", "/api/recommendations", "/api/sales/top-products"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        self.assertEqual(self.client.get("/api/customers/dues").json()["total_outstanding"], 145.0)

    def test_reports_json_and_text(self):
        r = self.client.get("/api/reports/monthly", params={"report_date": "2026-03-15"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"]["total_sales"], 330.0)
        t = self.client.get("/api/reports/weekly", params={"report_date": "2026-03-15", "format": "text"})
        self.assertIn("WEEKLY BUSINESS REPORT", t.text)

    def test_chat_validation(self):
        self.assertEqual(self.client.post("/api/chat", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"question": "   "}).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"question": "x" * 1001}).status_code, 422)
        bad_role = {"question": "hi", "history": [{"role": "system", "content": "x"}]}
        self.assertEqual(self.client.post("/api/chat", json=bad_role).status_code, 422)

    def test_chat_with_fake_agent(self):
        from agent import AgentResult
        fake = mock.Mock()
        fake.ask.return_value = AgentResult(answer="ok", tools_used=[{"name": "get_sales_summary",
                                                                       "arguments": {}, "status": "ok"}], model="m")
        with mock.patch.object(self.main, "get_agent", return_value=fake):
            r = self.client.post("/api/chat", json={"question": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "ok")
        self.assertEqual(r.json()["tools_used"][0]["name"], "get_sales_summary")

    def test_api_token_enforced_when_configured(self):
        with mock.patch.dict(os.environ, {"SMARTBIZ_API_TOKEN": "secret"}):
            self.assertEqual(self.client.get("/api/inventory").status_code, 401)
            self.assertEqual(self.client.get("/api/inventory", headers={"X-API-Key": "wrong"}).status_code, 401)
            self.assertEqual(self.client.get("/api/inventory", headers={"X-API-Key": "secret"}).status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_api_cannot_write(self):
        for method in ("post", "put", "delete", "patch"):
            r = getattr(self.client, method)("/api/sales/summary")
            self.assertIn(r.status_code, (404, 405))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class TestApiEmptyDb(unittest.TestCase):
    def test_empty_database_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.db")
            with mock.patch.dict(os.environ, {"SMARTBIZ_DB": path, "SMARTBIZ_API_TOKEN": ""}):
                import main
                with TestClient(main.app) as client:  # startup creates the empty schema
                    for url in ("/api/sales/summary", "/api/inventory/low-stock", "/api/customers/dues",
                                "/api/expenses/summary", "/api/recommendations", "/api/reports/daily"):
                        self.assertEqual(client.get(url).status_code, 200, url)
                    self.assertEqual(client.get("/api/sales/summary").json()["total_sales"], 0.0)


if __name__ == "__main__":
    unittest.main()
