"""HTTP 端到端测试；未安装 fastapi/httpx 时自动跳过。"""

import unittest

try:
    from fastapi.testclient import TestClient

    from app.main import app

    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False


@unittest.skipUnless(_HAS_DEPS, "fastapi/httpx 未安装，跳过 HTTP 端到端测试")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_success_readme_example(self):
        payload = {
            "k": 2,
            "records": [
                {"id": "r1", "region": "云南-昆明-2024", "observer": "张三", "species": "黑颈鹤"},
                {"id": "r2", "region": "云南-昆明-2023", "observer": "李四", "species": "黑颈鹤"},
                {"id": "r3", "region": "云南-大理-2024", "observer": "王五", "species": "灰鹤"},
                {"id": "r4", "region": "四川-成都-2024", "observer": "赵六", "species": "灰鹤"},
                {"id": "r5", "region": "四川-成都-2022", "observer": "孙七", "species": "黑颈鹤"},
                {"id": "r6", "region": "四川-绵阳-2023", "observer": "周八", "species": "灰鹤"},
            ],
            "fields": {
                "region": [
                    {"云南-昆明-2024": "云南-昆明-*", "云南-昆明-2023": "云南-昆明-*",
                     "云南-大理-2024": "云南-大理-*",
                     "四川-成都-2024": "四川-成都-*", "四川-成都-2022": "四川-成都-*",
                     "四川-绵阳-2023": "四川-绵阳-*"},
                    {"云南-昆明-2024": "云南-*-*", "云南-昆明-2023": "云南-*-*",
                     "云南-大理-2024": "云南-*-*",
                     "四川-成都-2024": "四川-*-*", "四川-成都-2022": "四川-*-*",
                     "四川-绵阳-2023": "四川-*-*"},
                    {"云南-昆明-2024": "*-*-*", "云南-昆明-2023": "*-*-*",
                     "云南-大理-2024": "*-*-*",
                     "四川-成都-2024": "*-*-*", "四川-成都-2022": "*-*-*",
                     "四川-绵阳-2023": "*-*-*"},
                ],
                "observer": [
                    {"张三": "*", "李四": "*", "王五": "*",
                     "赵六": "*", "孙七": "*", "周八": "*"}
                ],
            },
        }
        resp = self.client.post("/anonymize", json=payload)
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["selected_levels"], {"observer": 1, "region": 2})
        self.assertEqual(len(body["generalized_records"]), 6)
        self.assertEqual({g["count"] for g in body["groups"]}, {3})

    def test_422_records_fewer_than_k(self):
        resp = self.client.post(
            "/anonymize",
            json={"k": 2, "records": [{"f": "a"}], "fields": {"f": [{"a": "*"}]}},
        )
        self.assertEqual(resp.status_code, 422)

    def test_422_missing_mapping(self):
        resp = self.client.post(
            "/anonymize",
            json={
                "k": 2,
                "records": [{"f": "a"}, {"f": "b"}],
                "fields": {"f": [{"a": "*"}]},
            },
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("缺少原值", resp.json()["detail"])

    def test_422_k_string_and_bool(self):
        base = {"records": [{"f": "a"}, {"f": "a"}], "fields": {"f": [{"a": "*"}]}}
        for bad in ("2", True, 0, -3, 1.5):
            resp = self.client.post("/anonymize", json={"k": bad, **base})
            self.assertEqual(resp.status_code, 422, bad)

    def test_422_too_many_fields(self):
        rec = {f: "1" for f in "abcd"}
        resp = self.client.post(
            "/anonymize",
            json={
                "k": 1,
                "records": [rec],
                "fields": {f: [{"1": "*"}] for f in "abcd"},
            },
        )
        self.assertEqual(resp.status_code, 422)

    def test_422_null_field_value(self):
        resp = self.client.post(
            "/anonymize",
            json={"k": 1, "records": [{"f": None}], "fields": {"f": [{}]}},
        )
        self.assertEqual(resp.status_code, 422)

    def test_422_malformed_json(self):
        resp = self.client.post(
            "/anonymize", content="{not json", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
