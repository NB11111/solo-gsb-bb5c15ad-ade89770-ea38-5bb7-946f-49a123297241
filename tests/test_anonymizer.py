"""app.anonymizer 核心算法测试（仅标准库）。"""

import unittest

from app.anonymizer import AnonymizationError, anonymize


def single_field_payload(values, k, maps):
    """构造单字段 'f' 的请求；maps 为各级显式映射列表。"""
    return {
        "k": k,
        "records": [{"f": v, "seq": i} for i, v in enumerate(values)],
        "fields": {"f": maps},
    }


def all_star_map(values, token="*"):
    return {v: token for v in values}


class TestBasics(unittest.TestCase):
    def test_level_zero_when_already_k_anonymous(self):
        values = ["a", "a", "b", "b"]
        result = anonymize(single_field_payload(values, 2, [all_star_map(values)]))
        self.assertEqual(result["selected_levels"], {"f": 0})
        self.assertEqual(
            [r["f"] for r in result["generalized_records"]], values
        )
        self.assertEqual(result["total_records"], 4)
        groups = {(g["combination"]["f"]): g["count"] for g in result["groups"]}
        self.assertEqual(groups, {"a": 2, "b": 2})

    def test_generalization_when_needed(self):
        values = ["a1", "a2", "b1", "b2"]
        level1 = {
            "a1": "A", "a2": "A", "b1": "B", "b2": "B",
        }
        result = anonymize(
            single_field_payload(values, 2, [level1, all_star_map(values)])
        )
        self.assertEqual(result["selected_levels"], {"f": 1})
        self.assertEqual(
            [r["f"] for r in result["generalized_records"]],
            ["A", "A", "B", "B"],
        )

    def test_until_top_level(self):
        values = ["a1", "a2", "a3"]
        level1 = {"a1": "A", "a2": "A", "a3": "B"}
        result = anonymize(
            single_field_payload(values, 3, [level1, all_star_map(values)])
        )
        self.assertEqual(result["selected_levels"], {"f": 2})
        self.assertTrue(all(r["f"] == "*" for r in result["generalized_records"]))
        self.assertEqual(result["groups"][0]["count"], 3)

    def test_record_order_and_passthrough_fields_preserved(self):
        values = ["x", "y"]
        result = anonymize(
            single_field_payload(values, 2, [all_star_map(values)])
        )
        self.assertEqual([r["seq"] for r in result["generalized_records"]], [0, 1])
        self.assertTrue(all("seq" in r for r in result["generalized_records"]))

    def test_no_anonymized_fields(self):
        payload = {"k": 2, "records": [{"a": 1}, {"a": 2}]}
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {})
        self.assertEqual(result["groups"], [{"combination": {}, "count": 2}])
        self.assertEqual(result["generalized_records"], [{"a": 1}, {"a": 2}])

    def test_k_equals_one_keeps_original(self):
        values = ["a", "b", "c"]
        result = anonymize(single_field_payload(values, 1, [all_star_map(values)]))
        self.assertEqual(result["selected_levels"], {"f": 0})


class TestOptimizationAndTieBreak(unittest.TestCase):
    def test_min_sum_across_fields(self):
        # 字段 a 需要层级 1 才能成组（k=2），字段 b 原值已满足
        records = [
            {"a": "a1", "b": "x"},
            {"a": "a2", "b": "x"},
        ]
        payload = {
            "k": 2,
            "records": records,
            "fields": {
                "a": [{"a1": "*", "a2": "*"}],
                "b": [{"x": "*"}],
            },
        }
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {"a": 1, "b": 0})

    def test_no_tie_when_partial_generalization_still_singletons(self):
        # 两字段各 2 个原值：层级 1 的映射不做合并（a1->A1,a2->A2），
        # 因此混合提升 (1,2)/(2,1) 仍留下大小 1 的联合组合，必须都到 2 级。
        records = [
            {"a": "a1", "b": "b1"},
            {"a": "a2", "b": "b2"},
        ]
        a_levels = [
            {"a1": "A1", "a2": "A2"},
            {"a1": "*", "a2": "*"},
        ]
        b_levels = [
            {"b1": "B1", "b2": "B2"},
            {"b1": "*", "b2": "*"},
        ]
        payload = {
            "k": 2,
            "records": records,
            "fields": {"a": a_levels, "b": b_levels},
        }
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {"a": 2, "b": 2})

    def test_tie_break_real(self):
        # 4 条记录下两个可行解层级和相同：
        #   (a=0,b=1) 与 (a=1,b=0) 都能得到 2 个大小为 2 的组。
        # 字段名升序元组字典序 (0,1) < (1,0)，应选 (a=0,b=1)。
        records = [
            {"a": "a1", "b": "b1"},
            {"a": "a2", "b": "b1"},
            {"a": "a1", "b": "b2"},
            {"a": "a2", "b": "b2"},
        ]
        payload = {
            "k": 2,
            "records": records,
            "fields": {
                "a": [{"a1": "*", "a2": "*"}],
                "b": [{"b1": "*", "b2": "*"}],
            },
        }
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {"a": 0, "b": 1})
        # 泛化结果：a 保留原值，b 变 *
        records_out = result["generalized_records"]
        self.assertEqual([r["a"] for r in records_out], ["a1", "a2", "a1", "a2"])
        self.assertTrue(all(r["b"] == "*" for r in records_out))

    def test_groups_counted_on_joint_combination(self):
        # 两字段：联合组合在层级 0 时各 1；单提升任一字段到 1 级即可出现 2 个二元组。
        # (a=0,b=1) 与 (a=1,b=0) 同和，字典序选 (a=0,b=1)。
        records = [
            {"a": "a1", "b": "b1"},
            {"a": "a1", "b": "b2"},
            {"a": "a2", "b": "b1"},
            {"a": "a2", "b": "b2"},
        ]
        payload = {
            "k": 2,
            "records": records,
            "fields": {
                "a": [{"a1": "*", "a2": "*"}],
                "b": [{"b1": "*", "b2": "*"}],
            },
        }
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {"a": 0, "b": 1})


class TestErrors422(unittest.TestCase):
    def assert422(self, payload):
        with self.assertRaises(AnonymizationError):
            anonymize(payload)

    def test_records_fewer_than_k(self):
        self.assert422(
            {"k": 3, "records": [{"f": "a"}], "fields": {"f": [{"a": "*"}]}}
        )

    def test_empty_records(self):
        self.assert422({"k": 1, "records": []})

    def test_missing_mapping_value(self):
        payload = single_field_payload(["a", "b"], 2, [{"a": "*"}])
        self.assert422(payload)

    def test_missing_value_at_second_level(self):
        values = ["a", "b"]
        payload = single_field_payload(
            values,
            2,
            [{"a": "A", "b": "B"}, {"a": "*"}],  # 第二级缺 b
        )
        self.assert422(payload)

    def test_top_level_not_unified(self):
        values = ["a", "b", "c"]
        payload = single_field_payload(values, 3, [{"a": "A", "b": "B", "c": "C"}])
        self.assert422(payload)

    def test_too_many_fields(self):
        payload = {
            "k": 1,
            "records": [{"a": "1", "b": "2", "c": "3", "d": "4"}],
            "fields": {
                f: [{"1": "*", "2": "*", "3": "*", "4": "*"}]
                for f in ("a", "b", "c", "d")
            },
        }
        self.assert422(payload)

    def test_too_many_levels(self):
        values = ["a"]
        payload = single_field_payload(values, 1, [all_star_map(values)] * 4)
        self.assert422(payload)

    def test_zero_levels(self):
        payload = single_field_payload(["a"], 1, [])
        self.assert422(payload)

    def test_bad_k(self):
        for bad_k in (0, -1, "2", 1.5, True, False, None):
            with self.subTest(k=bad_k):
                self.assert422({"k": bad_k, "records": [{"f": "a"}]})

    def test_missing_k_or_records(self):
        self.assert422({"records": []})
        self.assert422({"k": 1})

    def test_non_string_field_value(self):
        self.assert422(
            {"k": 1, "records": [{"f": 1}], "fields": {"f": [{"1": "*"}]}}
        )
        self.assert422(
            {"k": 1, "records": [{"f": None}], "fields": {"f": []}}
        )
        self.assert422({"k": 1, "records": [{}], "fields": {"f": []}})

    def test_records_not_list_or_object(self):
        self.assert422({"k": 1, "records": "nope"})
        self.assert422({"k": 1, "records": ["nope"]})
        self.assert422({"k": 1, "records": [{"f": "a"}], "fields": "nope"})

    def test_unknown_field_only_extra_mapping_keys_allowed(self):
        # 映射里多出未出现的原值键是允许的（发布数据可能只含部分原值）
        result = anonymize(
            {
                "k": 2,
                "records": [{"f": "a"}, {"f": "a"}],
                "fields": {"f": [{"a": "*", "unused": "*"}]},
            }
        )
        self.assertEqual(result["selected_levels"], {"f": 0})


class TestFourLevelBoundary(unittest.TestCase):
    def test_three_explicit_levels_accepted(self):
        values = ["a", "b"]
        payload = single_field_payload(
            values,
            2,
            [
                {"a": "A1", "b": "B1"},
                {"a": "A2", "b": "B2"},
                {"a": "*", "b": "*"},
            ],
        )
        result = anonymize(payload)
        self.assertEqual(result["selected_levels"], {"f": 3})

    def test_three_fields_four_levels_combo_space(self):
        # 3 字段 × (0..3) 共 64 种组合，冒烟检查结果有效
        records = [
            {"a": "a1", "b": "b1", "c": "c1", "i": 1},
            {"a": "a2", "b": "b2", "c": "c2", "i": 2},
        ]
        star = {"a1": "*", "a2": "*"}
        stb = {"b1": "*", "b2": "*"}
        stc = {"c1": "*", "c2": "*"}
        payload = {
            "k": 2,
            "records": records,
            "fields": {
                "a": [dict(star), dict(star), dict(star)],
                "b": [dict(stb), dict(stb), dict(stb)],
                "c": [dict(stc), dict(stc), dict(stc)],
            },
        }
        result = anonymize(payload)
        # 两条记录：任一字段保持层级 0 时该字段值仍两两不同，联合组大小为 1；
        # 唯一可行解是三字段全部到层级 1（和 3，也是最小）。
        self.assertEqual(
            result["selected_levels"], {"a": 1, "b": 1, "c": 1}
        )
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["count"], 2)


if __name__ == "__main__":
    unittest.main()
