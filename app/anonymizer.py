"""k-匿名泛化核心算法（仅依赖标准库，便于独立测试）。

规则
----
* 待匿名字段最多 3 个；每个字段提供 1~3 级显式映射，加上"原值"层级，共至多 4 级。
* 层级 0 恒等（使用原始值）；层级 i (i>=1) 由请求中第 i 个映射决定，
  该层级把原值映射为统一的通配泛化值；最后一级必须把该字段在全部记录中
  出现的所有原值收敛为同一个值（统一通配符），从而保证方案可行。
* 对全部记录统一选择每字段层级，使泛化后每种字段值组合至少出现 k 次。
* 在可行方案中最小化层级序号之和；同分时按字段名升序排列的层级序号
  元组做字典序裁决（优先压低字段名字典序更小的字段的层级）。

任何输入语义错误都抛出 :class:`AnonymizationError`，由 API 层映射为 HTTP 422。
"""

from collections import Counter
from itertools import product
from typing import Any

# 协议约束
MAX_FIELDS = 3          # 最多三个待匿名字段
MAX_LEVELS = 4          # 每字段至多四级（含恒等的第 0 级）
MAX_EXPLICIT_LEVELS = MAX_LEVELS - 1


class AnonymizationError(ValueError):
    """输入可解析为 JSON，但不满足匿名化语义约束（对应 HTTP 422）。"""


def anonymize(payload: Any) -> dict:
    """执行 k-匿名泛化，返回可直接序列化的结果字典。

    payload 结构::

        {
          "k": int,
          "records": [{...}, ...],
          "fields": {
              "<字段名>": [
                  {"<原值>": "<泛化值>", ...},   # 层级 1
                  {"<原值>": "<泛化值>", ...},   # 层级 2（可选）
                  {"<原值>": "<*>", ...}         # 层级 3（可选，统一通配符）
              ]
          }
        }
    """
    k = _require_k(payload)
    records = _require_records(payload)
    fields_spec = _require_fields_spec(payload)

    if len(records) < k:
        raise AnonymizationError(
            f"记录总数 {len(records)} 小于 k={k}，任何发布方案都无法满足 k-匿名"
        )

    field_names = sorted(fields_spec)

    # 每字段: levels[i] = {原值: 泛化值}；levels[0] 恒等映射按需构造。
    levels_by_field: dict[str, list[dict[str, str]]] = {}
    distinct_values: dict[str, set[str]] = {}

    for name in field_names:
        explicit_maps = fields_spec[name]
        original_values = _collect_field_values(records, name)
        distinct_values[name] = original_values

        levels: list[dict[str, str]] = []
        for depth, mapping in enumerate(explicit_maps, start=1):
            values_to_generalized: dict[str, str] = {}
            for original in original_values:
                if original not in mapping:
                    raise AnonymizationError(
                        f"字段 {name!r} 第 {depth} 级映射缺少原值 {original!r} 的映射"
                    )
                generalized = mapping[original]
                if not isinstance(generalized, str) or generalized == "":
                    raise AnonymizationError(
                        f"字段 {name!r} 第 {depth} 级映射中，原值 {original!r} "
                        f"的泛化值必须是非空字符串"
                    )
                values_to_generalized[original] = generalized
            levels.append(values_to_generalized)

        # 最高一级必须把所有出现过的原值收敛为统一通配符，保证存在可行方案。
        top_values = set(levels[-1].values())
        if len(top_values) != 1:
            raise AnonymizationError(
                f"字段 {name!r} 的最高一级映射必须把全部原值收敛为统一通配符，"
                f"当前收敛为 {sorted(top_values)!r}"
            )

        levels_by_field[name] = levels

    # 枚举全部层级组合（至多 4^3 = 64 种），按 (层级序号之和, 字段名升序层级元组) 取最优。
    best_levels: dict[str, int] | None = None
    best_key: tuple[int, tuple[int, ...]] | None = None

    choices = [range(len(levels_by_field[name]) + 1) for name in field_names]
    for combo in product(*choices):
        chosen = dict(zip(field_names, combo))
        counts = _group_counts(records, field_names, chosen, levels_by_field)
        if counts and min(counts.values()) >= k:
            key = (sum(combo), combo)  # combo 本身就是字段名升序的层级元组
            if best_key is None or key < best_key:
                best_key = key
                best_levels = chosen

    if best_levels is None:  # 理论上不可达：最高层级组合一定可行，留作防御性检查
        raise AnonymizationError("不存在满足 k-匿名的统一泛化层级方案")

    generalized_records = [
        _generalize_record(record, field_names, best_levels, levels_by_field)
        for record in records
    ]

    counts = _group_counts(records, field_names, best_levels, levels_by_field)
    groups = [
        {"combination": dict(zip(field_names, combo)), "count": count}
        for combo, count in sorted(counts.items(), key=lambda item: item[0])
    ]

    return {
        "k": k,
        "total_records": len(records),
        "selected_levels": {name: best_levels[name] for name in field_names},
        "groups": groups,
        "generalized_records": generalized_records,
    }


# ---------- 输入校验 ----------

def _require_k(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise AnonymizationError("请求体必须是 JSON 对象")
    if "k" not in payload:
        raise AnonymizationError("缺少必填字段 k")
    k = payload["k"]
    # 显式拒绝 bool：Python 中 True/False 是 int 的子类
    if isinstance(k, bool) or not isinstance(k, int):
        raise AnonymizationError("k 必须是正整数")
    if k < 1:
        raise AnonymizationError("k 必须是正整数")
    return k


def _require_records(payload: dict) -> list[Any]:
    if "records" not in payload:
        raise AnonymizationError("缺少必填字段 records")
    records = payload["records"]
    if not isinstance(records, list):
        raise AnonymizationError("records 必须是数组")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AnonymizationError(f"第 {index} 条记录必须是 JSON 对象")
    return records


def _require_fields_spec(payload: dict) -> dict[str, list]:
    fields_spec = payload.get("fields", {})
    if fields_spec is None:
        return {}
    if not isinstance(fields_spec, dict):
        raise AnonymizationError("fields 必须是以字段名为键的对象")
    if len(fields_spec) > MAX_FIELDS:
        raise AnonymizationError(f"待匿名字段最多 {MAX_FIELDS} 个")

    normalized: dict[str, list] = {}
    for name, levels in fields_spec.items():
        if not isinstance(name, str) or name == "":
            raise AnonymizationError("待匿名的字段名必须是非空字符串")
        if not isinstance(levels, list):
            raise AnonymizationError(f"字段 {name!r} 的映射必须是数组")
        if not 1 <= len(levels) <= MAX_EXPLICIT_LEVELS:
            raise AnonymizationError(
                f"字段 {name!r} 必须提供 1~{MAX_EXPLICIT_LEVELS} 级显式映射"
                f"（连同原值层级共至多 {MAX_LEVELS} 级）"
            )
        for depth, mapping in enumerate(levels, start=1):
            if not isinstance(mapping, dict):
                raise AnonymizationError(
                    f"字段 {name!r} 第 {depth} 级映射必须是对象"
                )
            for original, generalized in mapping.items():
                if not isinstance(original, str) or original == "":
                    raise AnonymizationError(
                        f"字段 {name!r} 第 {depth} 级映射的原值必须是非空字符串"
                    )
                if not isinstance(generalized, str) or generalized == "":
                    raise AnonymizationError(
                        f"字段 {name!r} 第 {depth} 级映射中，原值 {original!r} "
                        f"的泛化值必须是非空字符串"
                    )
        normalized[name] = levels
    return normalized


def _collect_field_values(records: list[dict], field_name: str) -> set[str]:
    values: set[str] = set()
    for index, record in enumerate(records):
        if field_name not in record:
            raise AnonymizationError(
                f"第 {index} 条记录缺少待匿名字段 {field_name!r}"
            )
        value = record[field_name]
        if not isinstance(value, str) or value == "":
            raise AnonymizationError(
                f"第 {index} 条记录的字段 {field_name!r} 必须是非空字符串"
                f"（null 或缺失值请先在数据中补齐）"
            )
        values.add(value)
    return values


# ---------- 泛化与分组 ----------

def _generalize_one(value: str, level: int, levels: list[dict[str, str]]) -> str:
    return value if level == 0 else levels[level - 1][value]


def _signature(
    record: dict,
    field_names: list[str],
    chosen: dict[str, int],
    levels_by_field: dict[str, list[dict[str, str]]],
) -> tuple[str, ...]:
    return tuple(
        _generalize_one(
            record[name], chosen[name], levels_by_field[name]
        )
        for name in field_names
    )


def _group_counts(
    records: list[dict],
    field_names: list[str],
    chosen: dict[str, int],
    levels_by_field: dict[str, list[dict[str, str]]],
) -> dict[tuple[str, ...], int]:
    counter: Counter = Counter(
        _signature(record, field_names, chosen, levels_by_field)
        for record in records
    )
    return dict(counter)


def _generalize_record(
    record: dict,
    field_names: list[str],
    chosen: dict[str, int],
    levels_by_field: dict[str, list[dict[str, str]]],
) -> dict:
    result = dict(record)
    for name in field_names:
        result[name] = _generalize_one(
            record[name], chosen[name], levels_by_field[name]
        )
    return result
