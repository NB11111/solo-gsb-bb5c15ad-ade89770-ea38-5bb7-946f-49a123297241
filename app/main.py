"""FastAPI 入口：只做请求解析与 HTTP 状态映射，核心逻辑在 :mod:`app.anonymizer`。"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.anonymizer import MAX_EXPLICIT_LEVELS, MAX_FIELDS, AnonymizationError, anonymize

app = FastAPI(
    title="野外观测记录匿名化 API",
    description=(
        "对自然保护站公开的野外观测记录做 k-匿名全域泛化。"
        "服务为无状态计算：原始记录不落盘、不写日志。"
    ),
    version="1.0.0",
)


class AnonymizeRequest(BaseModel):
    """请求模型。所有语义校验（缺值、记录数不足等）在核心层完成并返回 422。"""

    k: int = Field(
        ...,
        strict=True,  # 禁止 "3" / true 等隐式转换
        description="每个泛化后字段值组合在发布结果中至少出现的次数（正整数）",
    )
    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="待匿名化的原始观测记录数组，记录可包含其他不做泛化的字段",
    )
    fields: dict[
        str,
        list[dict[str, str]],
    ] = Field(
        default_factory=dict,
        max_length=MAX_FIELDS,
        description=(
            f"待匿名字段（最多 {MAX_FIELDS} 个）；每个字段给出 1~"
            f"{MAX_EXPLICIT_LEVELS} 级映射，层级 0 为原值，层级 1 为第一个映射，"
            "依此类推；最后一级映射必须把全部原值收敛为统一通配符"
        ),
    )


class Group(BaseModel):
    combination: dict[str, str]
    count: int


class AnonymizeResponse(BaseModel):
    k: int
    total_records: int
    selected_levels: dict[str, int]
    groups: list[Group]
    generalized_records: list[dict[str, Any]]


@app.exception_handler(AnonymizationError)
async def anonymization_error_handler(
    _request: Request, exc: AnonymizationError
) -> JSONResponse:
    # 与 Pydantic 的 422 保持结构一致，detail 中给出可读的中文原因
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/health", tags=["运维"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/anonymize",
    response_model=AnonymizeResponse,
    tags=["匿名化"],
    summary="按 k-匿名泛化观测记录",
)
async def anonymize_records(request: AnonymizeRequest) -> dict[str, Any]:
    # model_dump 保留字典/列表结构；k 由 strict 模式保证为真正的 int
    return anonymize(request.model_dump())
