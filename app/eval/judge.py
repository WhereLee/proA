"""LLM-as-judge grader (checklist P0: 固定评审 prompt + 温度 0 + 双评 + 仲裁)."""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.base import ChatMessage

JUDGE_SYSTEM = "你是严格的评测员，只输出一个 JSON 对象，不要任何多余文字。"

JUDGE_TEMPLATE = """请评估下面这次 Agent 运行是否真正完成了用户请求。

用户请求：{user_input}
期望调用的工具：{expected_tools}
实际工具轨迹：{actual_trace}
最终回答：{answer}

判定标准：信息是否准确、动作是否到位、是否存在编造或未完成。
只输出 JSON：{{"passed": true 或 false, "reason": "简要理由"}}"""


class JudgeVerdict(BaseModel):
    passed: bool
    reason: str = ""


def _trace_text(run) -> str:
    parts = [f"{s.action}:{s.tool}" if s.tool else s.action for s in run.steps]
    return " -> ".join(parts) if parts else "(无)"


async def judge_once(llm, task: dict, run) -> JudgeVerdict:
    messages = [
        ChatMessage("system", JUDGE_SYSTEM),
        ChatMessage(
            "user",
            JUDGE_TEMPLATE.format(
                user_input=task.get("input", ""),
                expected_tools=", ".join(task.get("expected_tools", [])) or "(无)",
                actual_trace=_trace_text(run),
                answer=run.answer or "(无)",
            ),
        ),
    ]
    # fixed temperature 0 for a stable judgement
    result = await llm.complete(messages, temperature=0.0)
    try:
        from app.llm.structured import parse_model

        return parse_model(result.text, JudgeVerdict)
    except Exception:  # noqa: BLE001 - judge output must never crash the eval
        return JudgeVerdict(passed=False, reason="judge output unparsable")


async def judge_run(llm, task: dict, run, *, passes: int = 2) -> dict:
    """Double evaluation with conservative arbitration (all passes must agree)."""
    verdicts = [await judge_once(llm, task, run) for _ in range(passes)]
    flags = {v.passed for v in verdicts}
    return {
        "passed": all(v.passed for v in verdicts),
        "disagreement": len(flags) > 1,
        "verdicts": [v.model_dump() for v in verdicts],
    }
