"""本地 Skill 加载、范围过滤和 Agent Prompt 注入测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import asyncio

from agents.agent_orchestrator import GeneralAgent, Request, TechnicalAgent
from skills.manager import SkillManager


SKILLS_ROOT = Path(__file__).parents[1] / "skills"


def test_skill_manager_loads_nested_skill_documents() -> None:
    """SkillManager 应递归发现三个领域 Skill，而不是只扫描根目录。"""

    manager = SkillManager(str(SKILLS_ROOT))

    assert manager.skills == [
        "billing",
        "general",
        "technical",
    ]
    assert manager.specs["technical"].agents == ("technical",)
    assert manager.specs["billing"].enabled is True


def test_relative_skill_path_resolves_from_backend_package(monkeypatch) -> None:
    """从仓库根目录启动时，默认相对路径仍应找到 backend/skills。"""

    monkeypatch.chdir(Path(__file__).parents[2])

    manager = SkillManager("skills")

    assert "technical" in manager.skills


def test_skill_prompt_is_filtered_by_agent_scope() -> None:
    """不同 Agent 只能获得 frontmatter 声明给自己的 Skill。"""

    manager = SkillManager(str(SKILLS_ROOT))

    technical_prompt = manager.prompt_for_agent("technical")
    billing_prompt = manager.prompt_for_agent("billing")

    assert "技术支持处理规范" in technical_prompt
    assert "账单退款处理规范" not in technical_prompt
    assert "账单退款处理规范" in billing_prompt
    assert "技术支持处理规范" not in billing_prompt


class _FakeClient:
    """只记录 Agent 收到的 system Prompt 的异步客户端替身。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="已按 Skill 处理")])


def test_agent_injects_matching_skill_into_llm_system_prompt() -> None:
    """TechnicalAgent 调用 LLM 时应带入技术 Skill，而不是账单 Skill。"""

    client = _FakeClient()
    agent = TechnicalAgent(client=client, skill_manager=SkillManager(str(SKILLS_ROOT)))

    result = asyncio.run(agent.handle(Request(message="接口返回 401")))

    assert result.success is True
    assert "技术支持处理规范" in client.calls[0]["system"]
    assert "账单退款处理规范" not in client.calls[0]["system"]


def test_skill_prompt_is_capped() -> None:
    """Skill 文本超过配置上限时应截断，避免挤占模型上下文。"""

    manager = SkillManager(str(SKILLS_ROOT), max_prompt_chars=80)

    assert len(manager.prompt_for_agent("general")) <= 80
