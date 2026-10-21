"""ShopCare 本地 Skill 加载与 Agent Prompt 注入模块。

本模块负责递归扫描本地 ``SKILL.md`` 文件、解析有限的 YAML frontmatter，
并按 Skill 声明的 ``agents`` 范围生成只读 Prompt 片段。它不负责 HTTP 路由、
模型调用或任何外部业务系统操作。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """表示一个已加载的本地 Markdown Skill。

    Args:
        skill_id: 稳定的 Skill 标识，通常取其目录名。
        name: frontmatter 中面向用户的 Skill 名称。
        description: Skill 的用途说明。
        keywords: Skill 声明的匹配关键词。
        agents: 允许注入该 Skill 的 Agent 类型集合。
        enabled: 是否允许参与 Prompt 注入。
        content: 去除 frontmatter 后的 Markdown 正文。
        path: Skill 源文件路径。
    """

    skill_id: str
    name: str
    description: str
    keywords: Tuple[str, ...]
    agents: Tuple[str, ...]
    enabled: bool
    content: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], str]:
    """解析 Skill 文件头部的有限键值格式。

    Args:
        text: 完整的 Markdown 文件内容。

    Returns:
        ``(metadata, body)``；没有合法 frontmatter 时返回空元数据和原文。

    说明：Skill 文件只使用简单标量和逗号分隔列表，因此使用标准库解析，
    避免为配置文本引入未锁定的 YAML 第三方依赖。
    """

    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized.strip()
    end = normalized.find("\n---", 4)
    if end < 0:
        return {}, normalized.strip()
    raw_header = normalized[4:end]
    body = normalized[end + len("\n---") :].lstrip("\n")
    metadata: Dict[str, str] = {}
    for line in raw_header.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, body.strip()


def _split_csv(value: str) -> Tuple[str, ...]:
    """把 frontmatter 的逗号分隔值转换为去重后的元组。"""

    values = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(dict.fromkeys(values))


class SkillManager:
    """扫描、缓存并按 Agent 范围提供本地 Skill Prompt。

    Args:
        skills_dir: Skill 根目录；支持根目录 Markdown 和嵌套的 ``*/SKILL.md``。
        max_prompt_chars: 单个 Agent 注入的 Skill 文本上限，避免上下文无限膨胀。

    Returns:
        无；通过 ``skills``、``reload_skills`` 和 ``prompt_for_agent`` 读取状态。
    """

    def __init__(self, skills_dir: str = "skills", max_prompt_chars: int = 5000) -> None:
        resolved_dir = Path(skills_dir)
        # Why：开发者可能从仓库根目录以 ``--app-dir backend`` 启动，相对路径也必须稳定指向后端 Skill 目录。
        if not resolved_dir.is_absolute() and not resolved_dir.exists():
            backend_relative = Path(__file__).resolve().parent.parent / resolved_dir
            if backend_relative.exists():
                resolved_dir = backend_relative
        self.skills_dir = resolved_dir
        self.max_prompt_chars = max(0, int(max_prompt_chars))
        self.skills: List[str] = []
        self.specs: Dict[str, SkillSpec] = {}
        self.reload_skills()

    def _skill_paths(self) -> List[Path]:
        """返回按路径排序的根目录及嵌套 Skill 文件。"""

        if not self.skills_dir.is_dir():
            return []
        paths = list(self.skills_dir.glob("*.md"))
        paths.extend(self.skills_dir.rglob("SKILL.md"))
        return sorted(set(paths), key=lambda path: str(path).lower())

    def _load_path(self, path: Path) -> SkillSpec:
        """读取单个 Skill 文件并构造不可变描述。"""

        metadata, content = _parse_frontmatter(path.read_text(encoding="utf-8"))
        skill_id = path.parent.name if path.parent != self.skills_dir else path.stem
        agents = _split_csv(metadata.get("agents", ""))
        return SkillSpec(
            skill_id=skill_id,
            name=metadata.get("name", skill_id),
            description=metadata.get("description", ""),
            keywords=_split_csv(metadata.get("keywords", "")),
            agents=tuple(agent.lower() for agent in agents),
            enabled=metadata.get("enabled", "true").lower() == "true",
            content=content,
            path=path,
        )

    def reload_skills(self) -> Dict[str, object]:
        """重新扫描本地 Skill，并原子替换当前缓存。

        Returns:
            包含 ``success``、Skill 标识列表、数量和加载错误列表的快照。

        说明：单个文件损坏时只跳过该文件，保留其他有效 Skill，避免热加载
        失败导致整个 API 生命周期不可用。
        """

        loaded: Dict[str, SkillSpec] = {}
        errors: List[str] = []
        for path in self._skill_paths():
            try:
                spec = self._load_path(path)
                loaded[spec.skill_id] = spec
            except Exception as exc:  # pragma: no cover - 只在文件系统异常时触发
                logger.warning("Skill 加载失败: %s (%s)", path, exc)
                errors.append(f"{path}: {exc}")
        self.specs = loaded
        self.skills = sorted(loaded)
        return {
            "success": True,
            "skills": list(self.skills),
            "count": len(self.skills),
            "errors": errors,
        }

    def prompt_for_agent(self, agent_type: str) -> str:
        """返回指定 Agent 可见的 Skill Prompt 文本。

            Args:
        agent_type: Agent 类型，例如 ``consult``、``order`` 或 ``after_sales``。

        Returns:
            只包含启用且声明该 Agent 范围的 Skill 正文，超过上限时截断。
        """

        target = str(getattr(agent_type, "value", agent_type)).lower()
        sections: List[str] = []
        for spec in self.specs.values():
            if not spec.enabled or target not in spec.agents:
                continue
            keyword_text = "、".join(spec.keywords)
            sections.append(
                f"[Skill：{spec.name}]\n"
                f"用途：{spec.description}\n"
                f"适用主题：{keyword_text}\n"
                f"{spec.content}"
            )
        # Why：5000 字符上限控制每轮 Tool Use 的上下文成本，避免 Skill 文本挤占用户问题和检索证据。
        return "\n\n".join(sections)[: self.max_prompt_chars]

    async def close(self) -> None:
        """关闭本地 Skill 管理器；当前无外部资源需要释放。"""


__all__ = ["SkillManager", "SkillSpec"]
