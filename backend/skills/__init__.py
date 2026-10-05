"""ShopCare 本地 Skill 包。

该包只负责加载仓库内的 Markdown Skill 定义，不连接外部系统，也不执行
订单、支付、退款或工单写操作。
"""

from .skill_manager import SkillManager, SkillSpec

__all__ = ["SkillManager", "SkillSpec"]
