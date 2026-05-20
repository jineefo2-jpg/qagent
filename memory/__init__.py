"""
memory — 长记忆模块

两层架构：
  Layer 1 (profile.py):  结构化用户档案（偏好/股池/目标/背景），Redis 存储
  Layer 2 (episodic.py): 情景记忆（自由文本事实/分析结论），Chroma 向量库

仅登录用户可用。匿名用户跳过所有长记忆。
"""
from .profile import (
    UserProfile,
    get_profile,
    update_profile_fields,
    clear_profile,
    profile_summary_text,
)
from .episodic import (
    record_memory,
    search_memories,
    list_memories,
    delete_memory,
    clear_user_memories,
    memories_to_prompt_text,
    prune_stale_memories,
    memory_stats,
)

__all__ = [
    # Layer 1 (profile)
    "UserProfile",
    "get_profile",
    "update_profile_fields",
    "clear_profile",
    "profile_summary_text",
    # Layer 2 (episodic)
    "record_memory",
    "search_memories",
    "list_memories",
    "delete_memory",
    "clear_user_memories",
    "memories_to_prompt_text",
    "prune_stale_memories",
    "memory_stats",
]
