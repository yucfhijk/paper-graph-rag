# -*- coding: utf-8 -*-
"""common.py — 各步骤共用的工具函数。"""

import re


def norm_title(s):
    """标题归一化：小写、统一各种引号破折号、只保留字母数字。

    两个用途：
      1. 图检索做实体链接时的精确匹配
      2. 识别 OpenAlex 里的重复记录（同一篇论文的预印本版与正式版标题一致）
    """
    s = (s or "").lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("−", "-")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()
