# -*- coding: utf-8 -*-
"""
01_fetch.py — 抓取「图结构 × 大语言模型」方向论文元数据

数据源选 OpenAlex（https://openalex.org）而非 arXiv：
  1. arXiv 共享出口常被限流（实测 HTTP 429），OpenAlex 稳定 200
  2. 返回结构化 JSON，摘要 / 概念标签 / 作者 / 年份 一次到位
  3. referenced_works 字段自带引用关系，是后续建图的现成边

三道质量过滤（第一版没做，语料噪声很大）：
  A. 检索用 title_and_abstract.search 而非 search —— 命中量 6.2万 → 4.6千，精度大幅提升
  B. 主题词硬过滤 —— 必须命中 graph 类词，且（text_graph 组除外）命中 LLM 类词
  C. 标题-摘要一致性检查 —— OpenAlex 存在记录错误合并（如 W4385245566 标题讲
     Intelligent Tutoring，摘要却是 Mizar 定理证明，被引数还虚高到 78459），
     用标题关键词在摘要中的命中率 >= 25% 把这类脏数据剔掉

输出：data/raw/works_raw.json
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import norm_title  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，不设会乱码
except AttributeError:
    pass   # Jupyter 的 stdout 是 ipykernel.iostream.OutStream，没有 reconfigure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

API = "https://api.openalex.org/works"

# 填入邮箱可进入 OpenAlex polite pool（速率更宽松）。留空也能正常用。
MAILTO = ""

# 围绕「图 + 大模型」交叉方向分开查，最后按 OpenAlex ID 去重合并
# loose=True 表示该组不强制要求命中 LLM 词（text-attributed graph 本身偏 GNN）
QUERIES = {
    "gnn_llm":      ("graph neural network large language model", False),
    "graphrag":     ("graph retrieval augmented generation", False),
    "kgqa":         ("knowledge graph question answering large language model", False),
    "text_graph":   ("text-attributed graph", True),
    "graph_reason": ("graph reasoning large language model", False),
}

FROM_DATE = "2022-01-01"
PER_PAGE = 200
KEEP_TYPES = {"article", "proceedings-article", "review", "preprint", "book-chapter"}
MIN_ABSTRACT = 300          # 摘要过短的多为占位记录
TITLE_HIT_RATIO = 0.25       # 标题关键词在摘要中的最低命中率

GRAPH_RE = re.compile(
    r"\b(graph|graphs|gnn|gnns|graph-based|graph neural|graph structure|"
    r"graph-structured|knowledge graph|knowledge graphs)\b", re.I)
LLM_RE = re.compile(
    r"\b(llms?|language models?|large language|gpt|chatgpt|llama|bert|prompts?|"
    r"prompting|instruction tuning|foundation models?|transformers?|generative ai|"
    r"retrieval-augmented)\b", re.I)
TITLE_STOP = {
    "the", "and", "for", "with", "from", "into", "over", "via", "using", "based",
    "towards", "toward", "new", "can", "are", "their", "its", "this", "that",
    "case", "study", "approach", "method", "methods", "model", "models",
}

SELECT = ",".join([
    "id", "doi", "title", "display_name", "abstract_inverted_index",
    "publication_year", "publication_date", "authorships", "concepts",
    "referenced_works", "cited_by_count", "primary_location", "type",
])


def restore_abstract(inv):
    """OpenAlex 的摘要存成倒排索引 {词: [位置...]}，按位置还原成文本。"""
    if not inv:
        return ""
    pairs = []
    for word, positions in inv.items():
        for p in positions:
            pairs.append((p, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def title_consistent(title, abstract):
    """标题关键词至少 25% 出现在摘要里，否则视为记录错合并。"""
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", title)]
    words = {w for w in words if w not in TITLE_STOP}
    if not words:
        return True
    low = abstract.lower()
    return sum(1 for w in words if w in low) / len(words) >= TITLE_HIT_RATIO


def fetch(query, retries=4):
    params = {
        "filter": (f"title_and_abstract.search:{query},"
                   f"from_publication_date:{FROM_DATE},is_paratext:false"),
        "per-page": PER_PAGE,
        "sort": "relevance_score:desc",
        "select": SELECT,
    }
    if MAILTO:
        params["mailto"] = MAILTO
    for i in range(retries):
        try:
            r = requests.get(API, params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
            print(f"      HTTP {r.status_code}，第 {i + 1} 次重试")
        except Exception as e:
            print(f"      {type(e).__name__}: {str(e)[:70]}，第 {i + 1} 次重试")
        time.sleep(2 + 2 * i)
    return None


def dedup_and_remap(works):
    """归并同一篇论文的多条 OpenAlex 记录。

    OpenAlex 对同一篇论文常给出两条记录（预印本版 + 正式发表版），标题一致但 ID 不同，
    且旧记录的 referenced_works 往往是空的 —— 引用边全挂在正式版上。
    实测语料 396 篇里有 14 组重复，会让图检索的实体链接指错记录，引用查询直接返回空。

    处理：按归一化标题分组 → 保留被引量最高的一条 → 组内引用列表取并集 →
    被删除记录上的引用边重定向到保留记录（不丢信息）。
    """
    groups = defaultdict(list)
    for w in works:
        groups[norm_title(w["title"])].append(w)

    keep, remap = {}, {}
    merged_refs = {}
    for grp in groups.values():
        grp.sort(key=lambda x: -x["cited_by_count"])
        main = grp[0]
        keep[main["id"]] = main
        for w in grp[1:]:
            remap[w["id"]] = main["id"]
        refs = set()
        for w in grp:
            refs.update(w["referenced_works"])
        merged_refs[main["id"]] = refs

    for wid, w in keep.items():
        refs = {remap.get(r, r) for r in merged_refs[wid]}
        w["referenced_works"] = sorted(refs - {wid})   # 去掉归并后产生的自引用

    return sorted(keep.values(), key=lambda x: -x["cited_by_count"])


def main():
    pool = {}
    dropped = {"type": 0, "no_abstract": 0, "short": 0, "topic": 0, "inconsistent": 0}

    for tag, (q, loose) in QUERIES.items():
        print(f"[{tag}] {q}")
        data = fetch(q)
        if not data:
            print("      x 抓取失败，跳过")
            continue

        added = 0
        for w in data.get("results", []):
            wid = w.get("id")
            if not wid:
                continue
            if wid in pool:
                pool[wid]["tags"].append(tag)
                continue
            if w.get("type") not in KEEP_TYPES:
                dropped["type"] += 1
                continue

            abstract = restore_abstract(w.get("abstract_inverted_index"))
            title = (w.get("title") or w.get("display_name") or "").strip()
            if not abstract:
                dropped["no_abstract"] += 1
                continue
            if len(abstract) < MIN_ABSTRACT:
                dropped["short"] += 1
                continue

            blob = f"{title}\n{abstract}"
            if not GRAPH_RE.search(blob) or (not loose and not LLM_RE.search(blob)):
                dropped["topic"] += 1
                continue
            if not title_consistent(title, abstract):
                dropped["inconsistent"] += 1
                continue

            pool[wid] = {
                "id": wid,
                "doi": w.get("doi"),
                "title": title,
                "abstract": abstract,
                "year": w.get("publication_year"),
                "date": w.get("publication_date"),
                "authors": [
                    a.get("author", {}).get("display_name")
                    for a in (w.get("authorships") or [])
                ][:8],
                "concepts": [
                    c.get("display_name") for c in (w.get("concepts") or [])
                ][:6],
                "referenced_works": w.get("referenced_works") or [],
                "cited_by_count": w.get("cited_by_count", 0),
                "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                "tags": [tag],
            }
            added += 1
        print(f"      +{added} 篇（去重后累计 {len(pool)}）")
        time.sleep(1.2)

    works = sorted(pool.values(), key=lambda x: -x["cited_by_count"])
    before = len(works)
    works = dedup_and_remap(works)

    dest = OUT / "works_raw.json"
    dest.write_text(json.dumps(works, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"过滤丢弃: {dropped}")
    print(f"标题归并: {before} -> {len(works)} 篇（合并同一论文的预印本版与正式版记录）")
    print(f"最终入库 {len(works)} 篇 -> {dest}")

    years = {}
    for w in works:
        years[w["year"]] = years.get(w["year"], 0) + 1
    print("年份分布:", dict(sorted(years.items())))

    inside = {w["id"] for w in works}
    edges = sum(1 for w in works for r in w["referenced_works"] if r in inside)
    print(f"引用边: 其中两端都在语料库内的 {edges} 条")
    print(f"有库内出度的论文 {sum(1 for w in works if any(r in inside for r in w['referenced_works']))} 篇")


if __name__ == "__main__":
    main()
