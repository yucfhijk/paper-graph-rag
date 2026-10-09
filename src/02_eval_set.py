# -*- coding: utf-8 -*-
"""
02_eval_set.py — 构造三类检索评估集

T1 keyword  : 关键词型。query = 标题去停用词后的关键词，gold = 该论文。
              关键词明确时词法检索理应表现良好，作为对照基线。
T2 semantic : 语义型。query = 人工改写的自然语言描述（刻意避开标题用词），
              gold = 该论文。词法检索在此因词汇不匹配而掉分，是向量检索的用武之地。
T3 citation : 引用型。query = "哪些论文引用了 X"，gold = 库内引用 X 的论文集合。
              词法与向量检索都无法直接回答这类问题，只有图结构能解
              —— 这是本项目建图的直接理由，也是评估指标上最干净的对照组。

输出：data/eval_set.json
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，不设会乱码
except AttributeError:
    pass   # Jupyter 的 stdout 是 ipykernel.iostream.OutStream，没有 reconfigure

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "works_raw.json"
SEM = ROOT / "data" / "semantic_queries.json"
DEST = ROOT / "data" / "eval_set.json"

SEED = 42
N_KEYWORD = 30
N_CITATION = 25
MIN_CITED_IN_LIB = 2

STOP = {
    "the", "and", "for", "with", "from", "into", "over", "via", "using", "based",
    "towards", "toward", "new", "can", "are", "its", "this", "that", "not", "but",
    "all", "any", "how", "what", "when", "who", "why", "you", "your", "our", "they",
    "their", "has", "have", "had", "was", "were", "will", "would", "could", "should",
    "more", "most", "some", "such", "than", "then", "there", "these", "those",
    "which", "while", "about", "also", "been", "being", "between", "both", "each",
    "other", "only", "out", "own", "same", "very", "just", "now", "one", "two",
    "case", "study", "tool", "toward",
}


def norm_id(x):
    """语义题库里手写的 gold 是 OpenAlex 短码（W123），库里存的是完整 URL，统一成后者。"""
    x = x.strip().rstrip("/")
    return x if x.startswith("http") else f"https://openalex.org/{x}"


def keyword_query(title):
    """从标题提炼关键词：去停用词、去重保序、取前 6 个。"""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", title)
    seen, out = set(), []
    for w in words:
        k = w.lower()
        if k in STOP or k in seen:
            continue
        seen.add(k)
        out.append(w)
    return " ".join(out[:6])


def main():
    works = json.loads(RAW.read_text(encoding="utf-8"))
    by_id = {w["id"]: w for w in works}
    inside = set(by_id)
    rng = random.Random(SEED)

    items = []

    # ---------- T1 关键词型 ----------
    pool = [w for w in works if len(w["abstract"]) >= 400 and len(w["title"]) >= 30]
    for w in rng.sample(pool, min(N_KEYWORD, len(pool))):
        items.append({
            "id": f"K{len(items) + 1:02d}",
            "type": "keyword",
            "query": keyword_query(w["title"]),
            "gold": [w["id"]],
        })

    # ---------- T2 语义型 ----------
    sem = json.loads(SEM.read_text(encoding="utf-8"))
    skipped = []
    for s in sem:
        gid = norm_id(s["gold"])
        if gid not in inside:
            skipped.append(s["id"])
            continue
        items.append({
            "id": s["id"],
            "type": "semantic",
            "query": s["query"],
            "gold": [gid],
        })

    # ---------- T3 引用型 ----------
    indeg = {}
    for w in works:
        for r in w["referenced_works"]:
            if r in inside:
                indeg.setdefault(r, set()).add(w["id"])
    cites = sorted(k for k, v in indeg.items() if len(v) >= MIN_CITED_IN_LIB)
    rng.shuffle(cites)
    for cid in cites[:N_CITATION]:
        items.append({
            "id": f"C{len([i for i in items if i['type'] == 'citation']) + 1:02d}",
            "type": "citation",
            "query": f'Which papers cite the work titled "{by_id[cid]["title"]}"?',
            "gold": sorted(indeg[cid]),
        })

    DEST.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    c = Counter(i["type"] for i in items)
    print(f"评估集共 {len(items)} 条 -> {DEST}")
    print("  类型分布:", dict(c))
    if skipped:
        print("  语义题目 gold 不在库内，已跳过:", skipped)

    for t in ("keyword", "semantic", "citation"):
        gs = [len(i["gold"]) for i in items if i["type"] == t]
        if gs:
            print(f"  {t:9s} 每题 gold 数: 最小 {min(gs)} / 最大 {max(gs)} / 平均 {sum(gs) / len(gs):.1f}")

    print()
    print("样例:")
    for i in items[:2] + items[N_KEYWORD:N_KEYWORD + 2] + items[-2:]:
        print(f"  [{i['id']}/{i['type']}] {i['query'][:88]}")
        print(f"        gold={len(i['gold'])} 篇")


if __name__ == "__main__":
    main()
