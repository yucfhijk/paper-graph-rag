# -*- coding: utf-8 -*-
"""
evaluate.py — 在同一评估集上横向对比各检索器

设计要点：不是「给每种题型配它擅长的检索器」，而是让**每个检索器跑全部三类题**。
只有这样得到的矩阵才能回答那个关键问题：图结构到底解决了什么、又解决不了什么。

指标：
  HR@1 / HR@5   top-k 内是否命中任一 gold
  MRR@10        首个命中位置的倒数排名均值
  Recall@10     top-10 覆盖的 gold 比例（引用型题目 gold 是集合，这个指标才有意义）

输出：results/metrics.json、results/run_<retriever>.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Corpus, build  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
K = 10
RETRIEVERS = ["tfidf", "graph"]
TYPES = ["keyword", "semantic", "citation"]


def hr(ranked, gold, k):
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def rr(ranked, gold, k=10):
    g = set(gold)
    for i, d in enumerate(ranked[:k], 1):
        if d in g:
            return 1.0 / i
    return 0.0


def recall(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else 0.0


def main():
    corpus = Corpus()
    items = json.loads((ROOT / "data" / "eval_set.json").read_text(encoding="utf-8"))
    print(f"语料 {len(corpus)} 篇 | 评估集 {len(items)} 题\n")

    (ROOT / "results").mkdir(exist_ok=True)
    report = {"corpus_size": len(corpus), "eval_size": len(items), "k": K, "results": {}}

    for name in RETRIEVERS:
        r = build(name, corpus)
        buckets = defaultdict(list)
        runs = {}

        for it in items:
            ranked = r.search(it["query"], K)
            runs[it["id"]] = {
                "query": it["query"],
                "gold": it["gold"],
                "gold_titles": [corpus.title.get(g, "?") for g in it["gold"]][:5],
                "top10": ranked,
                "top10_titles": [corpus.title.get(d, "?") for d in ranked],
            }
            buckets[it["type"]].append({
                "hr1": hr(ranked, it["gold"], 1),
                "hr5": hr(ranked, it["gold"], 5),
                "hr10": hr(ranked, it["gold"], 10),
                "mrr": rr(ranked, it["gold"], 10),
                "recall10": recall(ranked, it["gold"], 10),
            })

        report["results"][name] = {}
        for t in TYPES:
            rows = buckets.get(t, [])
            if not rows:
                continue
            agg = {m: sum(x[m] for x in rows) / len(rows) for m in rows[0]}
            agg["n"] = len(rows)
            report["results"][name][t] = agg

        allrows = [x for t in TYPES for x in buckets.get(t, [])]
        report["results"][name]["ALL"] = {
            **{m: sum(x[m] for x in allrows) / len(allrows) for m in allrows[0]},
            "n": len(allrows),
        }

        (ROOT / "results" / f"run_{name}.json").write_text(
            json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")

    (ROOT / "results" / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'检索器':<10}{'题型':<11}{'题数':>5}{'HR@1':>8}{'HR@5':>8}{'MRR@10':>9}{'Recall@10':>11}")
    print("-" * 62)
    for name in RETRIEVERS:
        for t in TYPES + ["ALL"]:
            a = report["results"][name].get(t)
            if not a:
                continue
            print(f"{name:<10}{t:<11}{a['n']:>5}{a['hr1']:>8.3f}{a['hr5']:>8.3f}"
                  f"{a['mrr']:>9.3f}{a['recall10']:>11.3f}")
        print("-" * 62)

    print(f"\n明细已写入 results/metrics.json 与 results/run_*.json")


if __name__ == "__main__":
    main()
