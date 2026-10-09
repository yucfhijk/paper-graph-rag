# -*- coding: utf-8 -*-
"""
retriever.py — 可插拔检索器

三者共用 search(query, k) -> [doc_id, ...] 接口，便于在同一套评估集上横向对比：
  TfidfRetriever  词法检索（TF-IDF 余弦）。关键词明确的查询上是强基线。
  GraphRetriever  图结构检索。从查询里解析出论文标题做实体链接，
                  再沿引用图的入边展开，回答「谁引用了这篇」这类关系型问题。
                  内容型查询（关键词/语义）它答不了，返回空 —— 这正是要对照的点。

VectorRetriever 语义检索待接入（需要 sentence-transformers），
留出接口是为了保持「同一评估集、同一指标」的可比性。
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import norm_title  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，不设会乱码
except AttributeError:
    pass   # Jupyter 的 stdout 是 ipykernel.iostream.OutStream，没有 reconfigure

ROOT = Path(__file__).resolve().parent.parent


class Corpus:
    def __init__(self, path=None):
        path = path or (ROOT / "data" / "raw" / "works_raw.json")
        import json
        self.works = json.loads(Path(path).read_text(encoding="utf-8"))
        self.ids = [w["id"] for w in self.works]
        self.title = {w["id"]: w["title"] for w in self.works}
        self.cited = {w["id"]: w["cited_by_count"] for w in self.works}
        self.inside = set(self.ids)

    def text(self, w):
        """检索用文档文本：标题权重靠前置重复一次来体现。"""
        return f"{w['title']} . {w['title']} . {w['abstract']}"

    def __len__(self):
        return len(self.works)


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, corpus):
        self.corpus = corpus
        self.ids = corpus.ids
        self.vec = TfidfVectorizer(
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=300000,
        )
        self.mat = self.vec.fit_transform(corpus.text(w) for w in corpus.works)

    def search(self, query, k=10):
        q = self.vec.transform([query])
        scores = (self.mat @ q.T).toarray().ravel()
        top = scores.argsort()[::-1][:k]
        return [self.ids[i] for i in top if scores[i] > 0]


class GraphRetriever:
    name = "graph"

    def __init__(self, corpus):
        self.corpus = corpus
        self.by_title = defaultdict(list)
        for w in corpus.works:
            self.by_title[norm_title(w["title"])].append(w["id"])
        self.in_edges = {}
        for w in corpus.works:
            for r in w["referenced_works"]:
                if r in corpus.inside:
                    self.in_edges.setdefault(r, []).append(w["id"])
        # 入边按被引量降序；被引量相同时用 doc_id 兜底做 tie-breaker ——
        # in_edges 是 list（顺序由 corpus.works 决定），但排序键必须全序，
        # 否则同分项的相对顺序不稳定，逐题明细 run_*.json 会跨进程漂移。
        for k in self.in_edges:
            self.in_edges[k].sort(key=lambda i: (-corpus.cited.get(i, 0), i))

    def search(self, query, k=10):
        m = re.search(r'"(.+?)"', query)
        if not m:
            return []
        ids = self.by_title.get(norm_title(m.group(1)))
        if not ids:
            return []
        hit = {d for wid in ids for d in self.in_edges.get(wid, [])}
        # hit 是 set，迭代顺序受字符串 hash 随机化影响（每个进程都不同）。
        # 所以排序键必须是全序：被引量相同时用 doc_id 兜底，
        # 否则同分论文的相对顺序会漂移，run_*.json 每次跑都不一样。
        return sorted(hit, key=lambda i: (-self.corpus.cited.get(i, 0), i))[:k]

    def neighbors(self, seed_ids, k=10):
        """把种子论文的引用邻居并入候选（供后续混合检索使用）。"""
        out = []
        for sid in seed_ids:
            for n in self.in_edges.get(sid, []):
                if n not in out:
                    out.append(n)
        return out[:k]


def build(name, corpus):
    table = {"tfidf": TfidfRetriever, "graph": GraphRetriever}
    if name not in table:
        raise ValueError(f"未知检索器: {name}（可选 {list(table)}）")
    return table[name](corpus)
