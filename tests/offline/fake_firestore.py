"""
오프라인 테스트용 가짜 Firestore (메모리). qupict/db.py가 쓰는 기능만 흉내 낸다.

    문서 읽기 · 쓰기(set / merge / update의 "a.b" 경로 / delete), get_all(필드 고르기 포함),
    where(==, in, >=, <=, array_contains_any), select, stream, batch(묶음 단위로 반영), find_nearest(코사인)

실제 Firestore와 다른 점: 인덱스가 없어도 모든 조건이 된다, 트랜잭션은 바로 반영한다 (db._transact를 바꿔 끼움).
db.py를 이 가짜에 연결: fake_firestore.install(db) → 가짜 클라이언트를 돌려준다.
"""
from __future__ import annotations

import copy
import math

from google.api_core.exceptions import NotFound
from google.cloud import firestore


def _resolve(value, old=None):
    """ArrayUnion 같은 특수 값을 실제 값으로 바꾼다."""
    if isinstance(value, firestore.ArrayUnion):
        out = list(old) if isinstance(old, list) else []
        return out + [v for v in value.values if v not in out]
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    return value


def _merge(target: dict, data: dict) -> None:
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge(target[k], v)
        else:
            target[k] = _resolve(v, target.get(k))


class Snap:
    def __init__(self, ref, data, fields=None):
        self.reference, self.id, self.exists = ref, ref.id, data is not None
        if data is not None and fields is not None:
            data = {k: v for k, v in data.items() if k in set(f.split(".")[0] for f in fields)}
        self._data = copy.deepcopy(data)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def get(self, field):
        return self._data.get(field)


class Ref:
    def __init__(self, client, collection, doc_id):
        self.client, self.collection, self.id = client, collection, doc_id

    @property
    def key(self):
        return (self.collection, self.id)

    def get(self, field_paths=None, transaction=None):
        return Snap(self, self.client.docs.get(self.key), field_paths)

    def set(self, data, merge=False):
        self.client._apply([("set", self, data, merge)])

    def update(self, data):
        self.client._apply([("update", self, data, False)])

    def delete(self):
        self.client._apply([("delete", self, None, False)])


def _cosine_distance(a, b) -> float:
    a, b = list(a), list(b)
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb) if na and nb else 1.0


def _match(doc: dict, field: str, op: str, value) -> bool:
    if field not in doc:
        return False
    v = doc[field]
    if op == "==":
        return v == value
    if op == "in":
        return v in value
    if op == "array_contains_any":
        return isinstance(v, list) and any(x in v for x in value)
    if v is None or value is None or type(v) is not type(value) and not (hasattr(v, "timestamp") and hasattr(value, "timestamp")):
        return False   # 범위 조건은 같은 종류 값끼리만 (None은 빠진다)
    if op == ">=":
        return v >= value
    if op == "<=":
        return v <= value
    raise NotImplementedError(op)


class Query:
    def __init__(self, client, collection, filters=(), fields=None):
        self.client, self.collection, self.filters, self.fields = client, collection, list(filters), fields

    def document(self, doc_id):
        return Ref(self.client, self.collection, doc_id)

    def where(self, filter):
        self.client.queries.append((self.collection, filter.field_path, filter.op_string))
        return Query(self.client, self.collection, [*self.filters, (filter.field_path, filter.op_string, filter.value)],
                     self.fields)

    def select(self, fields):
        return Query(self.client, self.collection, self.filters, list(fields))

    def _docs(self):
        for (col, doc_id), doc in sorted(self.client.docs.items()):
            if col == self.collection and all(_match(doc, f, op, v) for f, op, v in self.filters):
                yield Ref(self.client, col, doc_id), doc

    def stream(self):
        return [Snap(ref, doc, self.fields) for ref, doc in self._docs()]

    def find_nearest(self, vector_field, query_vector, limit, distance_measure, distance_result_field=None):
        query = self

        class Nearest:
            def stream(self_inner):
                scored = sorted(((_cosine_distance(doc[vector_field], query_vector), ref, doc)
                                 for ref, doc in query._docs() if doc.get(vector_field) is not None),
                                key=lambda t: t[0])[:limit]
                out = []
                for dist, ref, doc in scored:
                    doc = dict(doc)
                    if distance_result_field:
                        doc[distance_result_field] = dist
                    out.append(Snap(ref, doc))
                return out
        return Nearest()


class Batch:
    def __init__(self, client):
        self.client, self.ops = client, []

    def set(self, ref, data, merge=False):
        self.ops.append(("set", ref, data, merge))

    def update(self, ref, data):
        self.ops.append(("update", ref, data, False))

    def delete(self, ref):
        self.ops.append(("delete", ref, None, False))

    def commit(self):
        self.client.commits.append(len(self.ops))
        self.client._apply(self.ops)


class Transaction:
    """읽기는 그냥 읽고, 쓰기는 바로 반영한다 (가짜라서 충돌이 없다)."""
    def __init__(self, client):
        self.client = client

    def set(self, ref, data, merge=False):
        self.client._apply([("set", ref, data, merge)])

    def update(self, ref, data):
        self.client._apply([("update", ref, data, False)])


class FakeFirestore:
    def __init__(self):
        self.docs: dict[tuple[str, str], dict] = {}
        self.commits: list[int] = []       # batch.commit마다 쓰기 수
        self.queries: list[tuple] = []     # where에 쓴 (컬렉션, 필드, 연산)

    def collection(self, name):
        return Query(self, name)

    def get_all(self, refs, field_paths=None):
        return [Snap(r, self.docs.get(r.key), field_paths) for r in refs]

    def batch(self):
        return Batch(self)

    def _apply(self, ops):
        for kind, ref, _, _ in ops:          # 묶음 하나는 전부 반영되거나 하나도 안 된다
            if kind == "update" and ref.key not in self.docs:
                raise NotFound(f"없는 문서: {ref.key}")
        for kind, ref, data, merge in ops:
            if kind == "delete":
                self.docs.pop(ref.key, None)
            elif kind == "set" and not merge:
                self.docs[ref.key] = _resolve(copy.deepcopy(data))
            elif kind == "set":
                _merge(self.docs.setdefault(ref.key, {}), copy.deepcopy(data))
            else:                            # update: "images.abc.llm_description" 같은 경로
                doc = self.docs[ref.key]
                for path, value in data.items():
                    *parents, last = path.split(".")
                    target = doc
                    for p in parents:
                        target = target.setdefault(p, {})
                    target[last] = _resolve(value, target.get(last))


def install(db) -> FakeFirestore:
    fake = FakeFirestore()
    db._client = lambda: fake
    db._transact = lambda fn: fn(Transaction(fake))
    return fake
