"""Persistent tissue libraries, K-Means representatives and cosine retrieval."""
from pathlib import Path
import json
import sqlite3
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .messages import UserError

TISSUES = {"her2st": "乳腺", "cscc": "皮肤", "dlpfc": "脑皮层"}


def representatives(features, count):
    features = np.asarray(features, dtype=np.float32)
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    count = min(count, len(features))
    with threadpool_limits(limits=8):
        centers = KMeans(n_clusters=count, random_state=0, n_init=3, max_iter=50).fit(normalized).cluster_centers_
    # Return REAL spots nearest centers, preserving evidence images/expression.
    selected = []
    for center in centers:
        distances = ((normalized - center) ** 2).sum(axis=1)
        selected.append(int(np.argmin(distances)))
    return np.unique(selected)


class PrototypeLibrary:
    def __init__(self, cfg):
        self.cfg = cfg
        self.directory = Path(cfg["paths"]["artifacts"]) / "library"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "prototypes.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS prototypes (
                id INTEGER PRIMARY KEY, tissue TEXT NOT NULL, slide TEXT NOT NULL,
                dataset TEXT NOT NULL, spot INTEGER NOT NULL, source TEXT NOT NULL,
                pixel_x REAL NOT NULL, pixel_y REAL NOT NULL,
                genes TEXT NOT NULL, feature BLOB NOT NULL, expression BLOB NOT NULL,
                created TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tissue, slide, spot, source))""")
            db.execute("CREATE INDEX IF NOT EXISTS tissue_index ON prototypes(tissue)")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @staticmethod
    def encode_records(tissue, slide, dataset, result, expression, source, indices):
        if source not in {"measured", "predicted"}:
            raise UserError("error.expressionSource")
        tissue = str(tissue).strip()
        if not tissue or len(tissue) > 80:
            raise UserError("error.tissueName")
        features = np.asarray(result["features"], dtype=np.float32)
        expression = np.asarray(expression, dtype=np.float32)
        genes = list(result["genes"])
        if features.shape != (len(expression), 2048) or expression.shape[1] != len(genes):
            raise UserError("error.prototypeDimensions")
        if not np.isfinite(features).all() or not np.isfinite(expression).all():
            raise UserError("error.nonfinitePrototype")
        rows = []
        for index in indices:
            index = int(index)
            x, y = result["pixels"][index]
            rows.append((tissue, slide, dataset, index, source, float(x), float(y),
                         json.dumps(genes, ensure_ascii=False), features[index].tobytes(), expression[index].tobytes()))
        return rows

    def add_records(self, tissue, slide, dataset, result, expression, source, indices):
        rows = self.encode_records(tissue, slide, dataset, result, expression, source, indices)
        with self.connect() as db:
            db.executemany("""INSERT INTO prototypes
                (tissue,slide,dataset,spot,source,pixel_x,pixel_y,genes,feature,expression)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tissue,slide,spot,source) DO UPDATE SET
                feature=excluded.feature, expression=excluded.expression, genes=excluded.genes,
                pixel_x=excluded.pixel_x, pixel_y=excluded.pixel_y""", rows)
        return len(rows)

    def replace_training(self, dataset, entries):
        rows = []
        for slide, result, expression, indices in entries:
            rows.extend(self.encode_records(TISSUES[dataset], slide, dataset, result, expression, "measured", indices))
        with self.connect() as db:
            db.execute("DELETE FROM prototypes WHERE dataset=? AND source='measured'", (dataset,))
            db.executemany("""INSERT INTO prototypes
                (tissue,slide,dataset,spot,source,pixel_x,pixel_y,genes,feature,expression)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    def add_slide(self, dataset, slide, result, expression=None, source="predicted", tissue=None):
        if expression is None:
            expression = result["prediction"]
        selected = representatives(result["features"], self.cfg["prototype"]["clusters_per_slide"])
        tissue = (tissue or TISSUES[dataset]).strip()
        rows = self.encode_records(tissue, slide, dataset, result, expression, source, selected)
        with self.connect() as db:
            db.execute("DELETE FROM prototypes WHERE tissue=? AND slide=? AND source=?", (tissue, slide, source))
            db.executemany("""INSERT INTO prototypes
                (tissue,slide,dataset,spot,source,pixel_x,pixel_y,genes,feature,expression)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    def catalog(self):
        with self.connect() as db:
            rows = db.execute("""SELECT tissue, COUNT(*) AS count, COUNT(DISTINCT slide) AS slides,
                SUM(source='measured') AS measured, SUM(source='predicted') AS predicted
                FROM prototypes GROUP BY tissue ORDER BY tissue""").fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def metadata(row):
        values = np.frombuffer(row["expression"], dtype=np.float32)
        genes = json.loads(row["genes"])
        order = np.argsort(values)[::-1][:8]
        return {key: row[key] for key in ("id", "tissue", "slide", "dataset", "spot", "source", "pixel_x", "pixel_y", "created")} | {
            "top_genes": [{"gene": genes[i], "value": float(values[i])} for i in order]}

    def browse(self, tissue=None, offset=0, limit=24):
        with self.connect() as db:
            if tissue:
                rows = db.execute("SELECT * FROM prototypes WHERE tissue=? ORDER BY id DESC LIMIT ? OFFSET ?",
                                  (tissue, limit, offset)).fetchall()
            else:
                rows = db.execute("SELECT * FROM prototypes ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [self.metadata(row) for row in rows]

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM prototypes WHERE id=?", (int(identifier),)).fetchone()
        if row is None:
            raise UserError("error.prototypeMissing", status=404)
        return dict(row)

    def retrieve(self, feature, genes, tissue):
        query = np.asarray(feature, dtype=np.float32).reshape(2048)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm <= 1e-12:
            raise UserError("error.invalidFeatures")
        panel = json.dumps(list(genes), ensure_ascii=False)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM prototypes WHERE tissue=? AND genes=? ORDER BY id", (tissue, panel)).fetchall()
        if not rows:
            return []
        bank = np.stack([np.frombuffer(row["feature"], dtype=np.float32) for row in rows])
        similarity = bank @ (query / norm) / np.maximum(np.linalg.norm(bank, axis=1), 1e-12)
        order = np.argsort(-similarity, kind="stable")[:self.cfg["prototype"]["retrieval_k"]]
        return [{**self.metadata(rows[i]), "similarity": float(similarity[i])} for i in order]
