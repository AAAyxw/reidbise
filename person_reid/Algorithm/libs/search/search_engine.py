from collections import defaultdict

import numpy as np
import faiss
import Algorithm.libs.config.model_cfgs as cfgs
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)


def normalize_modality(modality):
    if not modality:
        return "vis"
    key = str(modality).lower().strip()
    if key in ("visible", "vis", "rgb", "color"):
        return "vis"
    if key in ("vis+ir", "visible+ir", "ir+vis", "dual", "both"):
        return "vis_ir"
    if key in ("ir", "infrared"):
        return "ir"
    return key


class SearchEngine(object):
    def __init__(self, base_feat_lists, base_idx_lists, dims=1024, modalities=None):
        self._register_labels = base_idx_lists if base_idx_lists is not None else []
        self._modalities = [normalize_modality(m) for m in (modalities or [])]
        self._index = None
        self._dims = dims

        self._label_centroids = {}
        self._label_radii = {}
        if len(base_idx_lists) == 0:
            log_info.info("No feat register.Total num is 0")
            return

        base_feat_lists = np.array(base_feat_lists, dtype=np.float32)
        gallery_dim = int(base_feat_lists.shape[1])
        index_dim = int(dims) if dims is not None else gallery_dim

        if gallery_dim != index_dim:
            log_info.warning(
                "Gallery feature dim %s != configured index dim %s; using gallery dim.",
                gallery_dim,
                index_dim,
            )
            index_dim = gallery_dim

        self._dims = index_dim
        self._index = faiss.IndexFlatL2(index_dim)
        self._index.add(base_feat_lists)
        self._label_centroids, self._label_radii = self._build_label_prototypes(
            base_feat_lists, base_idx_lists,
        )
        log_info.info(
            "Faiss search engine load succeed!!! The dims is {}. Total num is {}".format(
                index_dim, len(base_idx_lists)
            )
        )
        for label, radius in self._label_radii.items():
            log_info.info("Gallery radius for '%s': %.4f", label, radius)

    @staticmethod
    def _normalize_feat(feat):
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def _build_label_prototypes(self, base_feat_lists, base_idx_lists):
        groups = defaultdict(list)
        for feat, label in zip(base_feat_lists, base_idx_lists):
            groups[label].append(self._normalize_feat(feat))

        centroids = {}
        radii = {}
        for label, feats in groups.items():
            arr = np.stack(feats, axis=0)
            centroid = arr.mean(axis=0)
            centroid = self._normalize_feat(centroid)
            if len(feats) == 1:
                radius = cfgs.GALLERY_SINGLE_RADIUS
            else:
                dists = [float(np.linalg.norm(f - centroid)) for f in feats]
                radius = max(dists) * cfgs.GALLERY_RADIUS_SCALE
            centroids[label] = centroid
            radii[label] = radius
        return centroids, radii

    def search(self, query_feat, top_k=10):
        if self._index is None or len(self._register_labels) == 0:
            return [], []
        query = np.asarray(query_feat, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dims:
            log_info.error(
                "Query feature dim %s != index dim %s; skip search.",
                query.shape[1],
                self._dims,
            )
            return [], []
        dist_list, idx_list = self._index.search(query, top_k)
        label_idx = [self._register_labels[sort_e_idx] for sort_e_idx in idx_list[0]]
        return label_idx, dist_list[0].tolist()

    def rerank(self, query_feat, labels, dists, query_modality="vis"):
        if len(labels) == 0 or len(dists) == 0:
            return labels, dists
        query_modality = normalize_modality(query_modality)
        reranked = []
        for i, (label, dist) in enumerate(zip(labels, dists)):
            gallery_modality = self._modalities[i] if i < len(self._modalities) else "vis"
            if query_modality == gallery_modality:
                adjusted_dist = dist * 0.97
            elif {query_modality, gallery_modality} <= {"vis", "vis_ir"}:
                adjusted_dist = dist * 1.03
            else:
                adjusted_dist = dist * 1.08
            reranked.append((label, adjusted_dist, i))
        reranked.sort(key=lambda x: x[1])
        return [item[0] for item in reranked], [item[1] for item in reranked]

    @staticmethod
    def l2_dist_to_cosine_sim(dist):
        return 1.0 - (float(dist) * float(dist)) / 2.0

    def decide_match(
        self,
        labels,
        dists,
        dist_thresh,
        query_feat=None,
        min_margin=None,
        min_cosine_sim=None,
    ):
        """
        Accept a match only when distance, cosine similarity, and (if needed) ID margin pass.
        """
        if not labels or not dists:
            return False, None, None

        min_margin = cfgs.MATCH_MIN_MARGIN if min_margin is None else min_margin
        min_cosine_sim = cfgs.MATCH_MIN_COSINE_SIM if min_cosine_sim is None else min_cosine_sim

        per_label = {}
        for label, dist in zip(labels, dists):
            dist = float(dist)
            if label not in per_label or dist < per_label[label]:
                per_label[label] = dist

        ranked = sorted(per_label.items(), key=lambda item: item[1])
        best_label, best_dist = ranked[0]
        cosine_sim = self.l2_dist_to_cosine_sim(best_dist)

        if best_dist > float(dist_thresh) or cosine_sim < float(min_cosine_sim):
            return False, None, best_dist

        if len(ranked) >= 2:
            second_label, second_dist = ranked[1]
            margin = second_dist - best_dist
            if margin < float(min_margin):
                log_info.debug(
                    "Reject ambiguous match %s(%.4f) vs %s(%.4f), margin=%.4f",
                    best_label, best_dist, second_label, second_dist, margin,
                )
                return False, None, best_dist

        if query_feat is not None and best_label in getattr(self, '_label_centroids', {}):
            query_vec = self._normalize_feat(query_feat)
            centroid_dist = float(np.linalg.norm(query_vec - self._label_centroids[best_label]))
            if centroid_dist > self._label_radii[best_label]:
                log_info.debug(
                    "Reject %s: centroid dist %.4f > gallery radius %.4f",
                    best_label, centroid_dist, self._label_radii[best_label],
                )
                return False, None, best_dist

        return True, best_label, best_dist
