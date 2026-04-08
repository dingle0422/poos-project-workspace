"""向量嵌入与语义聚类工具"""

from typing import Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class Embedder:
    """基于Sentence-Transformers的向量嵌入工具

    支持功能：
    - 文本向量化（归一化）
    - 余弦相似度计算
    - KMeans语义聚类
    - 批量处理
    """

    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers 未安装，请运行: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """将文本列表转为向量矩阵

        Args:
            texts: 文本列表
            normalize: 是否L2归一化（推荐True，余弦相似度用）

        Returns:
            ndarray of shape (n, dim)
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return np.array(embeddings)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度

        Args:
            vec1: 向量1 shape (dim,)
            vec2: 向量2 shape (dim,)

        Returns:
            余弦相似度，范围[-1, 1]
        """
        # 已是归一化向量，点积即余弦相似度
        return float(np.dot(vec1, vec2))

    def batch_similarity(self, query_vec: np.ndarray,
                         doc_vecs: np.ndarray) -> np.ndarray:
        """批量计算query与多个doc的余弦相似度

        Args:
            query_vec: 查询向量 shape (dim,)
            doc_vecs: 文档向量矩阵 shape (n, dim)

        Returns:
            相似度数组 shape (n,)
        """
        return np.dot(doc_vecs, query_vec)

    def cluster(self, vectors: np.ndarray, n_clusters: int = 10,
                min_cluster_size: int = 2) -> np.ndarray:
        """KMeans语义聚类

        Args:
            vectors: 向量矩阵 shape (n, dim)
            n_clusters: 聚类数
            min_cluster_size: 每类最小样本数

        Returns:
            聚类标签数组 shape (n,)
        """
        from sklearn.cluster import KMeans

        # 确保聚类数不超过样本数
        actual_clusters = min(n_clusters, len(vectors) // min_cluster_size)
        if actual_clusters < 2:
            return np.zeros(len(vectors), dtype=int)

        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        return labels

    def find_nearest(self, query_vec: np.ndarray,
                     doc_vecs: np.ndarray,
                     top_k: int = 5) -> list[tuple[int, float]]:
        """查找最相似的Top-K文档

        Args:
            query_vec: 查询向量 shape (dim,)
            doc_vecs: 文档向量矩阵 shape (n, dim)
            top_k: 返回数量

        Returns:
            [(doc_index, similarity), ...] 按相似度降序
        """
        scores = self.batch_similarity(query_vec, doc_vecs)
        sorted_indices = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in sorted_indices[:top_k]]

    def deduplicate_by_similarity(self, texts: list[str],
                                   similarity_threshold: float = 0.95
                                   ) -> list[tuple[str, list[int]]]:
        """基于向量相似度去重

        Args:
            texts: 文本列表
            similarity_threshold: 相似度阈值，超过则视为重复

        Returns:
            [(代表文本, [原始索引列表]), ...]
        """
        if len(texts) <= 1:
            return [(texts[0], [0])] if texts else []

        vectors = self.encode(texts)
        groups = []
        used = set()

        for i in range(len(texts)):
            if i in used:
                continue

            group_indices = [i]
            for j in range(i + 1, len(texts)):
                if j not in used and self.similarity(vectors[i], vectors[j]) >= similarity_threshold:
                    group_indices.append(j)
                    used.add(j)

            groups.append((texts[i], group_indices))
            used.add(i)

        return groups
