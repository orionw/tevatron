import pickle
import numpy as np
import glob
from argparse import ArgumentParser
import faiss
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)

class GPUFaissSearcher:
    def __init__(self):
        self.res = faiss.StandardGpuResources()
        self.index = None
        logger.info("Using GPU for FAISS")

    def build_index(self, vectors):
        vectors = vectors.astype(np.float32)  # Ensure float32
        dimension = vectors.shape[1]
        logger.info(f"Building index with dimension {dimension}")
        cpu_index = faiss.IndexFlatIP(dimension)
        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = False  # Ensure we use float32
        self.index = faiss.GpuIndexFlatIP(self.res, dimension, config)
        self.add(vectors)

    def add(self, vectors):
        vectors = vectors.astype(np.float32)  # Ensure float32
        if self.index is None:
            self.build_index(vectors)
        else:
            self.index.add(vectors.astype(np.float32))
        logger.info(f"Index now contains {self.index.ntotal} vectors")

    def search(self, queries, k):
        logger.info(f"Searching {queries.shape[0]} queries for top {k} results")
        return self.index.search(queries.astype(np.float32), k)

def test_faiss_index(index, test_vectors):
    logger.info("Testing FAISS index with sample vectors")
    n_test = min(100, test_vectors.shape[0])
    test_queries = test_vectors[:n_test].astype(np.float32)
    
    # Self-search test
    scores, indices = index.search(test_queries, 1)
    exact_match_ratio = np.mean(indices.flatten() == np.arange(n_test))
    logger.info(f"Exact match ratio in self-search: {exact_match_ratio:.2%}")
    logger.info(f"Self-search score stats - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")

    # Random vector search test
    random_vectors = np.random.randn(n_test, test_vectors.shape[1]).astype(np.float32)
    random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
    try:
        scores, _ = index.search(random_vectors, 1)
        logger.info(f"Random vector search score stats - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")
    except Exception as e:
        logger.error(f"Error during random vector search: {str(e)}")

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    logger.info(f"Loaded {len(reps)} vectors from {path}")
    return np.array(reps), lookup

def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def search_queries(retriever, q_reps, p_lookup, depth):
    all_scores, all_indices = retriever.search(q_reps, depth)
    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    return all_scores, np.array(psg_indices)

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = sorted(zip(q_doc_scores, q_doc_indices), key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')

def check_vector_properties(vectors, name):
    logger.info(f"Checking {name} properties:")
    logger.info(f"Shape: {vectors.shape}")
    logger.info(f"Min: {vectors.min():.6f}, Max: {vectors.max():.6f}")
    logger.info(f"Mean: {vectors.mean():.6f}, Std: {vectors.std():.6f}")
    norms = np.linalg.norm(vectors, axis=1)
    logger.info(f"Norm min: {norms.min():.6f}, max: {norms.max():.6f}, mean: {norms.mean():.6f}")
    
    # Check for NaN or Inf values
    if np.isnan(vectors).any() or np.isinf(vectors).any():
        logger.warning(f"NaN or Inf values detected in {name}")
    
    # Check for zero vectors
    zero_vectors = np.all(vectors == 0, axis=1)
    if zero_vectors.any():
        logger.warning(f"{np.sum(zero_vectors)} zero vectors detected in {name}")

def test_faiss_index(index, test_vectors):
    logger.info("Testing FAISS index with sample vectors")
    n_test = min(100, test_vectors.shape[0])
    test_queries = test_vectors[:n_test]
    
    # Self-search test
    scores, indices = index.search(test_queries, 1)
    exact_match_ratio = np.mean(indices.flatten() == np.arange(n_test))
    logger.info(f"Exact match ratio in self-search: {exact_match_ratio:.2%}")
    logger.info(f"Self-search score stats - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")

    # Random vector search test
    random_vectors = np.random.randn(n_test, test_vectors.shape[1])
    random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
    random_vectors = random_vectors.astype(np.float32)
    scores, _ = index.search(random_vectors, 1)
    logger.info(f"Random vector search score stats - min: {scores.min():.6f}, max: {scores.max():.6f}, mean: {scores.mean():.6f}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=True)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_to', required=True)
    parser.add_argument('--save_text', action='store_true')
    args = parser.parse_args()

    retriever = GPUFaissSearcher()
    look_up = []
    print(faiss.get_num_gpus())

    all_p_reps = []
    for file in glob.glob(args.passage_reps):
        p_reps, p_lookup = pickle_load(file)
        p_reps = p_reps.astype(np.float32)
        check_vector_properties(p_reps, f"Passage vectors from {file}")
        all_p_reps.append(p_reps)
        look_up.extend(p_lookup)

    all_p_reps = np.vstack(all_p_reps)
    retriever.add(all_p_reps)
    
    logger.info(f"Passage vectors dtype: {all_p_reps.dtype}")



    q_reps, q_lookup = pickle_load(args.query_reps)
    q_reps = q_reps.astype(np.float32)
    check_vector_properties(q_reps, "Query vectors")
    logger.info(f"Query vectors dtype: {q_reps.dtype}")

    test_faiss_index(retriever.index, all_p_reps)

    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args.depth)

    logger.info(f"Score stats - min: {all_scores.min():.6f}, max: {all_scores.max():.6f}, mean: {all_scores.mean():.6f}")

    # Additional diagnostics
    zero_score_queries = np.all(all_scores == 0, axis=1)
    if zero_score_queries.any():
        logger.warning(f"{np.sum(zero_score_queries)} queries returned all zero scores")
        
    if args.save_text:
        write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_to)
    else:
        pickle_save((all_scores, psg_indices), args.save_ranking_to)

    logger.info(f"Results saved to {args.save_ranking_to}")

if __name__ == '__main__':
    main()