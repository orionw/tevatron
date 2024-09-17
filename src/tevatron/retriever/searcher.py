import faiss
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        self.dim = init_reps.shape[1]
        # assert self.dim == 4096, "This implementation is optimized for 4096-dimensional vectors"
        
        # Get the number of GPUs
        self.num_gpus = 8 # faiss.get_num_gpus()
        # breakpoint()
        # assert self.num_gpus > 0, "No GPUs available"
        logger.info(f"Number of GPUs available: {self.num_gpus}")
        
        # Create GPU resources
        self.res = [faiss.StandardGpuResources() for _ in range(self.num_gpus)]
        
        # max vectors per GPU (40GB limit)
        self.max_vectors_per_gpu = 1500000 
        logger.info(f"Max vectors per GPU: {self.max_vectors_per_gpu}")

        # Create a list of GPU indices
        self.gpu_indices = []
        for i in range(self.num_gpus):
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = True  
            config.device = i
            index = faiss.GpuIndexFlatIP(self.res[i], self.dim, config)
            self.gpu_indices.append(index)

        self.current_gpu = 0
        self.vectors_per_gpu = [0] * self.num_gpus

    def add(self, p_reps: np.ndarray):
        assert p_reps.shape[1] == self.dim, f"Input vectors must have dimension {self.dim}"
        remaining_vectors = p_reps.shape[0]
        start_idx = 0

        while remaining_vectors > 0:
            # If current GPU index is full, move to next GPU
            if self.vectors_per_gpu[self.current_gpu] + remaining_vectors > self.max_vectors_per_gpu:
                vectors_to_add = self.max_vectors_per_gpu - self.vectors_per_gpu[self.current_gpu]
            else:
                vectors_to_add = remaining_vectors

            # Add vectors to current GPU
            self.gpu_indices[self.current_gpu].add(p_reps[start_idx:start_idx + vectors_to_add])

            # Update counters
            self.vectors_per_gpu[self.current_gpu] += vectors_to_add
            remaining_vectors -= vectors_to_add
            start_idx += vectors_to_add

            # Move to next GPU if current one is full
            if self.vectors_per_gpu[self.current_gpu] == self.max_vectors_per_gpu:
                self.current_gpu = (self.current_gpu + 1) % self.num_gpus

        logger.info(f"Vectors per GPU after addition: {self.vectors_per_gpu}")

    def search(self, q_reps: np.ndarray, k: int):
        assert q_reps.shape[1] == self.dim, f"Query vectors must have dimension {self.dim}"
        assert k > 0, "k must be positive"
        assert sum(self.vectors_per_gpu) > 0, "No vectors have been added to the index"

        all_scores = []
        all_indices = []
        offset = 0

        for i, gpu_index in enumerate(self.gpu_indices):
            if self.vectors_per_gpu[i] > 0:
                scores, indices = gpu_index.search(q_reps, min(k, self.vectors_per_gpu[i]))
                all_scores.append(scores)
                all_indices.append(indices + offset)
            offset += self.vectors_per_gpu[i]

        # Merge results from all GPUs
        merged_scores = np.concatenate(all_scores, axis=1)
        merged_indices = np.concatenate(all_indices, axis=1)

        # Sort to get top-k
        top_k_idx = np.argsort(merged_scores, axis=1)[:, -k:][:, ::-1]
        final_scores = np.take_along_axis(merged_scores, top_k_idx, axis=1)
        final_indices = np.take_along_axis(merged_indices, top_k_idx, axis=1)

        assert final_scores.shape == (q_reps.shape[0], k), f"Unexpected shape of final scores: {final_scores.shape}"
        assert final_indices.shape == (q_reps.shape[0], k), f"Unexpected shape of final indices: {final_indices.shape}"

        return final_scores, final_indices

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        assert q_reps.shape[1] == self.dim, f"Query vectors must have dimension {self.dim}"
        
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            end_idx = min(start_idx + batch_size, num_query)
            batch_q_reps = q_reps[start_idx:end_idx]
            
            batch_scores, batch_indices = self.search(batch_q_reps, k)
            
            all_scores.append(batch_scores)
            all_indices.append(batch_indices)
        
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        assert all_scores.shape == (num_query, k), f"Unexpected shape of all scores: {all_scores.shape}"
        assert all_indices.shape == (num_query, k), f"Unexpected shape of all indices: {all_indices.shape}"

        return all_scores, all_indices

class FaissSearcher(FaissFlatSearcher):
    def __init__(self, init_reps: np.ndarray, factory_str: str):
        super().__init__(init_reps)
        self.factory_str = factory_str
        logger.info(f"FaissSearcher initialized with factory string: {factory_str}")
        # Note: This implementation still uses FaissFlatSearcher
        # The factory_str is stored but not used in the current version