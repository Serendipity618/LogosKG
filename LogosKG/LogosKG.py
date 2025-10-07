import numpy as np
import scipy.sparse as sp
from numba import njit
from typing import List, Dict, Tuple
import torch
import warnings
import gc

warnings.filterwarnings("ignore")


class LogosKG:
    """
    Efficient knowledge graph multi-hop retrieval using three-matrix decomposition.

    """

    def __init__(self, triplets: List[Tuple[str, str, str]], backend: str = "numba",
                 device: str = "cpu", flush_memory: bool = True):
        """
        Initialize knowledge graph from triplets.

        Args:
            triplets: List of (subject, relation, object) tuples
            backend: 'scipy', 'numba', or 'torch'
            device: 'cpu' or 'cuda' (for torch)
            flush_memory: Clear triplets after building matrices
        """
        self.backend = backend
        self.device = device
        self.flush_memory = flush_memory

        # Build mappings and matrices
        entities = set()
        relations = set()
        for sub, rel, obj in triplets:
            entities.add(sub)
            entities.add(obj)
            relations.add(rel)

        self.cui_to_idx = {e: i for i, e in enumerate(entities)}
        self.idx_to_cui = {i: e for e, i in self.cui_to_idx.items()}
        self.rel_to_idx = {r: i for i, r in enumerate(relations)}
        self.idx_to_rel = {i: r for r, i in self.rel_to_idx.items()}

        self.num_entities = len(entities)
        self.num_relations = len(relations)
        self.num_triplets = len(triplets)

        idx_triplets = [(self.cui_to_idx[s], self.rel_to_idx[r], self.cui_to_idx[o])
                        for s, r, o in triplets]

        self._build_backend_matrices(idx_triplets)

        if flush_memory:
            del triplets, idx_triplets
            gc.collect()

    # ========================================
    # Public API
    # ========================================

    def retrieve_at_k_hop(self, entity_ids: List[str], hops: int, shortest_path: bool = False) -> List[str]:
        """
        Return entities at exactly k hops from any starting entity.

        Args:
            entity_ids: Starting entities
            hops: Exact distance
            shortest_path: If True, use shortest-path semantics (match baselines).
                          If False, allow any k-hop path (reachability semantics).

        Returns:
            Entities at exactly k hops
        """
        if self.backend == "torch" and len(entity_ids) > 1:
            return self._torch_batched_k_hop(entity_ids, hops, shortest_path)

        total = np.zeros(self.num_entities, dtype=np.uint8)

        for seed in entity_ids:
            if seed not in self.cui_to_idx:
                continue

            current = np.zeros(self.num_entities, dtype=np.uint8)
            current[self.cui_to_idx[seed]] = 1

            if shortest_path:
                visited = current.copy()
                for _ in range(hops):
                    current = self._execute_hop(current)
                    current = np.where(visited == 1, 0, current).astype(np.uint8)
                    visited = np.maximum(visited, current)
                    if current.sum() == 0:
                        break
            else:
                for _ in range(hops):
                    current = self._execute_hop(current)
                    if current.sum() == 0:
                        break

            total = np.maximum(total, current)

        return self._indices_to_entities(total, exclude=set(entity_ids))

    def retrieve_within_k_hop(self, entity_ids: List[str], hops: int) -> List[str]:
        """
        Return entities within k hops from any starting entity.

        Args:
            entity_ids: Starting entities
            hops: Maximum distance

        Returns:
            Entities within k hops
        """
        current = self._entities_to_vector(entity_ids)
        visited = current.copy()
        all_entities = current.copy()

        for _ in range(hops):
            current = self._execute_hop(current)
            current = np.where(visited == 1, 0, current).astype(np.uint8)
            visited = np.maximum(visited, current)
            all_entities = np.maximum(all_entities, current)
            if current.sum() == 0:
                break

        return self._indices_to_entities(all_entities, exclude=set(entity_ids))

    def retrieve_with_paths_at_k_hop(self, entity_ids: List[str], hops: int = 2) -> Dict:
        """
        Return entities at exactly k hops with ALL possible k-hop paths.

        Args:
            entity_ids: Starting entities
            hops: Exact path length

        Returns:
            {'entities': [...], 'paths': {entity: [[path1], [path2], ...]}}
        """
        all_paths = {}

        for seed in entity_ids:
            if seed not in self.cui_to_idx:
                continue

            # Track paths at current hop level
            current_hop_paths = {seed: [[seed]]}

            for hop in range(hops):
                next_hop_paths = {}

                # Get active entities and triplets
                current_entities = list(current_hop_paths.keys())
                current = self._entities_to_vector(current_entities)
                active_triplets, _ = self._execute_hop_with_triplets(current)

                # Extend all paths by one hop
                for ent_idx in np.where(current > 0)[0]:
                    ent_name = self.idx_to_cui[ent_idx]
                    if ent_name not in current_hop_paths:
                        continue

                    for trip_idx in self._get_entity_triplets(ent_idx):
                        if active_triplets[trip_idx] > 0:
                            rel_idx, obj_idx = self._get_triplet_info(trip_idx)
                            if rel_idx is None or obj_idx is None:
                                continue

                            rel_name = self.idx_to_rel[rel_idx]
                            obj_name = self.idx_to_cui[obj_idx]

                            for path in current_hop_paths[ent_name]:
                                new_path = path + [f"--{rel_name}-->", obj_name]
                                if obj_name not in next_hop_paths:
                                    next_hop_paths[obj_name] = []
                                next_hop_paths[obj_name].append(new_path)

                current_hop_paths = next_hop_paths

            # Collect paths from this seed
            for entity, paths in current_hop_paths.items():
                if entity not in all_paths:
                    all_paths[entity] = []
                all_paths[entity].extend(paths)

        result_entities = list(all_paths.keys())
        return {"entities": result_entities, "paths": all_paths}

    def retrieve_with_paths_within_k_hop(self, entity_ids: List[str], hops: int = 2) -> Dict:
        """
        Return entities within k hops with ALL possible paths (up to k hops).

        Args:
            entity_ids: Starting entities
            hops: Maximum path length

        Returns:
            {'entities': [...], 'paths': {entity: [[path1], [path2], ...]}}
        """
        all_paths = {e: [[e]] for e in entity_ids if e in self.cui_to_idx}

        for hop in range(hops):
            # Get all entities with paths at current hop length
            current_hop_entities = [e for e, paths in all_paths.items()
                                    if any(len(p) == hop * 2 + 1 for p in paths)]
            if not current_hop_entities:
                break

            # Get active triplets
            current = self._entities_to_vector(current_hop_entities)
            active_triplets, _ = self._execute_hop_with_triplets(current)

            # Extend paths by one hop
            for ent_idx in np.where(current > 0)[0]:
                ent_name = self.idx_to_cui[ent_idx]
                if ent_name not in all_paths:
                    continue

                for trip_idx in self._get_entity_triplets(ent_idx):
                    if active_triplets[trip_idx] > 0:
                        rel_idx, obj_idx = self._get_triplet_info(trip_idx)
                        if rel_idx is None or obj_idx is None:
                            continue

                        rel_name = self.idx_to_rel[rel_idx]
                        obj_name = self.idx_to_cui[obj_idx]

                        # Extend paths of correct length from this entity
                        for path in all_paths[ent_name]:
                            if len(path) == hop * 2 + 1:
                                new_path = path + [f"--{rel_name}-->", obj_name]
                                if obj_name not in all_paths:
                                    all_paths[obj_name] = []
                                all_paths[obj_name].append(new_path)

        # Remove starting entities
        result_entities = [e for e in all_paths.keys() if e not in entity_ids]
        result_paths = {e: all_paths[e] for e in result_entities}

        return {"entities": result_entities, "paths": result_paths}

    def set_backend(self, backend: str):
        """Switch backend (requires original triplets)."""
        self.backend = backend
        # Note: This won't work if flush_memory=True was used
        # Would need to store idx_triplets to rebuild

    def get_info(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_triplets": self.num_triplets,
            "backend": self.backend,
            "device": self.device,
        }

    # ========================================
    # Helper Methods
    # ========================================

    def _entities_to_vector(self, entity_ids: List[str]) -> np.ndarray:
        """Convert entity IDs to binary vector."""
        vec = np.zeros(self.num_entities, dtype=np.uint8)
        for eid in entity_ids:
            if eid in self.cui_to_idx:
                vec[self.cui_to_idx[eid]] = 1
        return vec

    def _indices_to_entities(self, vector: np.ndarray, exclude: set = None) -> List[str]:
        """Convert binary vector to entity IDs."""
        indices = np.where(vector > 0)[0]
        entities = [self.idx_to_cui[i] for i in indices]
        if exclude:
            entities = [e for e in entities if e not in exclude]
        return entities

    # ========================================
    # Core Execution (Backend Dispatch)
    # ========================================

    def _execute_hop(self, current: np.ndarray) -> np.ndarray:
        """Execute single hop: entities → triplets → next entities."""
        if self.backend == "scipy":
            curr = sp.csr_matrix(current.reshape(1, -1), dtype=np.bool_)
            result = curr @ self.sub_matrix @ self.obj_matrix
            return (result.toarray().ravel() > 0).astype(np.uint8)

        elif self.backend == "numba":
            return self._numba_combined_hop(
                current, self.sub_indices, self.sub_indptr,
                self.obj_indices, self.obj_indptr, self.num_entities
            )

        elif self.backend == "torch":
            curr_t = torch.from_numpy(current).to(dtype=torch.float32, device=self.device).reshape(1, -1)
            trips = torch.sparse.mm(curr_t, self.sub_matrix).squeeze()
            trips = (trips > 0).to(dtype=torch.float32).reshape(1, -1)
            next_ents = torch.sparse.mm(trips, self.obj_matrix).squeeze()
            return (next_ents > 0).cpu().numpy().astype(np.uint8)

        return np.zeros(self.num_entities, dtype=np.uint8)

    def _execute_hop_with_triplets(self, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute hop and return (active_triplets, next_entities)."""
        if self.backend == "scipy":
            curr = sp.csr_matrix(current.reshape(1, -1))
            trips = (curr @ self.sub_matrix).toarray().flatten()
            next_vec = ((sp.csr_matrix(trips.reshape(1, -1)) @ self.obj_matrix)
                        .toarray().flatten() > 0).astype(np.uint8)
            return trips, next_vec

        elif self.backend == "numba":
            trips = np.zeros(self.num_triplets, dtype=np.uint8)
            for i in range(len(current)):
                if current[i] > 0:
                    for idx in range(self.sub_indptr[i], self.sub_indptr[i + 1]):
                        trips[self.sub_indices[idx]] = 1
            next_vec = self._execute_hop(current)
            return trips, next_vec

        elif self.backend == "torch":
            ent_idx = np.where(current > 0)[0]
            if len(ent_idx) == 0:
                return (np.zeros(self.num_triplets, dtype=np.uint8),
                        np.zeros(self.num_entities, dtype=np.uint8))

            indices = torch.tensor([[0] * len(ent_idx), ent_idx], device=self.device)
            curr = torch.sparse_coo_tensor(
                indices, torch.ones(len(ent_idx), dtype=torch.float32, device=self.device),
                (1, self.num_entities), device=self.device
            ).coalesce()

            trips_sparse = torch.sparse.mm(curr, self.sub_matrix)
            trips = trips_sparse.to_dense().squeeze().cpu().numpy()
            next_vec = (torch.sparse.mm(trips_sparse, self.obj_matrix)
                        .to_dense().squeeze().cpu().numpy() > 0).astype(np.uint8)
            return trips, next_vec

        return (np.zeros(self.num_triplets, dtype=np.uint8),
                np.zeros(self.num_entities, dtype=np.uint8))

    def _get_entity_triplets(self, entity_idx: int):
        """Get triplet indices for an entity."""
        if self.backend == "scipy":
            return self.sub_matrix.getrow(entity_idx).nonzero()[1]
        elif self.backend == "numba":
            return self.sub_indices[self.sub_indptr[entity_idx]:self.sub_indptr[entity_idx + 1]]
        elif self.backend == "torch":
            row = self.sub_matrix[entity_idx].coalesce()
            return row.indices()[0].cpu().numpy() if row._nnz() > 0 else []

    def _get_triplet_info(self, triplet_idx: int) -> Tuple:
        """Get (relation_idx, object_idx) for a triplet."""
        if self.backend == "scipy":
            rel = self.rel_matrix.getrow(triplet_idx).nonzero()[1]
            obj = self.obj_matrix.getrow(triplet_idx).nonzero()[1]
            return (rel[0] if len(rel) > 0 else None,
                    obj[0] if len(obj) > 0 else None)

        elif self.backend == "numba":
            rel_start = self.rel_indptr[triplet_idx]
            obj_start = self.obj_indptr[triplet_idx]
            return (self.rel_indices[rel_start] if rel_start < self.rel_indptr[triplet_idx + 1] else None,
                    self.obj_indices[obj_start] if obj_start < self.obj_indptr[triplet_idx + 1] else None)

        elif self.backend == "torch":
            rel_row = self.rel_matrix[triplet_idx].coalesce()
            obj_row = self.obj_matrix[triplet_idx].coalesce()
            return (rel_row.indices()[0][0].item() if rel_row._nnz() > 0 else None,
                    obj_row.indices()[0][0].item() if obj_row._nnz() > 0 else None)

    # ========================================
    # Backend Construction
    # ========================================

    def _build_backend_matrices(self, idx_triplets: List[Tuple[int, int, int]]):
        """Build matrices for the selected backend."""
        if not idx_triplets:
            self._create_empty_matrices()
            return

        if self.backend == "scipy":
            self._build_scipy(idx_triplets)
        elif self.backend == "numba":
            self._build_numba(idx_triplets)
        elif self.backend == "torch":
            self._build_torch(idx_triplets)

    def _create_empty_matrices(self):
        """Create empty matrices."""
        if self.backend == "scipy":
            self.sub_matrix = sp.csr_matrix((self.num_entities, 0))
            self.rel_matrix = sp.csr_matrix((0, self.num_relations))
            self.obj_matrix = sp.csr_matrix((0, self.num_entities))
        elif self.backend == "numba":
            self.sub_indices = np.array([], dtype=np.uint32)
            self.sub_indptr = np.zeros(self.num_entities + 1, dtype=np.uint32)
            self.rel_indices = np.array([], dtype=np.uint32)
            self.rel_indptr = np.zeros(1, dtype=np.uint32)
            self.obj_indices = np.array([], dtype=np.uint32)
            self.obj_indptr = np.zeros(1, dtype=np.uint32)
        elif self.backend == "torch":
            empty_idx = torch.zeros((2, 0), dtype=torch.int64, device=self.device)
            empty_val = torch.zeros(0, dtype=torch.float32, device=self.device)
            self.sub_matrix = torch.sparse_coo_tensor(empty_idx, empty_val,
                                                      (self.num_entities, 0), device=self.device)
            self.rel_matrix = torch.sparse_coo_tensor(empty_idx, empty_val,
                                                      (0, self.num_relations), device=self.device)
            self.obj_matrix = torch.sparse_coo_tensor(empty_idx, empty_val,
                                                      (0, self.num_entities), device=self.device)

    def _build_scipy(self, idx_triplets):
        """Build SciPy matrices."""
        sub_r = np.array([s for s, r, o in idx_triplets], dtype=np.int32)
        sub_c = np.arange(self.num_triplets, dtype=np.int32)
        self.sub_matrix = sp.csr_matrix(
            (np.ones(self.num_triplets, dtype=np.bool_), (sub_r, sub_c)),
            shape=(self.num_entities, self.num_triplets), dtype=np.bool_
        ).sorted_indices()

        rel_r = np.arange(self.num_triplets, dtype=np.int32)
        rel_c = np.array([r for s, r, o in idx_triplets], dtype=np.int32)
        self.rel_matrix = sp.csr_matrix(
            (np.ones(self.num_triplets, dtype=np.bool_), (rel_r, rel_c)),
            shape=(self.num_triplets, self.num_relations), dtype=np.bool_
        )

        obj_r = np.arange(self.num_triplets, dtype=np.int32)
        obj_c = np.array([o for s, r, o in idx_triplets], dtype=np.int32)
        self.obj_matrix = sp.csr_matrix(
            (np.ones(self.num_triplets, dtype=np.bool_), (obj_r, obj_c)),
            shape=(self.num_triplets, self.num_entities), dtype=np.bool_
        ).sorted_indices()

    def _build_numba(self, idx_triplets):
        """Build Numba matrices."""
        sub_counts = np.zeros(self.num_entities, dtype=np.uint32)
        for s, r, o in idx_triplets:
            sub_counts[s] += 1

        self.sub_indptr = np.zeros(self.num_entities + 1, dtype=np.uint32)
        self.sub_indptr[1:] = np.cumsum(sub_counts)
        self.sub_indices = np.zeros(self.num_triplets, dtype=np.uint32)

        pos = np.copy(self.sub_indptr[:-1])
        for t_idx, (s, r, o) in enumerate(idx_triplets):
            self.sub_indices[pos[s]] = t_idx
            pos[s] += 1

        self.rel_indices = np.array([r for s, r, o in idx_triplets], dtype=np.uint32)
        self.rel_indptr = np.arange(self.num_triplets + 1, dtype=np.uint32)
        self.obj_indices = np.array([o for s, r, o in idx_triplets], dtype=np.uint32)
        self.obj_indptr = np.arange(self.num_triplets + 1, dtype=np.uint32)

    def _build_torch(self, idx_triplets):
        """Build Torch matrices."""
        sub_r = torch.tensor([s for s, r, o in idx_triplets], dtype=torch.int64, device=self.device)
        sub_c = torch.arange(self.num_triplets, dtype=torch.int64, device=self.device)
        self.sub_matrix = torch.sparse_coo_tensor(
            torch.stack([sub_r, sub_c]),
            torch.ones(self.num_triplets, dtype=torch.float32, device=self.device),
            (self.num_entities, self.num_triplets), device=self.device
        ).coalesce().to_sparse_csr()

        rel_r = torch.arange(self.num_triplets, dtype=torch.int64, device=self.device)
        rel_c = torch.tensor([r for s, r, o in idx_triplets], dtype=torch.int64, device=self.device)
        self.rel_matrix = torch.sparse_coo_tensor(
            torch.stack([rel_r, rel_c]),
            torch.ones(self.num_triplets, dtype=torch.float32, device=self.device),
            (self.num_triplets, self.num_relations), device=self.device
        ).coalesce()

        obj_r = torch.arange(self.num_triplets, dtype=torch.int64, device=self.device)
        obj_c = torch.tensor([o for s, r, o in idx_triplets], dtype=torch.int64, device=self.device)
        self.obj_matrix = torch.sparse_coo_tensor(
            torch.stack([obj_r, obj_c]),
            torch.ones(self.num_triplets, dtype=torch.float32, device=self.device),
            (self.num_triplets, self.num_entities), device=self.device
        ).coalesce().to_sparse_csr()

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _numba_combined_hop(inp, sub_idx, sub_ptr, obj_idx, obj_ptr, n_ent):
        """JIT-compiled combined hop operation."""
        n_trip = len(obj_ptr) - 1
        inter = np.zeros(n_trip, dtype=np.uint8)

        for i in range(len(inp)):
            if inp[i] > 0:
                for idx in range(sub_ptr[i], sub_ptr[i + 1]):
                    inter[sub_idx[idx]] = 1

        result = np.zeros(n_ent, dtype=np.uint8)
        for i in range(len(inter)):
            if inter[i] > 0:
                for idx in range(obj_ptr[i], obj_ptr[i + 1]):
                    result[obj_idx[idx]] = 1

        return result

    def _torch_batched_k_hop(self, entity_ids: List[str], hops: int, shortest_path: bool = False) -> List[str]:
        seed_idx = [self.cui_to_idx[e] for e in entity_ids if e in self.cui_to_idx]
        if not seed_idx:
            return []

        n_seeds = len(seed_idx)
        current = torch.zeros((n_seeds, self.num_entities), dtype=torch.float32, device=self.device)
        for i, idx in enumerate(seed_idx):
            current[i, idx] = 1

        if shortest_path:
            visited = current.clone()
            for _ in range(hops):
                trips = torch.sparse.mm(current, self.sub_matrix)
                trips = (trips > 0).to(dtype=torch.float32)
                nxt = torch.sparse.mm(trips, self.obj_matrix)
                nxt = (nxt > 0).to(dtype=torch.float32) * (1 - visited)
                current = nxt
                visited = torch.maximum(visited, current)
                if current.sum() == 0:
                    break
        else:
            for _ in range(hops):
                trips = torch.sparse.mm(current, self.sub_matrix)
                trips = (trips > 0).to(dtype=torch.float32)
                current = torch.sparse.mm(trips, self.obj_matrix)
                current = (current > 0).to(dtype=torch.float32)
                if current.sum() == 0:
                    break

        total = (current.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
        return self._indices_to_entities(total, exclude=set(entity_ids))
