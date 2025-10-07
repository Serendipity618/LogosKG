import numpy as np
import scipy.sparse as sp
from numba import njit
from typing import List, Dict, Tuple, Optional, Set
import torch
import pickle
import gc
from pathlib import Path
from collections import defaultdict


class LogosKGLarge:
    """
    Fast optimized partitioned knowledge graph with LRU caching.

    """

    def __init__(self, partition_dir: str, backend: str = 'numba', device: str = 'cpu',
                 max_loaded_partitions: int = 8, flush_memory: bool = True):
        """
        Initialize partitioned knowledge graph.

        Args:
            partition_dir: Directory containing partition files
            backend: 'scipy', 'numba', or 'torch'
            device: 'cpu' or 'cuda' (for torch)
            max_loaded_partitions: Maximum partitions to keep in memory
            flush_memory: Clear triplets after building matrices
        """
        self.partition_dir = Path(partition_dir)
        self.backend = backend
        self.device = device
        self.max_loaded_partitions = max_loaded_partitions
        self.flush_memory = flush_memory
        self._validate_backend(backend)

        # Cache management
        self.cache_dir = self.partition_dir / "cache" / backend
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.cache_order = []

        # Entity to partition mapping
        self.entity_to_partitions = {}

        # Statistics
        self.load_count = 0
        self.evict_count = 0

        # Load metadata
        self._load_metadata()

    # ========================================
    # Public API
    # ========================================

    def retrieve_at_k_hop(self, entity_ids: List[str], hops: int = 2, shortest_path: bool = False) -> List[str]:
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
        start_set = set(entity_ids)
        total_frontier = set()

        for seed in entity_ids:
            if seed not in self.entity_to_partitions:
                continue

            current_entities = {seed}

            if shortest_path:
                visited = set(current_entities)
                for _ in range(hops):
                    if not current_entities:
                        break
                    next_entities = set()
                    entities_by_partition = self._group_by_partition(current_entities)
                    for part_id, partition_entities in entities_by_partition.items():
                        next_ents = self._process_partition_hop(part_id, partition_entities)
                        next_entities.update(next_ents)
                    next_entities -= visited
                    current_entities = next_entities
                    visited |= current_entities
            else:
                for _ in range(hops):
                    if not current_entities:
                        break
                    next_entities = set()
                    entities_by_partition = self._group_by_partition(current_entities)
                    for part_id, partition_entities in entities_by_partition.items():
                        next_ents = self._process_partition_hop(part_id, partition_entities)
                        next_entities.update(next_ents)
                    current_entities = next_entities

            total_frontier |= current_entities

        return [e for e in total_frontier if e not in start_set]

    def retrieve_within_k_hop(self, entity_ids: List[str], hops: int = 2) -> List[str]:
        """
        Return entities within k hops from any starting entity.
        Multi-source with global visited tracking.
        """
        current_entities = set(entity_ids)
        all_reachable = set(entity_ids)

        for _ in range(hops):
            if not current_entities:
                break

            next_entities = set()
            entities_by_partition = self._group_by_partition(current_entities)

            for part_id, partition_entities in entities_by_partition.items():
                next_ents = self._process_partition_hop(part_id, partition_entities)
                next_entities.update(next_ents)

            next_entities -= all_reachable
            all_reachable.update(next_entities)
            current_entities = next_entities

        return list(all_reachable - set(entity_ids))

    def retrieve_with_paths_at_k_hop(self, entity_ids: List[str], hops: int = 2) -> Dict:
        """
        Return entities at exactly k hops with ALL possible k-hop paths.
        """
        all_paths = {}

        for seed in entity_ids:
            if seed not in self.entity_to_partitions:
                continue

            # paths_at_hop[h] = {entity: [paths of length h]}
            paths_at_hop = [{seed: [[seed]]}]  # hop 0

            for hop in range(hops):
                current_paths = paths_at_hop[hop]
                next_paths = {}

                # Process all entities at current hop
                current_entities = set(current_paths.keys())
                entities_by_partition = self._group_by_partition(current_entities)

                for part_id, partition_entities in entities_by_partition.items():
                    partition_data = self._load_partition(part_id)
                    if partition_data is None or partition_data['matrices'] is None:
                        continue

                    local_vector = self._entities_to_local_vector(partition_entities, partition_data)
                    if np.sum(local_vector) == 0:
                        continue

                    active_triplets, _ = self._execute_hop_with_triplets(local_vector, partition_data)

                    for ent_idx in np.where(local_vector > 0)[0]:
                        ent_name = partition_data['local_idx_to_cui'][ent_idx]
                        if ent_name not in current_paths:
                            continue

                        entity_triplets = self._get_entity_triplets(ent_idx, partition_data)

                        for trip_idx in entity_triplets:
                            if active_triplets[trip_idx] > 0:
                                rel_idx, obj_idx = self._get_triplet_info(trip_idx, partition_data)
                                if rel_idx is None or obj_idx is None:
                                    continue

                                rel_name = partition_data['local_idx_to_rel'][rel_idx]
                                obj_name = partition_data['local_idx_to_cui'][obj_idx]

                                for path in current_paths[ent_name]:
                                    new_path = path + [f"--{rel_name}-->", obj_name]
                                    if obj_name not in next_paths:
                                        next_paths[obj_name] = []
                                    next_paths[obj_name].append(new_path)

                paths_at_hop.append(next_paths)

            # Collect k-hop paths
            if hops < len(paths_at_hop):
                for entity, paths in paths_at_hop[hops].items():
                    if entity not in all_paths:
                        all_paths[entity] = []
                    all_paths[entity].extend(paths)

        result_entities = list(all_paths.keys())
        return {"entities": result_entities, "paths": all_paths}

    def retrieve_with_paths_within_k_hop(self, entity_ids: List[str], hops: int = 2) -> Dict:
        """
        Return entities within k hops with ALL possible paths (up to k hops).
        """
        # Store ALL paths across all hops
        all_paths = {}

        for seed in entity_ids:
            if seed not in self.entity_to_partitions:
                continue

            # paths_at_hop[h] = {entity: [paths of length h]}
            paths_at_hop = [{seed: [[seed]]}]  # hop 0

            for hop in range(hops):
                current_paths = paths_at_hop[hop]
                next_paths = {}

                current_entities = set(current_paths.keys())
                entities_by_partition = self._group_by_partition(current_entities)

                for part_id, partition_entities in entities_by_partition.items():
                    partition_data = self._load_partition(part_id)
                    if partition_data is None or partition_data['matrices'] is None:
                        continue

                    local_vector = self._entities_to_local_vector(partition_entities, partition_data)
                    if np.sum(local_vector) == 0:
                        continue

                    active_triplets, _ = self._execute_hop_with_triplets(local_vector, partition_data)

                    for ent_idx in np.where(local_vector > 0)[0]:
                        ent_name = partition_data['local_idx_to_cui'][ent_idx]
                        if ent_name not in current_paths:
                            continue

                        entity_triplets = self._get_entity_triplets(ent_idx, partition_data)

                        for trip_idx in entity_triplets:
                            if active_triplets[trip_idx] > 0:
                                rel_idx, obj_idx = self._get_triplet_info(trip_idx, partition_data)
                                if rel_idx is None or obj_idx is None:
                                    continue

                                rel_name = partition_data['local_idx_to_rel'][rel_idx]
                                obj_name = partition_data['local_idx_to_cui'][obj_idx]

                                for path in current_paths[ent_name]:
                                    new_path = path + [f"--{rel_name}-->", obj_name]
                                    if obj_name not in next_paths:
                                        next_paths[obj_name] = []
                                    next_paths[obj_name].append(new_path)

                paths_at_hop.append(next_paths)

            # Collect ALL paths from hop 1 to k
            for hop_level in range(1, len(paths_at_hop)):
                for entity, paths in paths_at_hop[hop_level].items():
                    if entity == seed:
                        continue
                    if entity not in all_paths:
                        all_paths[entity] = []
                    all_paths[entity].extend(paths)

        result_entities = list(all_paths.keys())
        return {"entities": result_entities, "paths": all_paths}

    def batch_retrieve_at_k_hop(self, batch_entity_ids: List[List[str]], hops: int = 2) -> List[List[str]]:
        """Batch processing with query reordering for cache efficiency."""
        return self._batch_process(batch_entity_ids, lambda ids: self.retrieve_at_k_hop(ids, hops))

    def batch_retrieve_within_k_hop(self, batch_entity_ids: List[List[str]], hops: int = 2) -> List[List[str]]:
        """Batch processing with query reordering for cache efficiency."""
        return self._batch_process(batch_entity_ids, lambda ids: self.retrieve_within_k_hop(ids, hops))

    def batch_retrieve_with_paths_at_k_hop(self, batch_entity_ids: List[List[str]], hops: int = 2) -> List[Dict]:
        """Batch processing with query reordering for cache efficiency."""
        return self._batch_process(batch_entity_ids, lambda ids: self.retrieve_with_paths_at_k_hop(ids, hops))

    def batch_retrieve_with_paths_within_k_hop(self, batch_entity_ids: List[List[str]], hops: int = 2) -> List[Dict]:
        """Batch processing with query reordering for cache efficiency."""
        return self._batch_process(batch_entity_ids, lambda ids: self.retrieve_with_paths_within_k_hop(ids, hops))

    def info(self) -> Dict:
        """Get system information."""
        return {
            'backend': self.backend,
            'device': self.device,
            'num_partitions': self.num_partitions,
            'max_loaded_partitions': self.max_loaded_partitions,
            'currently_loaded': len(self.cache),
            'loaded_partition_ids': list(self.cache.keys()),
            'cache_dir': str(self.cache_dir),
            'load_count': self.load_count,
            'evict_count': self.evict_count,
        }

    # ========================================
    # Helper Methods
    # ========================================

    def _validate_backend(self, backend: str):
        """Validate backend selection."""
        available = ['scipy', 'numba', 'torch']
        if backend not in available:
            raise ValueError(f"Backend '{backend}' not available. Choose from: {available}")

    def _load_metadata(self):
        """Load entity-to-partition mappings."""
        mapping_file = self.partition_dir / "subject_to_partition.pkl"

        if mapping_file.exists():
            with open(mapping_file, 'rb') as f:
                self.entity_to_partitions = pickle.load(f)
        else:
            print("Warning: subject_to_partition.pkl not found")

        partition_files = list(self.partition_dir.glob("part_*.pkl"))
        self.num_partitions = len(partition_files)

    def _group_by_partition(self, entities: Set[str]) -> Dict[int, Set[str]]:
        """Group entities by their partition IDs."""
        grouped = defaultdict(set)
        for entity in entities:
            if entity in self.entity_to_partitions:
                part_id = self.entity_to_partitions[entity]
                grouped[part_id].add(entity)
        return grouped

    def _entities_to_local_vector(self, entities: Set[str], partition_data: Dict) -> np.ndarray:
        """Convert entity set to local binary vector."""
        local_vector = np.zeros(partition_data['num_local_entities'], dtype=np.uint8)
        for entity in entities:
            if entity in partition_data['local_cui_to_idx']:
                local_idx = partition_data['local_cui_to_idx'][entity]
                local_vector[local_idx] = 1
        return local_vector

    def _batch_process(self, batch_entity_ids: List[List[str]], process_fn) -> List:
        """Process batch with query reordering for cache efficiency."""
        query_info = self._analyze_query_partitions(batch_entity_ids)
        sorted_query_info = self._sort_by_partition_similarity(query_info)

        reordered_batch = [batch_entity_ids[q_idx] for q_idx, _ in sorted_query_info]
        original_indices = [q_idx for q_idx, _ in sorted_query_info]

        reordered_results = [process_fn(entity_ids) for entity_ids in reordered_batch]

        results = [None] * len(batch_entity_ids)
        for i, original_idx in enumerate(original_indices):
            results[original_idx] = reordered_results[i]

        return results

    def _analyze_query_partitions(self, batch_entities: List[List[str]]) -> List[Tuple[int, Set[int]]]:
        """Analyze partition requirements for each query."""
        query_info = []
        for i, entity_list in enumerate(batch_entities):
            required_partitions = set()
            for entity in entity_list:
                if entity in self.entity_to_partitions:
                    required_partitions.add(self.entity_to_partitions[entity])
            query_info.append((i, required_partitions))
        return query_info

    def _sort_by_partition_similarity(self, query_info: List[Tuple[int, Set[int]]]) -> List[Tuple[int, Set[int]]]:
        """Sort queries by partition requirements for better cache hits."""
        return sorted(query_info, key=lambda item: tuple(sorted(item[1])))

    # ========================================
    # Partition Management
    # ========================================

    def _load_partition(self, part_id: int) -> Optional[Dict]:
        """Load partition with LRU caching."""
        # Cache hit
        if part_id in self.cache:
            self.cache_order.remove(part_id)
            self.cache_order.append(part_id)
            cached_data = self.cache[part_id]
            if self.backend == 'torch' and cached_data['matrices'] is not None:
                self._move_matrices_to_device(cached_data['matrices'])
            return cached_data

        # Evict if needed
        if len(self.cache) >= self.max_loaded_partitions:
            self._evict_partition()

        self.load_count += 1

        # Try cached processed partition
        cache_file = self.cache_dir / f"partition_{part_id}_processed.pkl"
        try:
            with open(cache_file, 'rb') as f:
                partition_data = pickle.load(f)
                if self.backend == 'torch' and partition_data['matrices'] is not None:
                    self._move_matrices_to_device(partition_data['matrices'])
            self.cache[part_id] = partition_data
            self.cache_order.append(part_id)
            return partition_data
        except:
            pass

        # Load and process raw partition
        partition_file = self.partition_dir / f"part_{part_id}.pkl"
        if not partition_file.exists():
            return None

        with open(partition_file, 'rb') as f:
            triplets = pickle.load(f)

        partition_data = self._process_partition(part_id, triplets)

        # Save processed partition
        with open(cache_file, 'wb') as f:
            pickle.dump(partition_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.cache[part_id] = partition_data
        self.cache_order.append(part_id)

        return partition_data

    def _evict_partition(self):
        """Evict LRU partition."""
        if self.cache_order:
            lru_id = self.cache_order.pop(0)
            del self.cache[lru_id]
            self.evict_count += 1
            gc.collect()

    def _move_matrices_to_device(self, matrices: Dict):
        """Move torch matrices to the specified device."""
        if 'sub_matrix' in matrices:
            matrices['sub_matrix'] = matrices['sub_matrix'].to(self.device)
            matrices['rel_matrix'] = matrices['rel_matrix'].to(self.device)
            matrices['obj_matrix'] = matrices['obj_matrix'].to(self.device)

    def _process_partition(self, part_id: int, triplets: List[Tuple[str, str, str]]) -> Dict:
        """Process raw triplets into partition data structure."""
        partition_info = {
            'id': part_id,
            'local_cui_to_idx': {},
            'local_idx_to_cui': {},
            'local_rel_to_idx': {},
            'local_idx_to_rel': {},
            'num_local_entities': 0,
            'num_local_relations': 0,
            'num_triplets': len(triplets),
            'matrices': None
        }

        # Build local indices
        local_triplets = []
        for sub, rel, obj in triplets:
            if sub not in partition_info['local_cui_to_idx']:
                idx = len(partition_info['local_cui_to_idx'])
                partition_info['local_cui_to_idx'][sub] = idx
                partition_info['local_idx_to_cui'][idx] = sub

            if obj not in partition_info['local_cui_to_idx']:
                idx = len(partition_info['local_cui_to_idx'])
                partition_info['local_cui_to_idx'][obj] = idx
                partition_info['local_idx_to_cui'][idx] = obj

            if rel not in partition_info['local_rel_to_idx']:
                idx = len(partition_info['local_rel_to_idx'])
                partition_info['local_rel_to_idx'][rel] = idx
                partition_info['local_idx_to_rel'][idx] = rel

            local_triplets.append((
                partition_info['local_cui_to_idx'][sub],
                partition_info['local_rel_to_idx'][rel],
                partition_info['local_cui_to_idx'][obj]
            ))

        partition_info['num_local_entities'] = len(partition_info['local_cui_to_idx'])
        partition_info['num_local_relations'] = len(partition_info['local_rel_to_idx'])

        if local_triplets:
            partition_info['matrices'] = self._build_matrices(local_triplets, partition_info)

        if self.flush_memory:
            del triplets, local_triplets
            gc.collect()

        return partition_info

    # ========================================
    # Core Execution
    # ========================================

    def _process_partition_hop(self, part_id: int, entities: Set[str]) -> Set[str]:
        """Process one hop for entities in a partition."""
        partition_data = self._load_partition(part_id)
        if partition_data is None or partition_data['matrices'] is None:
            return set()

        local_vector = self._entities_to_local_vector(entities, partition_data)
        if np.sum(local_vector) == 0:
            return set()

        next_vector = self._execute_hop(local_vector, partition_data)

        next_entities = set()
        for local_idx in np.where(next_vector > 0)[0]:
            next_entities.add(partition_data['local_idx_to_cui'][local_idx])

        return next_entities

    def _execute_hop(self, current: np.ndarray, partition_data: Dict) -> np.ndarray:
        """Execute one hop using backend-specific optimizations."""
        matrices = partition_data['matrices']

        if self.backend == 'scipy':
            curr = sp.csr_matrix(current.reshape(1, -1), dtype=np.bool_)
            result = curr @ matrices['sub_matrix'] @ matrices['obj_matrix']
            return (result.toarray().ravel() > 0).astype(np.uint8)

        elif self.backend == 'numba':
            return self._numba_combined_hop(
                current, matrices['sub_indices'], matrices['sub_indptr'],
                matrices['obj_indices'], matrices['obj_indptr'],
                partition_data['num_local_entities']
            )

        elif self.backend == 'torch':
            curr_t = torch.from_numpy(current).to(dtype=torch.float32, device=self.device)
            trips = torch.sparse.mm(matrices['sub_matrix'].t(), curr_t.unsqueeze(1)).squeeze()
            trips = (trips > 0).to(dtype=torch.float32)
            next_ents = torch.sparse.mm(matrices['obj_matrix'].t(), trips.unsqueeze(1)).squeeze()
            return (next_ents > 0).cpu().numpy().astype(np.uint8)

        return np.zeros(partition_data['num_local_entities'], dtype=np.uint8)

    def _execute_hop_with_triplets(self, current: np.ndarray, partition_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Execute hop and return (active_triplets, next_entities)."""
        matrices = partition_data['matrices']

        if self.backend == 'scipy':
            curr = sp.csr_matrix(current.reshape(1, -1))
            trips = (curr @ matrices['sub_matrix']).toarray().flatten()
            next_vec = ((sp.csr_matrix(trips.reshape(1, -1)) @ matrices['obj_matrix'])
                        .toarray().flatten() > 0).astype(np.uint8)
            return trips, next_vec

        elif self.backend == 'numba':
            trips = np.zeros(partition_data['num_triplets'], dtype=np.uint8)
            for i in range(len(current)):
                if current[i] > 0:
                    for idx in range(matrices['sub_indptr'][i], matrices['sub_indptr'][i + 1]):
                        trips[matrices['sub_indices'][idx]] = 1
            next_vec = self._execute_hop(current, partition_data)
            return trips, next_vec

        elif self.backend == 'torch':
            ent_idx = np.where(current > 0)[0]
            if len(ent_idx) == 0:
                return (np.zeros(partition_data['num_triplets'], dtype=np.uint8),
                        np.zeros(partition_data['num_local_entities'], dtype=np.uint8))

            indices = torch.tensor([[0] * len(ent_idx), ent_idx], device=self.device)
            curr = torch.sparse_coo_tensor(
                indices, torch.ones(len(ent_idx), dtype=torch.float32, device=self.device),
                (1, partition_data['num_local_entities']), device=self.device
            ).coalesce()

            trips_sparse = torch.sparse.mm(curr, matrices['sub_matrix'])
            trips = trips_sparse.to_dense().squeeze().cpu().numpy()
            next_vec = (torch.sparse.mm(trips_sparse, matrices['obj_matrix'])
                        .to_dense().squeeze().cpu().numpy() > 0).astype(np.uint8)
            return trips, next_vec

        return (np.zeros(partition_data['num_triplets'], dtype=np.uint8),
                np.zeros(partition_data['num_local_entities'], dtype=np.uint8))

    def _get_entity_triplets(self, entity_idx: int, partition_data: Dict):
        """Get triplet indices for an entity."""
        matrices = partition_data['matrices']

        if self.backend == 'scipy':
            return matrices['sub_matrix'].getrow(entity_idx).nonzero()[1]
        elif self.backend == 'numba':
            start = matrices['sub_indptr'][entity_idx]
            end = matrices['sub_indptr'][entity_idx + 1]
            return matrices['sub_indices'][start:end]
        elif self.backend == 'torch':
            row = matrices['sub_matrix'][entity_idx].coalesce()
            return row.indices()[0].cpu().numpy() if row._nnz() > 0 else []

    def _get_triplet_info(self, triplet_idx: int, partition_data: Dict) -> Tuple:
        """Get (relation_idx, object_idx) for a triplet."""
        matrices = partition_data['matrices']

        if self.backend == 'scipy':
            rel = matrices['rel_matrix'].getrow(triplet_idx).nonzero()[1]
            obj = matrices['obj_matrix'].getrow(triplet_idx).nonzero()[1]
            return (rel[0] if len(rel) > 0 else None,
                    obj[0] if len(obj) > 0 else None)

        elif self.backend == 'numba':
            rel_start = matrices['rel_indptr'][triplet_idx]
            obj_start = matrices['obj_indptr'][triplet_idx]
            return (matrices['rel_indices'][rel_start] if rel_start < matrices['rel_indptr'][triplet_idx + 1] else None,
                    matrices['obj_indices'][obj_start] if obj_start < matrices['obj_indptr'][triplet_idx + 1] else None)

        elif self.backend == 'torch':
            rel_row = matrices['rel_matrix'][triplet_idx].coalesce()
            obj_row = matrices['obj_matrix'][triplet_idx].coalesce()
            return (rel_row.indices()[0][0].item() if rel_row._nnz() > 0 else None,
                    obj_row.indices()[0][0].item() if obj_row._nnz() > 0 else None)

    # ========================================
    # Matrix Construction
    # ========================================

    def _build_matrices(self, triplets: List[Tuple[int, int, int]], partition_info: Dict) -> Optional[Dict]:
        """Build matrices for the backend."""
        if not triplets:
            return None

        if self.backend == 'scipy':
            return self._build_scipy(triplets, partition_info)
        elif self.backend == 'numba':
            return self._build_numba(triplets, partition_info)
        elif self.backend == 'torch':
            return self._build_torch(triplets, partition_info)

    def _build_scipy(self, triplets, partition_info) -> Dict:
        """Build optimized SciPy matrices with bool dtype."""
        n_ent = partition_info['num_local_entities']
        n_rel = partition_info['num_local_relations']
        n_trip = len(triplets)

        sub_r = np.array([s for s, r, o in triplets], dtype=np.int32)
        sub_c = np.arange(n_trip, dtype=np.int32)
        sub_matrix = sp.csr_matrix(
            (np.ones(n_trip, dtype=np.bool_), (sub_r, sub_c)),
            shape=(n_ent, n_trip), dtype=np.bool_
        ).sorted_indices()

        rel_r = np.arange(n_trip, dtype=np.int32)
        rel_c = np.array([r for s, r, o in triplets], dtype=np.int32)
        rel_matrix = sp.csr_matrix(
            (np.ones(n_trip, dtype=np.bool_), (rel_r, rel_c)),
            shape=(n_trip, n_rel), dtype=np.bool_
        )

        obj_r = np.arange(n_trip, dtype=np.int32)
        obj_c = np.array([o for s, r, o in triplets], dtype=np.int32)
        obj_matrix = sp.csr_matrix(
            (np.ones(n_trip, dtype=np.bool_), (obj_r, obj_c)),
            shape=(n_trip, n_ent), dtype=np.bool_
        ).sorted_indices()

        return {'sub_matrix': sub_matrix, 'rel_matrix': rel_matrix, 'obj_matrix': obj_matrix}

    def _build_numba(self, triplets, partition_info) -> Dict:
        """Build optimized Numba CSR matrices."""
        n_ent = partition_info['num_local_entities']
        n_rel = partition_info['num_local_relations']
        n_trip = len(triplets)

        sub_counts = np.zeros(n_ent, dtype=np.uint32)
        for s, r, o in triplets:
            sub_counts[s] += 1

        sub_indptr = np.zeros(n_ent + 1, dtype=np.uint32)
        sub_indptr[1:] = np.cumsum(sub_counts)
        sub_indices = np.zeros(n_trip, dtype=np.uint32)

        pos = np.copy(sub_indptr[:-1])
        for t_idx, (s, r, o) in enumerate(triplets):
            sub_indices[pos[s]] = t_idx
            pos[s] += 1

        rel_indices = np.array([r for s, r, o in triplets], dtype=np.uint32)
        rel_indptr = np.arange(n_trip + 1, dtype=np.uint32)
        obj_indices = np.array([o for s, r, o in triplets], dtype=np.uint32)
        obj_indptr = np.arange(n_trip + 1, dtype=np.uint32)

        return {
            'sub_indices': sub_indices, 'sub_indptr': sub_indptr, 'sub_shape': (n_ent, n_trip),
            'rel_indices': rel_indices, 'rel_indptr': rel_indptr, 'rel_shape': (n_trip, n_rel),
            'obj_indices': obj_indices, 'obj_indptr': obj_indptr, 'obj_shape': (n_trip, n_ent),
        }

    def _build_torch(self, triplets, partition_info) -> Dict:
        """Build optimized Torch matrices with CSR format."""
        n_ent = partition_info['num_local_entities']
        n_rel = partition_info['num_local_relations']
        n_trip = len(triplets)

        sub_r = torch.tensor([s for s, r, o in triplets], dtype=torch.int64, device=self.device)
        sub_c = torch.arange(n_trip, dtype=torch.int64, device=self.device)
        sub_matrix = torch.sparse_coo_tensor(
            torch.stack([sub_r, sub_c]),
            torch.ones(n_trip, dtype=torch.float32, device=self.device),
            (n_ent, n_trip), device=self.device
        ).coalesce().to_sparse_csr()

        rel_r = torch.arange(n_trip, dtype=torch.int64, device=self.device)
        rel_c = torch.tensor([r for s, r, o in triplets], dtype=torch.int64, device=self.device)
        rel_matrix = torch.sparse_coo_tensor(
            torch.stack([rel_r, rel_c]),
            torch.ones(n_trip, dtype=torch.float32, device=self.device),
            (n_trip, n_rel), device=self.device
        ).coalesce()

        obj_r = torch.arange(n_trip, dtype=torch.int64, device=self.device)
        obj_c = torch.tensor([o for s, r, o in triplets], dtype=torch.int64, device=self.device)
        obj_matrix = torch.sparse_coo_tensor(
            torch.stack([obj_r, obj_c]),
            torch.ones(n_trip, dtype=torch.float32, device=self.device),
            (n_trip, n_ent), device=self.device
        ).coalesce().to_sparse_csr()

        return {'sub_matrix': sub_matrix, 'rel_matrix': rel_matrix, 'obj_matrix': obj_matrix}

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
