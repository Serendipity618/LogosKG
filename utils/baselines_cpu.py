import networkx as nx
import igraph as ig
import snap
import graphblas as gb
import graph_tool.all as gt
from neo4j import GraphDatabase
import os
import math
from typing import List, Tuple, Dict, Optional

os.environ.setdefault("NEO4J_URI", "bolt://127.0.0.1:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASS", "neo4j")


class KnowledgeGraphRetrievalBaselinesCPU:

    def __init__(self, triplets: List[Tuple[str, str, str]], backend: Optional[str] = None):
        """
        Initialize CPU baseline with specified backend.

        Args:
            triplets: List of (subject, relation, object) tuples
            backend: One of 'networkx', 'igraph', 'graphtool', 'snap', 'graphblas', or None (build all)
        """
        self.triplets = triplets
        self.backend = backend
        self._build_mappings()

        if backend is None:
            # self._build_graphs()
            pass
        elif backend == 'networkx':
            self._build_networkx_graph()
        elif backend == 'igraph':
            self._build_igraph_graph()
        elif backend == 'graphtool':
            self._build_graph_tool_graph()
        elif backend == 'snap':
            self._build_snap_graph()
        elif backend == 'graphblas':
            self._build_graphblas_graph()

    def _build_mappings(self):
        entities, relations = set(), set()
        for s, r, o in self.triplets:
            entities.add(s)
            entities.add(o)
            relations.add(r)
        self.entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
        self.id_to_entity = {i: e for e, i in self.entity_to_id.items()}
        self.num_entities = len(entities)
        self.num_relations = len(relations)

    def _build_graphs(self):
        self._build_networkx_graph()
        self._build_igraph_graph()
        self._build_graph_tool_graph()
        self._build_snap_graph()
        self._build_graphblas_graph()

    # ---------- NetworkX ----------
    def _build_networkx_graph(self):
        if not hasattr(self, 'G'):
            G = nx.DiGraph()
            for s, r, o in self.triplets:
                G.add_edge(s, o, rel=r)
            self.G = G

    def networkx_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'G'):
            self._build_networkx_graph()

        start_set, found = set(start_cuis), set()
        for s in start_cuis:
            if s not in self.G:
                continue
            try:
                nodes = nx.descendants_at_distance(self.G, s, hops)
            except AttributeError:
                lengths = nx.single_source_shortest_path_length(self.G, s, cutoff=hops)
                nodes = {n for n, d in lengths.items() if d == hops}
            found.update(nodes)
        return list(found - start_set)

    # ---------- igraph ----------
    def _build_igraph_graph(self):
        if not hasattr(self, 'igraph_g'):
            edges = [(self.entity_to_id[s], self.entity_to_id[o]) for s, r, o in self.triplets]
            g = ig.Graph(n=self.num_entities, edges=edges, directed=True)
            g.vs["name"] = [self.id_to_entity[i] for i in range(self.num_entities)]
            self.igraph_g = g

    def igraph_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'igraph_g'):
            self._build_igraph_graph()

        seed_ids = [self.entity_to_id[c] for c in start_cuis if c in self.entity_to_id]
        if not seed_ids:
            return []
        start_set = set(start_cuis)
        neigh_lists = self.igraph_g.neighborhood(vertices=seed_ids, order=hops, mode='OUT', mindist=hops)
        found = {
            self.igraph_g.vs[v]["name"]
            for lst in neigh_lists for v in lst
            if self.igraph_g.vs[v]["name"] not in start_set
        }
        return list(found)

    # ---------- graph-tool ----------
    def _build_graph_tool_graph(self):
        if not hasattr(self, 'graph_tool_g'):
            g = gt.Graph(directed=True)
            g.add_vertex(self.num_entities)
            edges = [(self.entity_to_id[s], self.entity_to_id[o]) for s, r, o in self.triplets]
            g.add_edge_list(edges)
            self.graph_tool_g = g

    def graphtool_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'graph_tool_g'):
            self._build_graph_tool_graph()

        start_set, found = set(start_cuis), set()
        for s in start_cuis:
            if s not in self.entity_to_id:
                continue
            sid = self.entity_to_id[s]
            dist = gt.shortest_distance(self.graph_tool_g, source=self.graph_tool_g.vertex(sid), max_dist=hops)
            for v in self.graph_tool_g.vertices():
                dv = float(dist[v])
                if not math.isinf(dv) and int(dv) == hops:
                    name = self.id_to_entity[int(v)]
                    if name not in start_set:
                        found.add(name)
        return list(found)

    # ---------- SNAP ----------
    def _build_snap_graph(self):
        if not hasattr(self, 'snap_graph'):
            G = snap.TNGraph.New()
            for nid in range(self.num_entities):
                G.AddNode(nid)
            seen = set()
            for s, r, o in self.triplets:
                sid, oid = self.entity_to_id[s], self.entity_to_id[o]
                if (sid, oid) not in seen:
                    G.AddEdge(sid, oid)
                    seen.add((sid, oid))
            self.snap_graph = G

    def snap_khop(self, start_cuis: List[str], hops: int) -> List[str]:
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'snap_graph'):
            self._build_snap_graph()

        start_set, found = set(start_cuis), set()
        for s in start_cuis:
            if s not in self.entity_to_id:
                continue
            sid = self.entity_to_id[s]
            vec = snap.TIntV()
            snap.GetNodesAtHop(self.snap_graph, sid, hops, vec, True)
            for i in range(vec.Len()):
                name = self.id_to_entity[vec[i]]
                if name not in start_set:
                    found.add(name)
        return list(found)

    # ---------- GraphBLAS ----------
    def _build_graphblas_graph(self):
        if not hasattr(self, 'graphblas_matrix'):
            rows = [self.entity_to_id[s] for s, _, o in self.triplets]
            cols = [self.entity_to_id[o] for _, _, o in self.triplets]

            self.graphblas_matrix = gb.Matrix.from_coo(
                rows, cols, [True] * len(rows),
                nrows=self.num_entities, ncols=self.num_entities, dtype=bool,
                dup_op=gb.binary.lor
            )

    def graphblas_khop(self, start_cuis, hops):
        if hops <= 0 or not start_cuis:
            return []

        if not hasattr(self, 'graphblas_matrix'):
            self._build_graphblas_graph()

        result = gb.Vector.from_coo([], [], size=self.num_entities, dtype=bool)
        for cui in start_cuis:
            sid = self.entity_to_id.get(cui)
            if sid is None:
                continue
            frontier = gb.Vector.from_coo([sid], [True], size=self.num_entities, dtype=bool)
            visited = frontier.dup()
            for _ in range(hops):
                frontier = frontier.vxm(self.graphblas_matrix, op=gb.semiring.lor_land).new(mask=~visited.S)
                visited = visited.ewise_add(frontier, op=gb.monoid.lor).new()
            result = result.ewise_add(frontier, op=gb.monoid.lor).new()

        if result.nvals == 0:
            return []
        idx, _ = result.to_coo()
        start_set = set(start_cuis)
        return [self.id_to_entity[i] for i in idx if self.id_to_entity[i] not in start_set]

    def neo4j_khop(self, seeds, k, within=False):
        if k <= 0 or not seeds:
            return []

        rng = f"*..{k}" if within else f"*{k}"
        guard = "" if within or k == 0 else f"AND NOT (s)-[*..{k - 1}]->(n)"

        query = f"""
        UNWIND $seeds AS sid
        MATCH (s {{id: sid}})-[{rng}]->(n)
        WHERE NOT n.id IN $seeds
          {guard}
        RETURN DISTINCT n.id AS id
        """

        uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        pwd = os.environ.get("NEO4J_PASS", "neo4j")

        with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
            with driver.session() as session:
                return [r["id"] for r in session.run(query, seeds=seeds)]

    # ---------- utilities ----------
    def get_available_methods(self) -> List[str]:
        return [
            'networkx_khop',
            'igraph_khop',
            'graphtool_khop',
            'snap_khop',
            'graphblas_khop',
            'neo4j_khop',
        ]

    def get_graph_info(self) -> Dict:
        info = {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'num_triplets': len(self.triplets),
            'available_methods': self.get_available_methods(),
        }
        if hasattr(self, 'G') and self.G is not None:
            info['networkx_nodes'] = self.G.number_of_nodes()
            info['networkx_edges'] = self.G.number_of_edges()
        return info
