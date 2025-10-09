from collections import defaultdict
import hypergraphx as hgx
from typing import Dict
import sys
sys.path.append('../')
import summary_utils as sut


def load_hypergraph(file_path: str, node_id_map: Dict[str, int]):

    sn, se, sw, _, node_id_map = sut.load_summary(file_path, node_id_map)
    
    superH_tup = defaultdict(int)
    for sid, x in enumerate(se):
        if len(x) == 0:
            continue
        xtup = tuple(sorted(list(x)))
        superH_tup[xtup] += sw[sid]

    _superH = []
    _superHW = []
    for xtup, w in superH_tup.items():
        _superH.append(xtup)
        _superHW.append(w)
    
    hyperG = hgx.Hypergraph(edge_list=_superH, weighted=True, weights=_superHW)
    hyperG_nodes = set(hyperG.get_nodes())
    for i in range(len(sn)):
        if i not in hyperG_nodes:
            hyperG.add_node(i)
    return hyperG, sn, node_id_map
