# -*- coding: utf-8 -*-
import torch

# =================
element2idx = {
    "C": 0, "N": 1, "O": 2, "Cl": 3, "F": 4, "Br": 5,
    "S": 6, "P": 7, "I": 8, "H": 9
}

group2idx = {
    "Hydroxyl": 0,
    "Amino": 1,
    "Nitro": 2,
    "Carboxyl": 3,
    "Aromatic": 4,
    "Alkyl": 5,
    "Halogen": 6
}

# =================
ELEMENT_KG = {
    "C": {"isPartOf": ["Aromatic", "Alkyl"]},
    "N": {"isPartOf": ["Amino", "Nitro"]},
    "O": {"isPartOf": ["Hydroxyl", "Carboxyl"]},
    "Cl": {"isPartOf": ["Halogen"]},
    "F": {"isPartOf": ["Halogen"]},
    "Br": {"isPartOf": ["Halogen"]},
    "S": {"isPartOf": ["Thiol", "Sulfonyl"]},
    "I": {"isPartOf": ["Halogen"]},
    "H": {},
    "P": {}
}

# ============================================
#
# ============================================

def add_element_view(mol, x):
    device = x.device
    
    extra_nodes = []  
    extra_x = []      
    node_map = {}     
    atom2ele_edges = []
    atom2ele_attrs = []

    # Step 1
    for idx, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol not in element2idx:
            continue
        ele_key = ('element', symbol)
        if ele_key not in node_map:
            node_map[ele_key] = len(x) + len(extra_nodes)
            extra_nodes.append(ele_key)
            extra_x.append([element2idx[symbol],0])
        ele_idx = node_map[ele_key]
        atom2ele_edges += [[idx, ele_idx], [ele_idx, idx]]
        atom2ele_attrs += [[0, 0], [0, 0]]  #

    # Step 2
    group_edges = []
    group_attrs = []
    for (etype, ename) in extra_nodes:
        if etype != 'element':
            continue
        if ename not in ELEMENT_KG:
            continue
        for group in ELEMENT_KG[ename].get("isPartOf", []):
            group_key = ('group', group)
            if group_key not in node_map:
                node_map[group_key] = len(x) + len(extra_nodes)
                extra_nodes.append(group_key)
                group_idx = group2idx.get(group, len(group2idx))
                extra_x.append([group_idx,0])
            group_idx = node_map[group_key]
            ele_idx = node_map[('element', ename)]
            group_edges += [[group_idx, ele_idx], [ele_idx, group_idx]]
            group_attrs += [[1, 0], [1, 0]]    

    # Step 3
    if extra_x:
        extra_x_tensor = torch.tensor(extra_x, dtype=torch.long, device=device)
        x_element = torch.cat([x, extra_x_tensor], dim=0)
        edge_index_element = torch.tensor(atom2ele_edges + group_edges, dtype=torch.long, device=device).t().contiguous()
        edge_attr_element = torch.tensor(atom2ele_attrs + group_attrs, dtype=torch.long, device=device)
        return x_element, edge_index_element, edge_attr_element
    else:
        return x, torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0, 1), dtype=torch.long, device=device)
