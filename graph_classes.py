class Graph:
    def __init__(self):
        self.edge_list = []

    def add_edge(self,is_dir,id_1,id_2,relation,label=None):
        if is_dir:
            edge = DirEdge(id_1,id_2,relation,label)
        else:
            edge = UndirEdge(id_1,id_2,relation)
        self.edge_list.append(edge)

class DirEdge:
    def __init__(self,orig_node,targ_node,relation,label=None):
        self.orig_node = orig_node
        self.targ_node = targ_node
        self.relation = relation
        self.label = label

class UndirEdge:
    def __init__(self,node_one,node_two,relation):
        self.node_one = node_one
        self.node_two = node_two
        self.relation = relation
