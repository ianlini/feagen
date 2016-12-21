from os.path import dirname
import re
from collections import defaultdict

import six
from mkdir_p import mkdir_p
import networkx as nx


def draw_dag(nx_dag, path):
    mkdir_p(dirname(path))
    agraph = nx.nx_agraph.to_agraph(nx_dag)
    for edge in agraph.edges_iter():
        edge.attr['label'] = edge.attr['keys']
        if edge.attr['keys'] == "[]":
            edge.attr['label'] = ""
        if (edge.attr['skipped_keys'] != "[]"
                and edge.attr['skipped_keys'] is not None):
            edge.attr['label'] += "(%s skipped)" % edge.attr['skipped_keys']
    for node in agraph.nodes_iter():
        if node.attr['skipped'] == "True":
            node.attr['label'] = str(node) + " (skipped)"
            node.attr['fontcolor'] = 'grey'
    agraph.layout('dot')
    agraph.draw(path)


class DataDAG(object):
    # TODO: This is not really a DAG

    def __init__(self):
        self._data_key_node_dict = {}
        self._nx_dag = nx.DiGraph()

    def add_node(self, function_name, function):
        # pylint: disable=protected-access
        config = function._feagen_will_generate

        for key in config['keys']:
            if key in self._data_key_node_dict:
                raise ValueError("duplicated data key '{}' in {} and {}".format(
                    key, self._data_key_node_dict[key], function_name))
            self._data_key_node_dict[key] = function_name

        self._nx_dag.add_node(function_name, config)

    def get_node_keys_dict(self, data_keys):
        node_keys_dict = defaultdict(list)
        for data_key in data_keys:
            node_keys_dict[self[data_key]].append(data_key)
        return node_keys_dict

    def add_edges_from(self, requirements):
        edges = []
        for function_name, function_requirements in requirements:
            node_keys_dict = self.get_node_keys_dict(function_requirements)
            for source_node, data_keys in six.viewitems(node_keys_dict):
                edges.append((source_node, function_name, {'keys': data_keys}))
        self._nx_dag.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(self._nx_dag):
            raise ValueError("The dependency graph has cycle.")

    def __getitem__(self, key):
        found_node = None
        found_key = None
        for regex_key, node in six.viewitems(self._data_key_node_dict):
            if re.match("(%s)$" % regex_key, key) is not None:
                if found_node is None:
                    found_node = node
                    found_key = regex_key
                else:
                    raise ValueError("The data key '{}' matches multiple keys: "
                                     "'{}' in {} and '{}' in {}.".format(
                                         key, found_key, found_node,
                                         regex_key, node))
        if found_node is None:
            raise KeyError(key)
        return found_node

    def get_node_attr(self, key):
        node = self[key]
        node_attr = self._nx_dag.node[node]
        return node_attr

    def draw(self, path):
        draw_dag(self._nx_dag, path)

    def get_subgraph_with_ancestors(self, nodes):
        subgraph_nodes = set(nodes)
        for node in nodes:
            subgraph_nodes |= nx.ancestors(self._nx_dag, node)
        return self._nx_dag.subgraph(subgraph_nodes)
