from os.path import dirname
import re
from collections import defaultdict
from past.builtins import basestring

import six
from six.moves import map
from mkdir_p import mkdir_p
import networkx as nx


def draw_dag(nx_dag, path):
    if dirname(path) != '':
        mkdir_p(dirname(path))
    agraph = nx.nx_agraph.to_agraph(nx_dag)
    for edge in agraph.edges_iter():
        if edge.attr['nonskipped_keys'] is None:
            edge.attr['label'] = edge.attr['keys']
        else:
            edge.attr['label'] = ""
            if edge.attr['nonskipped_keys'] != "[]":
                edge.attr['label'] += edge.attr['nonskipped_keys']
            if (edge.attr['skipped_keys'] != "[]"
                    and edge.attr['skipped_keys'] is not None):
                edge.attr['label'] += "(%s skipped)" % edge.attr['skipped_keys']
    for node in agraph.nodes_iter():
        if node.attr['skipped'] == "True":
            node.attr['label'] = str(node) + " (skipped)"
            node.attr['fontcolor'] = 'grey'
    agraph.layout('dot')
    agraph.draw(path)


class RegexDiGraph(object):
    """Directed graph that each node is represented by regular expression.

    We can use a string to find a node that has regex matching the string. The
    matching groups can also be the arguments to define the edge. Because the
    successor is matched depending on regex, the graph will be dynamically built
    given some entry points.
    """

    def __init__(self):
        self._key_node_dict = {}
        self._node_attr_dict = {}
        self._nx_dag = nx.DiGraph()

    def add_node(self, name, keys=(), re_escape_keys=(), attr=None):
        # pylint: disable=protected-access
        if name in self._node_attr_dict:
            raise ValueError("duplicated node name '{}' for {} and {}"
                             .format(name, self._node_attr_dict[name], attr))
        self._node_attr_dict[name] = attr
        if isinstance(keys, basestring):
            keys = (keys,)
        if isinstance(re_escape_keys, basestring):
            re_escape_keys = (re_escape_keys,)
        re_escape_keys = map(re.escape, re_escape_keys)
        keys = list(keys) + list(re_escape_keys)
        if len(keys) == 0:
            raise ValueError("keys and re_escape_keys for {} are both empty."
                             .format(name))
        for key in keys:
            if key in self._key_node_dict:
                raise ValueError("duplicated data key '{}' for {} and {}"
                                 .format(key, self._key_node_dict[key], name))
            self._key_node_dict[key] = name

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

    def match_node(self, key):
        found_node = None
        for regex_key, node in six.viewitems(self._key_node_dict):
            match_object = re.match("(?:%s)\Z" % regex_key, key)
            if match_object is not None:
                if found_node is None:
                    found_node = node
                    found_regex_key = regex_key
                    found_match_object = match_object
                else:
                    raise ValueError("The data key '{}' matches multiple keys: "
                                     "'{}' for {} and '{}' for {}.".format(
                                         key, found_regex_key, found_node,
                                         regex_key, node))
        if found_node is None:
            raise KeyError(key)
        return found_regex_key, found_node, found_match_object

    def get_node_attr(self, key):
        node = self[key]
        node_attr = self._nx_dag.node[node]
        return node_attr

    def get_subgraph_with_ancestors(self, nodes):
        subgraph_nodes = set(nodes)
        for node in nodes:
            subgraph_nodes |= nx.ancestors(self._nx_dag, node)
        return self._nx_dag.subgraph(subgraph_nodes)

    def _grow_ancestors(self, nx_digraph, root_node_name, successor_keys):
        # grow the graph using DFS
        for key in successor_keys:
            regex_key, node, match_object = self.match_node(key)
            attr = self._node_attr_dict[node]
            if node not in nx_digraph:
                nx_digraph.add_node(node, attr)
            if not nx_digraph.has_edge(root_node_name, node):
                nx_digraph.add_edge(root_node_name, node, keys=set())
            edge_attr = nx_digraph[root_node_name][node]
            if key in edge_attr['keys']:
                continue
            edge_attr['keys'].add(key)
            re_args = match_object.groupdict()
            node_successor_keys = map(lambda k: k.format(**re_args),
                                      attr['require'])
            self._grow_ancestors(nx_digraph, node, node_successor_keys)

    def build_directed_graph(self, data_keys, root_node_name='root'):
        nx_digraph = nx.DiGraph()
        nx_digraph.add_node(root_node_name)
        self._grow_ancestors(nx_digraph, root_node_name, data_keys)
        return nx_digraph

    def draw(self, path, data_keys, root_node_name='root', reverse=False):
        nx_digraph = self.build_directed_graph(data_keys, root_node_name)
        nx_digraph.reverse(copy=False)
        draw_dag(nx_digraph, path)
