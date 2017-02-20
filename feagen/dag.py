from os.path import dirname
import re
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
            if edge.attr['nonskipped_keys'] not in ["set()", "set([])"]:
                edge.attr['label'] += edge.attr['nonskipped_keys']
            if (edge.attr['skipped_keys'] not in ["set()", "set([])"]
                    and edge.attr['skipped_keys'] is not None):
                edge.attr['label'] += "(%s skipped)" % edge.attr['skipped_keys']
    for node in agraph.nodes_iter():
        if node.attr['skipped'] == "True":
            node.attr['label'] = node.attr['__name__'] + " (skipped)"
            node.attr['fontcolor'] = 'grey'
        else:
            node.attr['label'] = node.attr['__name__']
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
        self._node_key_dict = {}
        self._node_attr_dict = {}
        self._node_succesor_dict = {}
        self._node_mode_dict = {}

    def add_node(self, name, keys=(), re_escape_keys=(), successor_keys=(),
                 attr=None, mode='one'):
        # pylint: disable=protected-access
        if name in self._node_attr_dict:
            raise ValueError("duplicated node name '{}' for {} and {}"
                             .format(name, self._node_attr_dict[name], attr))
        self._node_attr_dict[name] = attr
        if isinstance(keys, basestring):
            keys = (keys,)
        if isinstance(re_escape_keys, basestring):
            re_escape_keys = (re_escape_keys,)
        self._node_key_dict[name] = {
            'keys': tuple(sorted(keys)),
            're_escape_keys': tuple(sorted(re_escape_keys)),
        }
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
        self._node_succesor_dict[name] = tuple(sorted(set(successor_keys)))
        self._node_mode_dict[name] = mode

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
        regex_key, node, match_object = self.match_node(key)
        node_attr = self._node_attr_dict[node]
        return node_attr

    def get_subgraph_with_ancestors(self, nodes):
        subgraph_nodes = set(nodes)
        for node in nodes:
            subgraph_nodes |= nx.ancestors(self._nx_dag, node)
        return self._nx_dag.subgraph(subgraph_nodes)

    def _grow_ancestors(self, nx_digraph, root_node_key, successor_keys,
                        re_args={}):
        successor_keys = {k: k.format(**re_args) for k in successor_keys}
        # grow the graph using DFS
        for template_key, key in six.viewitems(successor_keys):
            regex_key, node, match_object = self.match_node(key)

            # for merging node, we use key as the 'key' in nx_digraph
            mode = self._node_mode_dict[node]
            if mode == 'full':
                node_key = (self._node_key_dict[node]['keys']
                            + self._node_key_dict[node]['re_escape_keys'])
            elif mode == 'one':
                node_key = key
            else:
                raise ValueError("Mode '%s' is not supported." % mode)

            re_args = match_object.groupdict()
            if node_key not in nx_digraph:
                attr = self._node_attr_dict[node].copy()
                attr.setdefault('__name__', node)
                attr['__re_args__'] = re_args
                nx_digraph.add_node(node_key, attr)
                self._grow_ancestors(nx_digraph, node_key,
                                     self._node_succesor_dict[node], re_args)
            if not nx_digraph.has_edge(root_node_key, node_key):
                nx_digraph.add_edge(root_node_key, node_key,
                                    keys=set(), template_keys={})
            edge_attr = nx_digraph[root_node_key][node_key]
            edge_attr['keys'].add(key)
            edge_attr['template_keys'].update(((template_key, key),))

    def build_directed_graph(self, keys, root_node_key='root'):
        nx_digraph = nx.DiGraph()
        nx_digraph.add_node(root_node_key, {'__name__': root_node_key})
        self._grow_ancestors(nx_digraph, root_node_key, keys)
        return nx_digraph

    def draw(self, path, keys, root_node_key='root', reverse=False):
        nx_digraph = self.build_directed_graph(keys, root_node_key)
        if reverse:
            nx_digraph.reverse(copy=False)
        draw_dag(nx_digraph, path)
        return nx_digraph
