import os.path
import inspect
import re
from collections import defaultdict

import h5py
from mkdir_p import mkdir_p
import six
import networkx as nx


def draw_dag(nx_dag, path):
    agraph = nx.nx_agraph.to_agraph(nx_dag)
    for edge in agraph.edges_iter():
        edge.attr['label'] = edge.attr['keys']
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

        self._nx_dag.add_node(function_name, config, function=function)

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

    def draw(self, path):
        draw_dag(self._nx_dag, path)

    def get_subgraph_with_ancestors(self, nodes):
        subgraph_nodes = set(nodes)
        for node in nodes:
            subgraph_nodes |= nx.ancestors(self._nx_dag, node)
        return self._nx_dag.subgraph(subgraph_nodes)


class FeatureGeneratorType(type):

    def __init__(cls, name, bases, attrs):
        # pylint: disable=protected-access
        super(FeatureGeneratorType, cls).__init__(name, bases, attrs)

        attrs = inspect.getmembers(
            cls, lambda a: hasattr(a, '_feagen_will_generate'))

        dag = DataDAG()
        # build DAG
        requirements = []
        for function_name, function in attrs:
            dag.add_node(function_name, function)
            del function._feagen_will_generate
            if hasattr(function, '_feagen_require'):
                requirements.append((function_name, function._feagen_require))
                del function._feagen_require
        dag.add_edges_from(requirements)
        cls._dag = dag


class FeatureGenerator(six.with_metaclass(FeatureGeneratorType, object)):

    def __init__(self, global_feature_hdf_path):
        self.intermediate_data = {}
        global_feature_hdf_dir = os.path.dirname(global_feature_hdf_path)
        if global_feature_hdf_dir != '':
            mkdir_p(global_feature_hdf_dir)
        self.global_feature_h5f = h5py.File(global_feature_hdf_path, 'a')

    def generate(self, data_keys):
        if isinstance(data_keys, str):
            data_keys = (data_keys,)
        node_keys_dict = self._dag.get_node_keys_dict(data_keys)
        involved_dag = self._dag.get_subgraph_with_ancestors(
            six.viewkeys(node_keys_dict))
        edges = []
        for source_node, data_keys in six.viewitems(node_keys_dict):
            edges.append((source_node, 'generate', {'keys': data_keys}))
        involved_dag.add_edges_from(edges)
        import ipdb; ipdb.set_trace()

        return involved_dag

    @classmethod
    def draw_dag(cls, path):
        cls._dag.draw(path)  # pylint: disable=protected-access
