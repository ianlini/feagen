import os.path
import inspect
import re

import h5py
from mkdir_p import mkdir_p
import six
import networkx as nx


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


    def add_edges_from(self, requirements):
        edges = []
        for function_name, function_requirements in requirements:
            for requirement in function_requirements:
                edges.append((self._data_key_node_dict[requirement],
                              function_name))
        self._nx_dag.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(self._nx_dag):
            raise ValueError("The dependency graph has cycle.")

    def __getitem__(self, key):
        found_node = None
        found_key = None
        for regex_key, node in six.viewitems(self._data_node_dict):
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

    def __contains__(self, key):
        return key in self._data_node_dict


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

    def generate(self, feature_names):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        self.require(feature_names, self._feature_dag)

    def require(self, keys, dag):
        for key in keys:
            getattr(self, dag[key])(key)

    @classmethod
    def draw_dag(cls, path):
        # pylint: disable=protected-access
        agraph = nx.nx_agraph.to_agraph(cls._dag._nx_dag)
        agraph.layout('dot')
        agraph.draw(path)
