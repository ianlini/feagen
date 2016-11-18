import os.path
import inspect
import re

import h5py
from mkdir_p import mkdir_p
import six


class RegexDAG(object):
    # TODO: This is not really a DAG

    def __init__(self):
        self._data_node_dict = {}

    def add(self, key, value, upstream=None):  # pylint: disable=unused-argument
        # TODO: set DAG
        self._data_node_dict[key] = value

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

    def __new__(mcs, clsname, bases, dct):
        # pylint: disable=protected-access
        cls = super(FeatureGeneratorType, mcs).__new__(
            mcs, clsname, bases, dct)
        attrs = inspect.getmembers(
            cls, lambda a: hasattr(a, '_feagen_data_type'))

        dag_dict = {
            'features': RegexDAG(),
            'intermediate_data': RegexDAG(),
        }

        # register the data key
        for attr_key, attr_val in attrs:
            set_name = attr_val._feagen_data_type
            dag = dag_dict[set_name]
            for key in attr_val._feagen_will_generate_keys:
                if key in dag:
                    raise ValueError("duplicated {} '{}' in {} and {}".format(
                        set_name, key, dag._data_node_dict[key], attr_key))
                dag.add(key, attr_key)

        cls._feature_dag = dag_dict['features']
        cls._intermediate_data_dag = dag_dict['intermediate_data']

        return cls


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
