import inspect
from past.builtins import basestring

import six
import networkx as nx
from bistiming import SimpleTimer

from .dag import RegexDiGraph, draw_dag
from .bundling import DataBundlerMixin
from .data_handlers import (
    MemoryDataHandler,
    H5pyDataHandler,
    PandasHDFDataHandler,
    PickleDataHandler,
)


class FeatureGeneratorType(type):

    def __init__(cls, name, bases, attrs):  # noqa
        # pylint: disable=protected-access
        super(FeatureGeneratorType, cls).__init__(name, bases, attrs)

        attrs = inspect.getmembers(
            cls, lambda a: hasattr(a, '_feagen_will_generate'))

        dag = RegexDiGraph()
        # build the dynamic DAG
        handler_set = set()
        for function_name, function in attrs:
            node_attrs = function._feagen_will_generate
            del function.__dict__['_feagen_will_generate']
            handler_set.add(node_attrs['handler'])
            node_attrs['func_name'] = function_name
            if hasattr(function, '_feagen_require'):
                node_attrs['require'] = function._feagen_require
                del function.__dict__['_feagen_require']
            else:
                node_attrs['require'] = ()
            if node_attrs['mode'] == 'one':
                dag.add_node(function_name,
                             keys=node_attrs['keys'],
                             successor_keys=node_attrs['require'],
                             attr=node_attrs)
            elif node_attrs['mode'] == 'full':
                dag.add_node(function_name,
                             re_escape_keys=node_attrs['keys'],
                             successor_keys=node_attrs['require'],
                             attr=node_attrs,
                             mode='full')
            else:
                raise ValueError("Mode '%s' is not supported."
                                 % node_attrs['mode'])

        cls._dag = dag
        cls._handler_set = handler_set


def _run_function(function, handler_key, will_generate_keys, kwargs):
    with SimpleTimer("Generating {} {} using {}"
                     .format(handler_key, will_generate_keys,
                             function.__name__),
                     end_in_new_line=False):  # pylint: disable=C0330
        result_dict = function(**kwargs)
    if result_dict is None:
        result_dict = {}
    return result_dict


def _check_result_dict_type(result_dict, function_name):
    if not (hasattr(result_dict, 'keys')
            and hasattr(result_dict, '__getitem__')):
        raise ValueError("the return value of mehod {} should have "
                         "keys and __getitem__ methods".format(function_name))


class DataGenerator(six.with_metaclass(FeatureGeneratorType, DataBundlerMixin)):

    def __init__(self, handlers):
        handler_set = set(six.viewkeys(handlers))
        if handler_set != self._handler_set:
            redundant_handlers_set = handler_set - self._handler_set
            lacked_handlers_set = self._handler_set - handler_set
            raise ValueError('Handler set mismatch. {} redundant and {} lacked.'
                             .format(redundant_handlers_set,
                                     lacked_handlers_set))
        self._handlers = handlers

    def get(self, key):
        node_attr = self._dag.get_node_attr(key)
        handler = self._handlers[node_attr['handler']]
        data = handler.get(key)
        return data

    def _dag_prune_can_skip(self, nx_digraph, generation_order):
        for node in reversed(generation_order):
            node_attr = nx_digraph.node[node]
            handler = self._handlers[node_attr['handler']]
            node_attr['skipped'] = True
            for _, target, edge_attr in nx_digraph.out_edges_iter(node,
                                                                  data=True):
                if nx_digraph.node[target]['skipped']:
                    edge_attr['skipped_keys'] = edge_attr['keys']
                    edge_attr['nonskipped_keys'] = set()
                else:
                    required_keys = edge_attr['keys']
                    edge_attr['skipped_keys'] = set()
                    edge_attr['nonskipped_keys'] = set()
                    for required_key in required_keys:
                        if handler.can_skip(required_key):
                            edge_attr['skipped_keys'].add(required_key)
                        else:
                            edge_attr['nonskipped_keys'].add(required_key)
                    if len(edge_attr['nonskipped_keys']) > 0:
                        node_attr['skipped'] = False

    def build_involved_dag(self, data_keys):
        # get the nodes and edges that will be considered during the generation
        involved_dag = self._dag.build_directed_graph(data_keys,
                                                      root_node_key='generate')
        involved_dag.reverse(copy=False)
        generation_order = nx.topological_sort(involved_dag)[:-1]
        involved_dag.node['generate']['skipped'] = False
        self._dag_prune_can_skip(involved_dag, generation_order)
        return involved_dag, generation_order

    def draw_involved_dag(self, path, data_keys):
        involved_dag, _ = self.build_involved_dag(data_keys)
        draw_dag(involved_dag, path)

    def _get_upstream_data(self, dag, node):
        data = {}
        for source, _, edge_attr in dag.in_edges_iter(node, data=True):
            source_attr = dag.node[source]
            source_handler = self._handlers[source_attr['handler']]
            formatted_key_data = source_handler.get(edge_attr['keys'])
            # change the key to template
            data.update({template_key: formatted_key_data[key]
                         for template_key, key in six.viewitems(
                             edge_attr['template_keys'])})
        return data

    def _generate_one(self, dag, node, func_name, handler_key, handler_kwargs,
                      re_args, mode):
        if mode == 'one':
            will_generate_keys = (node,)
        elif mode == 'full':
            will_generate_keys = node
        else:
            raise ValueError("Mode '%s' is not supported." % mode)
        handler = self._handlers[handler_key]
        data = self._get_upstream_data(dag, node)
        function_kwargs = handler.get_function_kwargs(
            will_generate_keys=will_generate_keys,
            data=data,
            **handler_kwargs
        )
        if mode == 'one':
            function_kwargs['will_generate_key'] = node
        if len(re_args) > 0:
            function_kwargs['re_args'] = re_args
        function = getattr(self, func_name)
        result_dict = _run_function(function, handler_key, node,
                                    function_kwargs)
        _check_result_dict_type(result_dict, node)
        handler.check_result_dict_keys(result_dict, will_generate_keys, node,
                                       handler_key, **handler_kwargs)
        handler.write_data(result_dict)

    def generate(self, data_keys, dag_output_path=None):
        if isinstance(data_keys, basestring):
            data_keys = (data_keys,)
        involved_dag, generation_order = self.build_involved_dag(data_keys)
        if dag_output_path is not None:
            draw_dag(involved_dag, dag_output_path)

        # generate data
        for node in generation_order:
            node_attr = involved_dag.node[node]
            if node_attr['skipped']:
                continue
            self._generate_one(
                involved_dag, node, node_attr['func_name'],
                node_attr['handler'], node_attr['handler_kwargs'],
                node_attr['__re_args__'], node_attr['mode'])

        return involved_dag

    @classmethod
    def draw_dag(cls, path, data_keys):
        # pylint: disable=protected-access
        dag = cls._dag.draw(path, data_keys, root_node_key='generate',
                            reverse=True)
        if not nx.is_directed_acyclic_graph(dag):
            print("Warning! The graph is not acyclic!")


class FeatureGenerator(DataGenerator):

    def __init__(self, handlers=None, h5py_hdf_path=None, pandas_hdf_path=None,
                 pickle_dir=None):
        if handlers is None:
            handlers = {}
        if ('memory' in self._handler_set
                and 'memory' not in handlers):
            handlers['memory'] = MemoryDataHandler()
        if ('h5py' in self._handler_set
                and 'h5py' not in handlers):
            if h5py_hdf_path is None:
                raise ValueError("h5py_hdf_path should be specified "
                                 "when initiating FeatureGenerator.")
            handlers['h5py'] = H5pyDataHandler(h5py_hdf_path)
        if ('pandas_hdf' in self._handler_set
                and 'pandas_hdf' not in handlers):
            if pandas_hdf_path is None:
                raise ValueError("pandas_hdf_path should be specified "
                                 "when initiating FeatureGenerator.")
            handlers['pandas_hdf'] = PandasHDFDataHandler(pandas_hdf_path)
        if ('pickle' in self._handler_set
                and 'pickle' not in handlers):
            if pickle_dir is None:
                raise ValueError("pickle_dir should be specified "
                                 "when initiating FeatureGenerator.")
            handlers['pickle'] = PickleDataHandler(pickle_dir)
        super(FeatureGenerator, self).__init__(handlers)
