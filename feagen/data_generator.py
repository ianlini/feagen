import inspect
from past.builtins import basestring

import six
import networkx as nx
from bistiming import SimpleTimer

from .dag import DataDAG
from .bundling import DataBundlerMixin
from .data_handlers import (
    MemoryDataHandler,
    H5pyDataHandler,
)


class FeatureGeneratorType(type):

    def __init__(cls, name, bases, attrs):
        # pylint: disable=protected-access
        super(FeatureGeneratorType, cls).__init__(name, bases, attrs)

        attrs = inspect.getmembers(
            cls, lambda a: hasattr(a, '_feagen_will_generate'))

        dag = DataDAG()
        # build DAG
        requirements = []
        handler_set = set()
        for function_name, function in attrs:
            dag.add_node(function_name, function)
            handler_set.add(function._feagen_will_generate['handler'])
            del function.__dict__['_feagen_will_generate']
            if hasattr(function, '_feagen_require'):
                requirements.append((function_name, function._feagen_require))
                del function.__dict__['_feagen_require']
        dag.add_edges_from(requirements)
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
        data = handler.get(key)[key]
        return data

    def _get_upstream_data(self, dag, node):
        data = {}
        for source, _, attr in dag.in_edges_iter(node, data=True):
            source_handler = self._handlers[dag.node[source]['handler']]
            data.update(source_handler.get(attr['keys']))
        return data

    def _generate_one(self, dag, node, handler_key, will_generate_keys, mode,
                      handler_kwargs):
        if mode == 'one':
            will_generate_key = will_generate_keys
            will_generate_keys = (will_generate_key,)
        handler = self._handlers[handler_key]
        data = self._get_upstream_data(dag, node)
        function_kwargs = handler.get_function_kwargs(
            will_generate_keys=will_generate_keys,
            data=data,
            **handler_kwargs
        )
        if mode == 'one':
            function_kwargs['will_generate_key'] = will_generate_key
        function = getattr(self, node)
        result_dict = _run_function(function, handler_key, will_generate_keys,
                                    function_kwargs)
        _check_result_dict_type(result_dict, node)
        handler.check_result_dict_keys(result_dict, will_generate_keys, node,
                                       handler_key, **handler_kwargs)
        handler.write_data(result_dict)

    def _dag_prune_can_skip(self, involved_dag, generation_order):
        for node in reversed(generation_order):
            node_attr = involved_dag.node[node]
            handler = self._handlers[node_attr['handler']]
            can_skip_node = True
            for _, target, edge_attr in involved_dag.out_edges_iter(node,
                                                                    data=True):
                if (target != 'generate'
                        and involved_dag.node[target]['skipped']):
                    edge_attr['skipped_keys'] = edge_attr['keys']
                    edge_attr['keys'] = []
                else:
                    required_keys = edge_attr['keys']
                    edge_attr['skipped_keys'] = []
                    edge_attr['keys'] = []
                    for required_key in required_keys:
                        if handler.can_skip(required_key):
                            edge_attr['skipped_keys'].append(required_key)
                        else:
                            edge_attr['keys'].append(required_key)
                    if len(edge_attr['keys']) > 0:
                        can_skip_node = False
            node_attr['skipped'] = True if can_skip_node else False

    def generate(self, data_keys):
        if isinstance(data_keys, basestring):
            data_keys = (data_keys,)

        # get the nodes ad edges that should be considered during the generation
        node_keys_dict = self._dag.get_node_keys_dict(data_keys)
        involved_dag = self._dag.get_subgraph_with_ancestors(
            six.viewkeys(node_keys_dict))
        edges = []
        for source_node, data_keys in six.viewitems(node_keys_dict):
            edges.append((source_node, 'generate',
                          {'keys': list(set(data_keys))}))
        involved_dag.add_edges_from(edges)
        generation_order = nx.topological_sort(involved_dag)[:-1]
        self._dag_prune_can_skip(involved_dag, generation_order)

        # generate data
        for node in generation_order:
            node_attr = involved_dag.node[node]
            if node_attr['skipped']:
                continue
            mode = node_attr['mode']
            if mode == 'full':
                self._generate_one(
                    involved_dag, node, node_attr['handler'],
                    node_attr['keys'], mode, node_attr['handler_kwargs'])
            elif mode == 'one':
                will_generate_key_set = set()
                for _, _, attr in involved_dag.out_edges_iter(node, data=True):
                    will_generate_key_set |= set(attr['keys'])
                for data_key in will_generate_key_set:
                    self._generate_one(
                        involved_dag, node, node_attr['handler'],
                        data_key, mode, node_attr['handler_kwargs'])
            else:
                raise ValueError("Mode '%s' is not supported." % mode)

        return involved_dag

    @classmethod
    def draw_dag(cls, path):
        cls._dag.draw(path)  # pylint: disable=protected-access


class FeatureGenerator(DataGenerator):

    def __init__(self, global_data_hdf_path=None, handlers=None):
        if handlers is None:
            handlers = {}
        if ('memory' in self._handler_set
                and 'memory' not in handlers):
            handlers['memory'] = MemoryDataHandler()
        if ('h5py' in self._handler_set
                and 'h5py' not in handlers):
            if global_data_hdf_path is None:
                raise ValueError("global_data_hdf_path should be specified "
                                 "when initiating FeatureGenerator.")
            handlers['h5py'] = H5pyDataHandler(global_data_hdf_path)
        super(FeatureGenerator, self).__init__(handlers)
