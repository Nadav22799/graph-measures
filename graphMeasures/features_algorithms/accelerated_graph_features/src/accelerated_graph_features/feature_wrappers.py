from .feature_wrapper_decorator import FeatureWrapper


@FeatureWrapper
def example_feature(graph, **kargs):
    # example_feature is a C++ function exposed to python
    from . import _features as feat

    print(graph['indices'])
    print(graph['neighbors'])

    # Here 0 is the default value for the argument
    example_arg = kargs.get('example_arg', 0)
    res = feat.example_feature(graph)
    # Any post-processing goes here
    return res


@FeatureWrapper
def clustering_coefficient(graph, **kwargs):
    from . import _features as feat

    res = feat.clustering_coefficient(graph)

    return res


@FeatureWrapper
def k_core(graph, **kwargs):
    from . import _features as feat

    res = feat.k_core(graph)

    return res


@FeatureWrapper
def node_page_rank(graph, **kwargs):
    from . import _features as feat

    dumping = kwargs.get('dumping', 0.85)
    max_iter = kwargs.get('max_iters', 100)

    res = feat.node_page_rank(graph, dumping, max_iter)

    return res


@FeatureWrapper
def bfs_moments(graph, **kwargs):
    from . import _features as feat

    res = feat.bfs_moments(graph)

    return res


@FeatureWrapper
def attraction_basin(graph, **kwargs):
    from . import _features as feat

    alpha = kwargs.get('alpha', 2)
    res = feat.attraction_basin(graph, alpha)
    for i, x in enumerate(res):
        if x < 0:
            res[i] = float('nan')
    return res


@FeatureWrapper
def flow(graph, **kwargs):
    from . import _features as feat

    t = kwargs.get('threshold', 0)
    res = feat.flow(graph, t)

    return res

@FeatureWrapper
def motif(graph, **kwargs):
    from . import _features as feat

    try:
        level = kwargs['level']
    except KeyError:
        raise AttributeError('Level must be specified!')
    try:
        edges = kwargs['edges']
    except KeyError:
        raise AttributeError('edges must be specified!')
    

    gpu = kwargs.get('gpu', False)
    device = kwargs.get('cudaDevice', 2)
    if not gpu:
        res = feat.motif(graph, level)
    else:
        res = feat.motif_gpu(graph, level, device, edges)
    return res
