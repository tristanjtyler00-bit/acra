try:
    import ChemputerAPI
    from chempiler import Chempiler
    from chemputerxdl import graphgen
except ImportError:
    ...


def map_to_graph(xdl_proc, graph=None):
    ERROR_LOG = []
    if not graph:
        G, ERROR_LOG = graphgen.generator.graph_from_template(
            xdl_proc,
            save="graph.json",
            auto_fix_issues=True,
            fix_reactors=True,
        )
    else:
        G, ERROR_LOG = graphgen.generator.graph_from_template(
            xdl_proc,
            template=graph,
            save="graph.json",
            auto_fix_issues=True,
            fix_reactors=True,
        )
    return G, ERROR_LOG


def get_chempiler(output_dir, graph, simulation=True):
    platform_controller = Chempiler(
        "manual",
        "graph.json",
        output_dir=output_dir,
        device_modules=[ChemputerAPI],
        simulation=simulation,
        interactive=False,
    )
    return platform_controller