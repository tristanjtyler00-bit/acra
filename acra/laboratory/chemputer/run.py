import logging

logger = logging.getLogger(__name__)

_CHEMPUTER_IMPORT_ERROR = None

try:
    from xdl import XDL
    from chempiler import Chempiler
    from chemputerxdl.platform import ChemputerPlatform
    import ChemputerAPI
    import SerialLabware
    import commanduino
except ImportError as e:
    # The Chemputer stack is optional (see CHEMPU_AVAILABLE in acra/__init__.py
    # and main.py imports this module unconditionally), so importing this
    # module must not fail. But swallowing the error hides *why* the stack is
    # unusable and surfaces later as an unrelated NameError. Keep the real
    # error, log it now, and re-raise it from the functions that need it.
    _CHEMPUTER_IMPORT_ERROR = e
    logger.warning(
        "Chemputer stack failed to import: %s: %s", type(e).__name__, e, exc_info=True
    )

from .setup_chemputer import map_to_graph, get_chempiler
from acra.utils.logging import append_to_log


def _require_chemputer_stack():
    if _CHEMPUTER_IMPORT_ERROR is not None:
        raise ImportError(
            "The Chemputer stack could not be imported: "
            f"{type(_CHEMPUTER_IMPORT_ERROR).__name__}: {_CHEMPUTER_IMPORT_ERROR}"
        ) from _CHEMPUTER_IMPORT_ERROR


def disable_loggers():
    logging.getLogger("main_logger.vacuum_executioner_logger").disabled = True
    logging.getLogger("main_logger.vacuum_executioner_logger").disabled = True
    logging.getLogger('SerialLabware.connections.TCPIPConnection').disabled = True
    logging.getLogger('SerialLabware.controllers.CF41Chiller.chiller_filter').disabled = True
    logging.getLogger('chempiler_video').disabled = True
    logging.getLogger('commanduino.commandmanager.CommanduinoLabware').disabled = True
    logging.getLogger('SerialLabware.controllers.RCTDigitalHotplate.stirrer_reactor').disabled = True
    logging.getLogger('SerialLabware.controllers.CVC3000VacuumPump.vacuum_pump').disabled = True
    logging.getLogger('SerialLabware.controllers.HeiTorque100PrecisionStirrer.stirrer_separator').disabled = True
    logging.getLogger('SerialLabware.controllers.RCTDigitalHotplate.stirrer_reactor').disabled = True
    logging.getLogger('SerialLabware.controllers.RCTDigitalHotplate.stirrer_reactor').disabled = True

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    to_check = ["SerialLabware", "CommanduinoLabware", "commanduino", "Commanduino", "main_logger", "chempiler", "XDL", "chemputerxdl", "ChemputerAPI"]
    for logger in loggers:
        if any([name in logger.name for name in to_check]):
            logger.disabled = True
    return None


def run_experiment(xdl_proc, graph_file, output_dir=".", simulation=False):
    _require_chemputer_stack()
    x = XDL(xdl_proc, platform=ChemputerPlatform)
    graph = x.prepare_for_execution(graph_file, interactive=False)
    c = Chempiler(
        'manual',
        graph,
        output_dir=output_dir,
        simulation=simulation,
        device_modules=[ChemputerAPI]
    )
    x.execute(c)


def execute_in_simulation(
    xdl_proc,
    experiment="procedure",
    output_dir="log_files",
    graph="graph.json",
    simulation=True,
    interactive=False,
):
    # Outside the try below on purpose: that block returns errors as a list of
    # strings, which would turn a missing stack back into a quiet return value.
    _require_chemputer_stack()

    ERROR_LOG = []
    logging.basicConfig(level=logging.CRITICAL)
    try:
        xdl = XDL(xdl_proc, platform=ChemputerPlatform)
        _, ERROR_LOG = map_to_graph(xdl)
        platform = get_chempiler(
            output_dir=output_dir,
            graph=graph,
            simulation=simulation,
        )
        xdl.prepare_for_execution("graph.json", interactive=interactive)
        disable_loggers()
        xdl.execute(platform, interactive=interactive)
        return []
    
    except Exception as e:
        if isinstance(e.args[0], str):
            if "'XDL' object has no attribute 'steps'" in e.args[0]:
                return ["XDL object has no attribute 'steps'"] + ERROR_LOG
        elif isinstance(e.args[0], list):
            for error in e.args[0]:
                # remove those errors
                if "'XDL' object has no attribute 'steps'" in error:
                    e.args[0].remove(error)
        if isinstance(e.args[0], list):
            return e.args[0] + ERROR_LOG
        elif isinstance(e.args[0], str):
            return [e.args[0]] + ERROR_LOG
        return [e.args[0]] + ERROR_LOG


def check_if_experiment(xdl, log_file, platform_graph, experiment_output_dir):
    append_to_log(
        log_file,
        f"{'-'*40} Executing experiment {'-'*40}",
    )
    print("Make sure to have the Chemputer platform connected ...")
    
    # execute experiment upon user input
    execute = input("Do you want to execute the experiment? (y/n): ")
    if execute.lower() == "y":
        run_experiment(xdl, platform_graph, experiment_output_dir)
        return True
    else:
        append_to_log(
            log_file,
            "Experiment was not executed.",
        )
        return False