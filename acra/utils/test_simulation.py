import logging

try:
    from xdl import XDL
    import ChemputerAPI
    from chempiler import Chempiler
    from chemputerxdl import graphgen
    from chemputerxdl.platform import ChemputerPlatform
except ImportError:
    ...

def run_xdl_simulation(xdl_test):
    logging.basicConfig(level=logging.ERROR)
    x = XDL(xdl_test, platform=ChemputerPlatform)
    G, _ = graphgen.graph_from_template(x, auto_fix_issues=True, fix_reactors=True)
    x.prepare_for_execution("temp_template.json", interactive=False)
    
    c = Chempiler(
        'manual',
        "temp_template.json",
        output_dir='.',
        simulation=True,
        device_modules=[ChemputerAPI],
        interactive=False
    )
    x.execute(c, interactive=False)