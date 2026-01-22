import os
import pandas as pd
import regex as re

try:
    from xdl import XDL
    from chemputerxdl.platform import ChemputerPlatform
except ImportError:
    os.environ["CHEMPU_AVAILABLE"] = "False"


from acra.utils.logging import append_to_log


def xdl_feature_suggestions(
    not_executable: list[dict],
    storage_path: str = "../data/xdl_suggestions/xdl_features.csv",
) -> None:
    if os.path.exists(storage_path):
        df = pd.read_csv(storage_path)
        df = df.drop(columns=["Unnamed: 0"])

    else:
        df = pd.DataFrame(
            columns=["thought", "interpretation", "reasoning", "suggestion"]
        )
    df = pd.concat([df, pd.DataFrame(not_executable)], ignore_index=True)
    df.to_csv(storage_path)
    return None


def extract_xdl(response, log_file="log.txt"):
    try:
        if "<XDL>" in response:
            xdl_procedure = (
                "<XDL>"
                + response.split("<XDL>")[1].split("</XDL>")[0]
                + "</XDL>"
            )
        elif "<Synthesis>" in response:
            xdl_procedure = (
                "\n<Synthesis>"
                + response.split("<Synthesis>")[1].split("</Synthesis>")[0]
                + "</Synthesis>\n</XDL>"
            )
        else:
            raise IndexError

        # typical replacements that cause trouble in XDL, but are not documented
        xdl_procedure = xdl_procedure.replace("μ", "u")  # greek letter mu
        xdl_procedure = xdl_procedure.replace("µ", "u")  # micro symbol
        xdl_procedure = xdl_procedure.replace('"<', '"')
        xdl_procedure = xdl_procedure.replace('>"', '"')
        xdl_procedure = xdl_procedure.replace('">', '"')
        xdl_procedure = xdl_procedure.replace('<"', '"')

    except IndexError:
        append_to_log(
            log_file,
            "No XDL procedure was generated. Or response was not in the expected format.\n",
        )
        if "<ERROR>" in response:
            # error = response.split("<ERROR>")[1].split("</ERROR>")[0]
            xdl_procedure = "ERROR: No procedure generated. Please try again. Make to strictly follow the required format."
        else:
            # error = "No XDL procedure was generated. Or response was not in the expected format."
            xdl_procedure = "ERROR: No procedure generated. Please try again. Make to strictly follow the required format."
    return post_process_xdl(xdl_procedure)


def post_process_xdl(xdl_proc):
    # repeat tag is not closed
    xdl_proc = re.sub(
        r'<Repeat repeats="([0-9]+)"\s*>?',
        r'<Repeat repeats="\1">\n',
        xdl_proc,
    )

    # remove comments
    xdl_proc = re.sub(r"<!--.*?-->", "", xdl_proc, flags=re.DOTALL)
    xdl_proc = [x for x in xdl_proc.split("\n") if x.strip()]
    return "\n".join(xdl_proc)


def check_xdl(xdl_proc):
    """
    Check for errors in the XDL code

    Args:
    xdl_proc: str, XDL code

    Returns:
    feedback: dict, containing warnings and errors
    xdl: XDL object, if no errors
    """
    warnings_list = []
    errors_list = []
    xdl = None
    try:
        xdl = XDL(xdl_proc, platform=ChemputerPlatform)
    except Exception as e:
        errors_list = e.args[0]
        if isinstance(errors_list, str):
            errors_list = [errors_list]
        # print(errors_list)
        errors_list = [
            error
            for error in errors_list
            if "'XDL' object has no attribute" not in error
        ]
        if len(errors_list) == 0:
            errors_list = [
                """Unknown error. Try again and make sure to strickly follow the
                required format. And ONLY use provided steps and attributes!
                Make sure to enclose all attributes in quotes, and close all
                tags."""
            ]

    finally:
        ...

    feedback = {
        "warnings": "\n".join(warnings_list) if warnings_list else [],
        "errors": errors_list,
    }
    return feedback, xdl


def xdl_to_human_readable(xdl):
    human_readable = xdl.human_readable()
    return human_readable