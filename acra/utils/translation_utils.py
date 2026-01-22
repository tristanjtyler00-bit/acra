from acra.utils.logging import append_to_log
from acra.agents.agents import Procedure_Agent, CritiqueAgent
from acra.utils.xdl_utils import xdl_to_human_readable

from acra.config import CRITIQUE_AGENT


def set_up_error_log():
    return {
        "succesfull": False,
        "syntactic_passed": False,
        "translatable_passed": False,
        "simulation_passed": False,
    }

def get_researcher_input(questions: list[str]):
    q_a_mapping = {}
    for line in questions:
        q_a_mapping[line] = input(line)
    return q_a_mapping


def error_enrichment(error):
    propery_limits = {
        "pressure": "Pressure should be given in mbar|bar|mmhg|atm|pa. If no exact value is given make a best guess i.e. 0 mbar. For evaporate steps, always use auto setting unless there is a good reason not to.",
        "volume": "Volume should be given in l|litres|ml|dl|ul|.",
        "amount": "Amount should be given in g|kg|mg|ug| or mol|mmol or equiv.",
        "temp": "Should be given in C|K|F. If no exact value is given make a best guess. i.e. 25 C.",
        "time": "days|h|min|s. If no exact value is given make a best guess. i.e. 1 h.",
    }

    if "XDLFailedPropLimitError" in error:
        for key in propery_limits.keys():
            if key in error:
                return error + " " + propery_limits[key]
    elif "KeyError" in error:
        return ""
    return error


def extract_step_by_step(response, log_file="log.txt"):
    try:
        step_by_step = response.split("step-by-step instructions")[1].split(
            "UCP"
        )[0]
    except IndexError:
        append_to_log(
            log_file,
            "No step-by-step procedure was generated. Or response was not in the expected format.\n",
        )
        step_by_step = "No step-by-step procedure was generated. Or response was not in the expected format."
    return step_by_step


def check_inconsistencies(
    xdl,
    xdl_proc,
    nlp_procedure,
    chemicals="No additional information provided.",
    log_file="log.txt",
    model=CRITIQUE_AGENT.get("model", "gpt-4o"),
    platform="chemputer",
):
    human_readable = xdl_to_human_readable(xdl)
    critique = CritiqueAgent(model=model, platform=platform)
    inconsistencies = critique.generate_suggestion(
        procedure=nlp_procedure,
        xdl_code=xdl_proc,
        human_readable=human_readable,
        chemicals=chemicals,
    )

    append_to_log(log_file, f"{40*'#'} Inconsistencies {40*'#'}")
    append_to_log(log_file, critique.prompt, verbose=False)
    append_to_log(log_file, critique.response, verbose=False)
    append_to_log(log_file, inconsistencies, verbose=False)

    return inconsistencies


def incoorporate_correction(xdl, xdl_proc, nlp_procedure, missing_steps):
    human_readable = xdl_to_human_readable(xdl)
    critique = CritiqueAgent()
    response = critique.include_suggestion_in_xdl(
        procedure=nlp_procedure,
        xdl_code=xdl_proc,
        human_readable=human_readable,
        missing_steps=missing_steps,
    )
    return response


def prepare_missing_steps(missing_steps):
    """
    Strip the missing steps and not executable steps from the missing_steps dict
    """
    missing = []
    for steps in missing_steps.get("missing_steps", []):
        missing.append(
            {
                "step": steps["thought"],
                "reasoning": steps["reasoning"],
                "suggestion": steps["suggestion"],
                "xdl to add/correct": steps["xdl_step"],
            }
        )

    return {"missing_steps": missing}


# classification of the procedure
def get_classification(
    procedure: Procedure_Agent, log_file: str = "log.txt"
) -> bool:
    classification = procedure.classification
    append_to_log(
        log_file, f"{'-'*40} Classification: {classification} {'-'*40}"
    )
    if classification == "C":
        append_to_log(log_file, "This procedure is a incomplete procedure.")
        return False
    elif classification == "B":
        append_to_log(log_file, "This procedure is a reaction blueprint.")
        append_to_log(
            log_file,
            "Generating XDL procedure, but chemical names will have to be assigned.",
        )
        return True
    else:
        append_to_log(log_file, f"{40 * '-'} Generating XDL. {40 * '-'}")
        return True