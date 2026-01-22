import os
from pathlib import Path
from rich import print

from acra.agents.agents import AmbiguityAgent, Procedure_Agent, XDL_memory
from acra.paperscraper import PaperScraper
from acra.utils import labbook_utils
from acra.utils.logging import append_to_log

from acra.config import (
    AMBIGUIQTY_AGENT,
    PAPER_SCRAPER_AGENT,
    PROCEDURE_AGENT,
    RUN_CONFIG,
    XDL_AGENT,
)
from acra.laboratory.chemputer import execute_in_simulation, check_if_experiment
from acra.utils.xdl_utils import (
    xdl_feature_suggestions,
    extract_xdl,
    check_xdl,
)
from acra.utils.translation_utils import (
    check_inconsistencies,
    get_classification,
    incoorporate_correction,
    prepare_missing_steps,
    get_researcher_input,
    set_up_error_log,
)


# XDL generation
def write_xdl(
    xdl_agent: XDL_memory,
    nlp_procedure: str,
    filename="procedure.xdl",
    iterations=6,
    log_file="log.txt",
    graph="graph.json",
    chemicals="No additional information provided.",
    simulation=True,
    platform="chemputer",
) -> tuple[str, dict]:
    """Generate XDL from a procedure

    Args:
        xdl_agent (XDL_memory):
            XDL agent instance to generate XDL
        nlp_procedure (str):
            Procedure to translate
        filename (str, optional):
            File to store procedure to. Defaults to "procedure.xdl".
        iterations (int, optional):
            Number of iterations until XDL must be error free. Defaults to 6.
        log_file (str, optional):
            File to write logs to. Defaults to "log.txt".
        graph (STR, optional):
            path to graph file to use for simulation. Defaults to graph.json -> default template.
        chemicals (str, optional):
            Information about chemical extracted form pubchem... Defaults to "No additional information provided.".
        simulation (bool, optional):
            Whether to execute in simulation. Defaults to True.

    Returns:
        tuple[str, dict]: XDL procedure and error log
    """
    error_log = set_up_error_log()
    iterative = False
    final_round = False
    xdl_procedure = ""
    feedback = None
    inc_check = True  # check for inconsistencies only once

    for idx in range(iterations):
        error_log[idx] = {}
        append_to_log(log_file, f"{40 * '#'} Iteration {idx} {40 * '#'}")

        if not iterative:
            reply = xdl_agent.generate_xdl(
                new_procedure=nlp_procedure,
                chemicals=chemicals,
            )
        elif iterative:
            reply = xdl_agent.generate_xdl(
                new_procedure=nlp_procedure,
                iterative=iterative,
                generated_xdl=xdl_procedure,
                feedback=feedback,
                chemicals=chemicals,
            )

        xdl_procedure = extract_xdl(reply, log_file)
        append_to_log(
            log_file,
            f"XDL Agent prompt: {xdl_agent.prompt}\n\nXDL Agent Response: {reply}",
            verbose=False,
        )
        iterative = True
        if "<XDL>" in xdl_procedure:
            error_log[idx]["XDL generated"] = True

            with open(filename, "w", encoding="utf-8") as f:
                f.write(xdl_procedure)
            
            if os.environ["CHEMPU_AVAILABLE"] == "N":
                print("Returning XDL without iterative improvement since Chemputer is not available.")
                error_log[idx]["XDL generated"] = False
                return xdl_procedure, error_log

        else:
            error_log[idx]["XDL generated"] = False
            if os.environ["CHEMPU_AVAILABLE"] == "N":
                print("Returning XDL without iterative improvement since Chemputer is not available.")
                return xdl_procedure, error_log
            continue
        feedback, xdl = check_xdl(filename)
        append_to_log(log_file, xdl_procedure)
        append_to_log(log_file, feedback)
        error_log[idx]["Feedback"] = feedback
        error_log[idx]["Errors"] = len(feedback["errors"])

        # check for inconsistencies (e.g. missing steps, etc.)
        # can only be done once XDL passes validation.
        if not feedback["errors"] and inc_check:
            inc_check = False
            missing_steps = check_inconsistencies(
                xdl, xdl_procedure, nlp_procedure, log_file=log_file, platform=platform
            )
            error_log[idx]["judge"] = {"missing_steps": missing_steps}
            error_log["not_executable"] = missing_steps.get("not_executable", [])
            append_to_log(log_file, missing_steps)

            xdl_feature_suggestions(missing_steps.get("not_executable", []))
            if missing_steps:
                feedback["errors"] = missing_steps
                # reprompt to correct inconsistencies
                reply = incoorporate_correction(
                    xdl,
                    xdl_procedure,
                    nlp_procedure,
                    prepare_missing_steps(missing_steps),
                )

                xdl_procedure = extract_xdl(reply, log_file)
                if "<XDL>" in xdl_procedure:
                    error_log[idx]["judge"]["XDL generated"] = True
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(xdl_procedure)
                else:
                    error_log[idx]["judge"]["XDL generated"] = False
                    continue
            feedback, xdl = check_xdl(filename)
            feedback["not_executable"] = missing_steps.get("not_executable", [])
            if not feedback["errors"] or idx == iterations - 1:
                final_round = True
            append_to_log(
                log_file,
                f"{40 * '#'} Included inconcsistencies {40 * '#'} \n {xdl_procedure} \n\n {feedback}",
            )
            error_log[idx]["judge"]["Feedback"] = feedback
            error_log[idx]["Errors"] += len(feedback.get("errors", []))

        if not feedback["errors"] or final_round:
            error_log["xdl_after_judge"] = xdl_procedure
            break

    if feedback["errors"] or xdl is None:
        append_to_log(log_file, f"{40 * '#'} COULD NOT GENERATE VALID XDL {40 * '#'}")
        return xdl_procedure, error_log
    error_log["syntactic_passed"] = True
    error_log["translatable_passed"] = (
        True if not error_log.get("not_executable", []) else False
    )

    # check if XDL should be simulated
    if not simulation:
        if error_log["syntactic_passed"]:
            error_log["succesfull"] = True
        return xdl_procedure, error_log

    feedback = None
    append_to_log(log_file, f"{40 * '#'} Starting Simulation {40 * '#'}")
    for jdx in range(idx + 1, iterations + 1):
        error_log[jdx] = {}
        errors = execute_in_simulation(xdl_procedure, graph=graph)
        error_log[jdx]["Errors"] = len(errors)
        error_log[jdx]["Simulation errors"] = errors
        append_to_log(log_file, " ".join(errors) if errors else "No errors.")

        if len(errors) != 0:
            append_to_log(log_file, f"{40 * '#'} Iteration {jdx} {40 * '#'}")
            reply = xdl_agent.generate_xdl(
                new_procedure=nlp_procedure,
                iterative=iterative,
                generated_xdl=xdl_procedure,
                feedback={"errors": errors},
            )
            xdl_procedure = extract_xdl(reply, log_file)
            append_to_log(
                log_file,
                f"XDL Agent prompt: {xdl_agent.prompt}\n\nXDL Agent Response: {reply}",
                verbose=False,
            )
            append_to_log(log_file, f"XDL: {xdl_procedure}")
            if "<XDL>" in xdl_procedure:
                error_log[jdx]["XDL generated"] = True
                feedback, xdl = check_xdl(
                    xdl_procedure
                )  # TODO might need to use feedback
                error_log[jdx]["Feedback"] = feedback

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(xdl_procedure)
            elif jdx == iterations:
                error_log[jdx]["XDL generated"] = False
                error_log["succesfull"] = False
                break
            else:
                error_log[jdx]["XDL generated"] = False
                continue
        else:
            error_log[jdx]["XDL generated"] = True
            error_log["succesfull"] = True
            error_log["simulation_passed"] = True
            break
    return xdl_procedure, error_log


# Translate paper to XDL
def paper_to_xdl(
    paper_url: list[str],
    iterations: int = RUN_CONFIG.get("iterations", 6),
    filename: str = RUN_CONFIG.get("filename", "procedure.xdl"),
    expand_memory: bool = RUN_CONFIG.get("expand_memory", True),
    log_file: str = RUN_CONFIG.get("log_file", "log.txt"),
    continuation: bool = RUN_CONFIG.get("continuation", False),
    trial_dir: str = RUN_CONFIG.get("trial_dir", "trial_0"),
    ask_ambiguous: bool = RUN_CONFIG.get("ask_ambiguous", False),
    graph: str = RUN_CONFIG.get("graph", None),
    save_dir: str = RUN_CONFIG.get("save_dir", "../data/xdls/"),
    scraper_model=PAPER_SCRAPER_AGENT.get("model", "gpt-4o"),
    ambiguity_model=AMBIGUIQTY_AGENT.get("model", "gpt-4o"),
    xdl_model=XDL_AGENT.get("model", "gpt-4o"),
    procedure_model=PROCEDURE_AGENT.get("model", "gpt-4o"),
    generate_xdl=True,
    parse_only=False,
    paper_id: str = 0,
    use_ambiguity_agent=True,
    execute_experiment=False,
    platform_graph=None,
    experiment_output_dir=".",
    platform="chemputer",
) -> dict:
    """
    Entry point for translating a paper to XDL
    Given a paper URL, or the path to a paper, the function extracts the procedures from the paper
    and generates XDL procedures for each procedure in the paper.
    Multiple documents can be passed at once (i.e. paper and supplementary information).

    NOTE: All models should be OpenAI models

    Args:
    paper_url: list[str]
        URL to the paper(s) or path to the paper(s)
    iterations: int
        number of iterations to generate XDL
    filename: str
        name of the XDL file. Only used for temporary storage
    expand_memory: bool
        whether to store the XDL in the XDL memory to make it available for future use. Generated will be stored regardless
    log_file: str
        path to the log file
    continuation: bool
        whether to continue from the last trial. If False, the memory databases etc. will be embedded again
    trial_dir: str
        name of the trial directory where all the data is stored
    ask_ambiguous: bool
        whether to ask the user for input when ambiguities are found
    graph: str
        path to the graph template
    save_dir: str
        path were all the XDLs are stored. Independent of the memory database and save_dir in the XDL agent
    scraper_model: str
        model to use for the paper scraper
    ambiguity_model: str
        model to use for the ambiguity agent
    xdl_model: str
        model to use for the XDL agent
    procedure_model: str
        model to use for the procedure agent
    generate_xdl: bool
        whether to generate XDL. For benchmarking purposes, this can be set to False
    parse_only: bool
        whether to only parse the paper and return the response. For benchmarking purposes, this can be set to True
    paper_id: str
        ID of the paper. For unambiguous identification of the procedures
    use_ambiguity_agent: bool
        whether to use the ambiguity agent to resolve ambiguities
    execute_experiment: bool
        whether to execute the experiment on a real platform. Default is False
    platform_graph: str,
        path to the graph file for the Chemputer platform. For executing the experiment
    experiment_output_dir: str
        directory where the output of the experiment is stored
    platform: str
        platform to execute the experiment on. Default is chemputer

    Returns:
       dict: labbook containing the procedures, XDLs, and other information
    """
    log_file = (
        XDL_AGENT.get("storage_path", "./data/memory/") + f"{trial_dir}/" + log_file
    )

    print("Embedding Paper ...")
    paper_scraper = PaperScraper(
        urls=paper_url,
        log_file=log_file,
        model=scraper_model,
        trial_dir=trial_dir,
    )

    ambiguity_agent = AmbiguityAgent(model=ambiguity_model)
    xdl_agent = XDL_memory(
        trial_dir=trial_dir,
        continuation=continuation,
        ambiguity_agent=ambiguity_agent,
        model=xdl_model,
    )
    labbook = {}

    append_to_log(log_file, f"Paper URL: {paper_url}")
    append_to_log(log_file, f"{'-'*40} Extracting procedures from paper {'-'*40}")

    response = paper_scraper.parse_paper()

    append_to_log(log_file, f"{'-'*40} Saving knowledge graph {'-'*40}")
    labbook_utils.save_to_labbook(
        response,
        Path(paper_scraper.storage_path).parents[0] / "ps_response.json",
    )

    # labbook = labbook_utils.procedure_to_labbook(response)
    if not response.get("procedure_texts", []):
        append_to_log(log_file, "No procedures were found in the paper.\n")
        return None

    titles = response["procedure_texts"].keys()
    append_to_log(log_file, response, verbose=False)
    append_to_log(
        log_file,
        f"Extracted {len(titles)} procedures from the paper.\n",
    )

    if not titles:
        append_to_log(log_file, "No procedures were found in the paper.\n")
        return None
    if parse_only:
        return response

    # iterate over all procedures in the paper and generate XDL
    for title_idx, title in enumerate(titles):
        try:
            procedure = (
                response["procedure_texts"]
                .get(title, {})
                .get("full_procedure_text", "")
            )

            if not procedure:
                continue
            init_title = title

            title = f"{paper_id}_{title_idx}_" + "".join(
                x for x in procedure[:20] if x.isalnum()
            )
            if os.path.exists(xdl_agent.labbook_path / f"{title}.json"):
                continue

            labbook = {}
            labbook["procedure"] = procedure
            labbook[title] = {}
            labbook["title"] = init_title
            append_to_log(log_file, f"{'-'*40} Procedure: {title} {'-'*40}")
            append_to_log(
                log_file,
                f"{'-'*40}Generating XDL procedure for procedure {title} {'-'*40}",
            )
            append_to_log(log_file, procedure)

            procedure = procedure.replace("μ", "u")  # greek letter mu
            procedure = procedure.replace("µ", "u")  # micro symbol

            p = Procedure_Agent(
                procedure,
                ambiguity_agent=ambiguity_agent,
                model=procedure_model,
                paper_path=paper_scraper.storage_path,
                use_ambiguity=use_ambiguity_agent,
            )
            try:
                p.get_chemicals()
            except Exception:
                p.temperature = 0.05
                p.get_chemicals()
                p.temperature = 0.0

            labbook["chemicals"] = p.chemicals
            # resolve ambiguities
            unresolved_ambiguities = p.unresolved_ambiguities
            if unresolved_ambiguities and ask_ambiguous:
                append_to_log(
                    log_file,
                    f"Unresolved ambiguities: {unresolved_ambiguities}",
                )
                if isinstance(unresolved_ambiguities, dict):
                    ambiguity_mapping = ambiguity_agent.get_researcher_input(
                        unresolved_ambiguities
                    )
                    ambiguity_agent.embed_db(ambiguity_mapping)
                else:
                    ambiguity_mapping = None
            else:
                ambiguity_mapping = None
            labbook[title]["ambiguity_mapping"] = ambiguity_mapping

            # generate clean procedure with resolved ambiguities
            append_to_log(log_file, f"{'-'*40} Cleaned Literature procedure {'-'*40}")
            try:
                p.generate_clean_procedure(ambiguity_mapping)
            except Exception:
                p.temperature = 0.05
                p.generate_clean_procedure(ambiguity_mapping)
                p.temperature = 0.0

            # check if procedure is complete, might be moved to after generating clean procedure
            labbook["classification"] = p.classification
            if not get_classification(p, log_file=log_file):
                # save to labbook
                labbook_utils.save_to_labbook(
                    labbook,
                    xdl_agent.labbook_path / f"{title}.json",
                )
                continue

            append_to_log(log_file, p.clean_response["procedure"])
            clean_procedure = p.clean_response["procedure"]
            labbook[title]["clean_procedure"] = clean_procedure

            if generate_xdl:
                append_to_log(log_file, f"{'-'*40} Writing XDL procedure {'-'*40}")
                xdl, error_log = write_xdl(
                    xdl_agent,
                    clean_procedure,
                    filename=filename,
                    iterations=iterations,
                    log_file=log_file,
                    graph=graph,
                    simulation=True,
                    platform=platform,
                )
                labbook[title]["xdl"] = xdl
                labbook[title]["error_log"] = error_log
            else:
                error_log = {}
                xdl = None

            if (
                execute_experiment
                and xdl
                and platform_graph
                and error_log.get("succesfull", False)
            ):
                if check_if_experiment(
                    xdl, log_file, platform_graph, experiment_output_dir
                ):
                    labbook[title]["experiment_data"] = experiment_output_dir

            if expand_memory:
                # only correct XDL are stored, but labbook is updated with all procedures
                if error_log.get("succesfull", False):
                    append_to_log(
                        log_file,
                        f"\n\n{40 * '-'} Saving XDL {40 * '-'} \n\n {xdl}",
                        f"\n\n{40 * '-'} Saving XDL {40 * '-'} \n\n {xdl}",
                    )
                    xdl_agent.add_procedure(title, clean_procedure, xdl)
            append_to_log(log_file, f"\n\n{40 * '-'} Saving to labbook {40 * '-'}")
            labbook_utils.save_to_labbook(
                labbook,
                xdl_agent.labbook_path / f"{title}.json",
            )
            if save_dir and xdl:
                with open(
                    save_dir + f"{title}.xdl",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(xdl)
        except Exception as e:
            print("Error during XDL creation", e)
            ...
    return labbook


# Translate procedure to XDL
def procedure_to_xdl(
    procedure: str,
    trial_dir: str = RUN_CONFIG.get("trial_dir", "trial_0"),
    filename: str = RUN_CONFIG.get("filename", "procedure.xdl"),
    log_file: str = RUN_CONFIG.get("log_file", "log.txt"),
    iterations: int = RUN_CONFIG.get("iterations", 6),
    continuation: bool = RUN_CONFIG.get("continuation", False),
    expand_memory: bool = RUN_CONFIG.get("expand_memory", True),
    ask_ambiguous: bool = RUN_CONFIG.get("ask_ambiguous", False),
    graph=RUN_CONFIG.get("graph", None),
    save_dir: str = RUN_CONFIG.get("save_dir", "../data/xdls/"),
    ambiguity_model=AMBIGUIQTY_AGENT.get("model", "gpt-4o"),
    xdl_model=XDL_AGENT.get("model", "gpt-4o"),
    procedure_model=PROCEDURE_AGENT.get("model", "gpt-4o"),
    k_xdl: int = XDL_AGENT.get("k", 5),
    use_ambiguity_agent=True,
    execute_experiment=False,
    platform_graph=None,
    experiment_output_dir=".",
    platform="chemputer",
) -> tuple[str, dict]:
    """
    Entry point for translating a procedure to XDL

    NOTE: All models should be OpenAI models

    Args:
        procedure (str):
            procedure to translate
        trial_dir (str, optional):
            dir to store outputs to. Defaults to RUN_CONFIG.get("trial_dir", "trial_0").
        filename (str, optional):
            filename to write XDL procedure to. Defaults to RUN_CONFIG.get("filename", "procedure.xdl").
        log_file (str, optional):
            filename to write log file to. Defaults to RUN_CONFIG.get("log_file", "log.txt").
        iterations (int, optional):
            max number of iterations to generate error-free XDL. Defaults to RUN_CONFIG.get("iterations", 6).
        continuation (bool, optional):
            Whether to start from previous run (important for 'memory modules'). Defaults to RUN_CONFIG.get("continuation", False).
        expand_memory (bool, optional):
            Whether to save successful translations for future use. Defaults to RUN_CONFIG.get("expand_memory", True).
        ask_ambiguous (bool, optional):
            Whether to prompt for expert input. Defaults to RUN_CONFIG.get("ask_ambiguous", False).
        graph (_type_, optional):
            graph template for simulation of execution. Defaults to RUN_CONFIG.get("graph", None).
        save_dir (str, optional):
            directory to save run data to. Defaults to RUN_CONFIG.get("save_dir", "../data/xdls/").
        ambiguity_model (_type_, optional):
            model to use for ambiguity agent. Defaults to AMBIGUIQTY_AGENT.get("model", "gpt-4o").
        xdl_model (_type_, optional):
            model to use for XDL agent. Defaults to XDL_AGENT.get("model", "gpt-4o").
        procedure_model (_type_, optional):
            model to use to sanitize procedures. Defaults to PROCEDURE_AGENT.get("model", "gpt-4o").
        k_xdl (int, optional):
            number of previous XDL to include in translation. Defaults to XDL_AGENT.get("k", 5).
        use_ambiguity_agent (bool, optional):
            whether to include previous ambiguities. Defaults to True.
        execute_experiment (bool, optional):
            Attempt to execute the translated procedure on a connected platform. Defaults to False.
        platform_graph (_type_, optional):
            graph for the respective platform to execute on. Defaults to None.
        experiment_output_dir (str, optional):
            direcotry to save experiment data to. Defaults to ".".
        platform (str, optional):
            platform to execute the experiment on. Defaults to "chemputer".

    Returns:
       dict: labbook containing the procedures, XDLs, and other information
    """

    if os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(procedure):
        with open(procedure, "r", encoding="utf-8") as f:
            procedure = f.read()
    if not os.path.exists(
        XDL_AGENT.get("storage_path", "./data/memory/") + f"{trial_dir}/"
    ):
        os.makedirs(
            XDL_AGENT.get("storage_path", "./data/memory/") + f"{trial_dir}/",
            exist_ok=True,
        )
    log_file = (
        XDL_AGENT.get("storage_path", "./data/memory/") + f"{trial_dir}/" + log_file
    )

    ambiguity_agent = AmbiguityAgent(model=ambiguity_model)
    xdl_agent = XDL_memory(
        trial_dir=trial_dir,
        continuation=continuation,
        ambiguity_agent=ambiguity_agent,
        model=xdl_model,
        k=k_xdl,
        platform=platform,
    )

    procedure_ids = os.listdir(xdl_agent.labbook_path)
    if not procedure_ids:
        procedure_id = 0
    else:
        try:
            procedure_id = max([int(x.split("_")[0]) for x in procedure_ids]) + 1
        except Exception:
            procedure_id = 0

    xdl_agent.responses = []
    if xdl_agent.database.shape[0] == 0:
        xdl_agent.embed_db()

    labbook = {
        "procedure": procedure,
    }
    append_to_log(
        log_file,
        f"{40 * '-'} Generating XDL procedure {40 * '-'}\n{procedure}",
        f"{40 * '-'} Generating XDL procedure {40 * '-'}\n{procedure}",
    )

    procedure = procedure.replace("μ", "u")  # greek letter mu
    procedure = procedure.replace("µ", "u")  # micro symbol

    p = Procedure_Agent(
        procedure,
        ambiguity_agent=ambiguity_agent,
        model=procedure_model,
        use_ambiguity=use_ambiguity_agent,
    )

    try:
        p.get_chemicals()
    except Exception:
        p.temperature = 0.05
        p.get_chemicals()
        p.temperature = 0.0

    labbook["chemicals"] = p.chemicals

    # check for unresolved ambiguities
    unresolved_ambiguities = p.unresolved_ambiguities
    if unresolved_ambiguities and ask_ambiguous:
        ambiguity_mapping = get_researcher_input(unresolved_ambiguities)
    else:
        ambiguity_mapping = None

    # generate clean procedure with resolved ambiguities
    try:
        p.generate_clean_procedure(ambiguity_mapping)
    except Exception:
        p.temperature = 0.05
        p.generate_clean_procedure(ambiguity_mapping)
        p.temperature = 0.0

    clean_procedure = p.clean_response["procedure"]
    title = f"{procedure_id}_" + "".join(x for x in clean_procedure[:20] if x.isalnum())
    # check if procedure is complete, might be moved to after generating clean procedure

    labbook["classification"] = p.classification
    if not get_classification(p, log_file=log_file):
        # save to labbook
        labbook_utils.save_to_labbook(
            labbook,
            xdl_agent.labbook_path / f"{title}.json",
        )
        return None
    labbook["clean_procedure"] = clean_procedure

    append_to_log(
        log_file,
        f"\n\n{40 * '-'} Cleaned XDL procedure {40 * '-'}\n{clean_procedure}",
        f"\n\n{40 * '-'} Cleaned XDL procedure {40 * '-'}\n{clean_procedure}",
    )

    # write XDL procedure
    append_to_log(log_file, f"\n\n{40 * '-'} Writing XDL procedure {40 * '-'}")
    reply, error_log = write_xdl(
        xdl_agent,
        clean_procedure,
        filename=filename,
        iterations=iterations,
        log_file=log_file,
        graph=graph,
        chemicals=labbook.get("chemicals", "No additional information provided."),
        simulation=True,
        platform=platform,
    )
    append_to_log(log_file, f"LAST XDL: {reply}")
    labbook["xdl"] = reply
    labbook["error_log"] = error_log

    if execute_experiment and error_log.get("succesfull", False) and platform_graph:
        if check_if_experiment(reply, log_file, platform_graph, experiment_output_dir):
            labbook["experiment_data"] = experiment_output_dir

    if expand_memory:
        # only correct XDL are stored, but labbook is updated with all procedures
        if error_log["succesfull"]:
            append_to_log(log_file, f"\n\n{40 * '-'} Saving XDL {40 * '-'}")
            xdl_agent.add_procedure(title, clean_procedure, reply)
    append_to_log(log_file, f"\n\n{40 * '-'} Saving to labbook {40 * '-'}")
    print(xdl_agent.labbook_path / f"{title}.json")
    labbook_utils.save_to_labbook(
        labbook,
        xdl_agent.labbook_path / f"{title}.json",
    )
    if save_dir:
        with open(
            save_dir + f"{title}.xdl",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(reply)
    return reply, error_log
