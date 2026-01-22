import json
from pathlib import Path


def valid_procedures(paper_kg):
    valid_procedures = []
    for chunk in paper_kg.get("procedure_texts", {}):
        procd = paper_kg["procedure_texts"][chunk].get("new_procedure", "")
        if procd != "":
            valid_procedures.append(chunk)
    return valid_procedures


def get_procedure_text(paper_kg, procedure):
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            return paper_kg["procedure_texts"][chunk]["new_procedure"]


def get_purfication(paper_kg, procedure):
    techniques = {}
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            for technique in paper_kg["procedure_texts"][chunk].get(
                "purification", ""
            ):
                techniques[technique] = paper_kg["procedure_texts"][chunk][
                    "purification"
                ][technique]
    return techniques


def get_yield(paper_kg, procedure):
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            return paper_kg["procedure_texts"][chunk].get("yield", "")


def get_products(paper_kg, procedure):
    products = []
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            products = paper_kg["procedure_texts"][chunk].get("product", [])
    return products


def get_chemicals_and_info(paper_kg, procedure):
    chemicals = []
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            for chemical in set(
                paper_kg["procedure_texts"][chunk].get("chemicals", [])
            ):
                if paper_kg["chemicals"].get(chemical):
                    chemical_info = paper_kg["chemicals"][chemical].get(
                        "additional_info", []
                    )
                elif [
                    chemical_key
                    for chemical_key in paper_kg["chemicals"]
                    if chemical in paper_kg["chemicals"][chemical_key].keys()
                ]:
                    chemical_key = [
                        chemical_key
                        for chemical_key in paper_kg["chemicals"]
                        if chemical
                        in paper_kg["chemicals"][chemical_key].keys()
                    ][0]
                    chemical_info = paper_kg["chemicals"][chemical_key].get(
                        "additional_info", []
                    )
                else:
                    chemical_info = [""]
                chemicals.append(
                    chemical + "\n" + " ".join(chemical_info)
                    if isinstance(chemical_info, list)
                    else chemical + "\n" + chemical_info
                )
    return chemicals


def get_characterization_techniques(paper_kg, procedure):
    techniques = {}
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            for technique in paper_kg["procedure_texts"][chunk].get("analysis", {}):
                techniques[technique] = paper_kg["procedure_texts"][chunk][
                    "analysis"
                ][technique]
    return techniques


def get_additional_info(paper_kg, procedure):
    additional_info = []
    for chunk in paper_kg.get("procedure_texts", {}):
        if chunk == procedure:
            for info in paper_kg["procedure_texts"][chunk].get("additional_info", []):
                additional_info.append(info)
    return additional_info


def procedure_to_labbook(paper_kg):
    procedures = valid_procedures(paper_kg)
    labbook = {}

    for procedure in procedures:
        procedure_text = get_procedure_text(paper_kg, procedure)
        products = get_products(paper_kg, procedure)
        chemicals = get_chemicals_and_info(paper_kg, procedure)
        techniques = get_characterization_techniques(paper_kg, procedure)
        additional_info = get_additional_info(paper_kg, procedure)
        purfication = get_purfication(paper_kg, procedure)
        yield_ = get_yield(paper_kg, procedure)  # yield is a reserved keyword
        procedure_titles = paper_kg["procedure_titles"]

        labbook[procedure] = {
            "procedure_text": procedure_text,
            "products": products,
            "chemicals": chemicals,
            "purification": purfication,
            "yield": yield_,
            "characterization": techniques,
            "additional_info": additional_info,
            "procedure_titles": procedure_titles,
        }
    return labbook


def save_to_labbook(data, path):
    if not Path(path).exists():
        with open(path, "w") as f:
            f.write(json.dumps(data))
    else:
        # load and update
        try:
            with open(path, "r") as f:
                labbook = json.load(f)
                labbook.update(data)
                data = labbook
        except Exception:
            ...
        

        with open(path, "w") as f:
            f.write(json.dumps(data))
