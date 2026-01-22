import os
import re
import shutil
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from rich import print
from scipy.spatial.distance import cosine

from acra.agents.prompts import ambiguity, procedure_prompt, steps, xdl_prompt, steps_OT
from acra.config import PROCEDURE_AGENT, XDL_AGENT
from acra.laboratory.chemicals import chemicals
from acra.utils import json_utils
from .base_agent import BaseAgent


steps_mapping = {
    "chemputer": steps.steps,
    "OT": steps_OT.steps,
}
from .base_agent import BaseAgent


class XDL_memory(BaseAgent):
    def __init__(
        self,
        trial_dir="trial_0",
        continuation: bool = False,
        prompt_template: str = xdl_prompt.xdl_prompt,
        iterative_prompt_template: str = xdl_prompt.iterative_xdl_prompt,
        data_path=XDL_AGENT.get("data_path", "../acra/data/"),
        storage_path=XDL_AGENT.get("storage_path", "../acra/data/memory/"),
        model=XDL_AGENT.get("model", "gpt-4o"),
        embedding_model=XDL_AGENT.get(
            "embedding_model", "text-embedding-3-large"
        ),
        k=XDL_AGENT.get("k", 5),
        temperature=XDL_AGENT.get("temperature", 0),
        timeout=XDL_AGENT.get("timeout", 150),
        use_memory=XDL_AGENT.get("use_memory", True),
        ambiguity_agent=None,
        relevant_amgibuities=None,
        platform="chemputer",
    ):
        """
        XDL Memory Agent

        Parameters
        ----------
        trial_dir: str
            Trial directory name. Will be used to store all the data
            Storage path will be storage_path + trial_dir
        continuation: bool
            If True, will load previous database
        prompt_template: str
            Prompt template for generating XDL
        iterative_prompt_template: str
            Prompt template for iterative XDL improvement
        api_key: str
            OpenAI API key
        data_path: str
            Path to the data folder (i.e. previous procedures)
        storage_path: str
            Path to the storage folder
        model: str
            OpenAI model to use
        embedding_model: str
            OpenAI embedding model to use
        k: int
            Number of similar procedures to retrieve
        temperature: int
            Temperature for OpenAI model
        timeout: int    
            Timeout for OpenAI model
        use_memory: bool
            If True, will use memory to store translated procedures and include previous procedures in the prompt
        ambiguity_agent: Ambiguity_Agent
            Ambiguity agent to retrieve previous ambiguities
            if None, will not retrieve previous ambiguities
        relevant_amgibuities: list
            List of relevant ambiguities
        platform: str
            Platform to use, default is "chemputer"
        """
        super().__init__(model, embedding_model, temperature, timeout, response_format=None)

        self.data_path = Path(data_path)
        self.storage_path = (
            Path(Path(storage_path) / trial_dir) / "XDL_procedures"
        )
        self.labbook_path = Path(Path(storage_path) / trial_dir) / "labbook"
        self.papers_path = Path(Path(storage_path) / trial_dir) / "papers"
        self._generate_folder_structure()
        self.folder_to_file = {
            "xdls": "xdl",
            "graphs": "json",
            "procedures": "txt",
            "reaction_smiles": "txt",
        }

        self.prompt_template = prompt_template
        self.iterative_prompt_template = iterative_prompt_template
        self.k = k
        self.temperature = temperature
        self.timeout = timeout
        self.responses = []
        self.database = None
        self.ambiguity_agent = ambiguity_agent
        self.relevant_amgibuities = relevant_amgibuities
        self.use_memory = use_memory
        self.examples = None

        if platform not in steps_mapping:
            platform = "chemputer"
            print("Platform not found, using default platform 'chemputer'")
        self.steps = steps_mapping[platform]

        if platform not in steps_mapping:
            platform = "chemputer"
            print("Platform not found, using default platform 'chemputer'")
        self.steps = steps_mapping[platform]

        if not continuation:
            self._generate_folder_structure()
            self.database = pd.DataFrame(columns=["embedding", "text"])
            self.database.to_pickle(
                f"{self.storage_path}/vectordb/vectordb.pkl"
            )
            if self.use_memory:
                self.embed_db()
            print(f"XDL database with {self.database.shape[0]} current procedures loaded")
        else:
            try:
                self.database = pd.read_pickle(
                    f"{self.storage_path}/vectordb/vectordb.pkl"
                )
                print(self.database.shape)
            except FileNotFoundError:
                self.database = pd.DataFrame(columns=["embedding", "text"])
                print("No database found")

    def _generate_folder_structure(self) -> None:
        """
        Generate Memory folder structure
        Will be
        .../memory/trial_x/
            .../XDL_procedures/
                .../xdls/
                .../procedures/
                .../reaction_smiles/
                .../graphs/
                .../vec/
                .../vectordb/
                    .../vectordb.csv
                    .../vectordb.pkl
            .../labbook/
                .../proc_x.txt
            .../papers/
        """

        try:
            os.makedirs(self.storage_path, exist_ok=True)
            os.makedirs(self.storage_path / "vectordb", exist_ok=True)
            os.makedirs(self.labbook_path, exist_ok=True)
            os.makedirs(self.papers_path, exist_ok=True)

            new_dirs = ["xdls", "procedures", "reaction_smiles", "graphs"]
            for dir in new_dirs:
                dst = self.storage_path / dir
                src = self.data_path / dir
                shutil.copytree(src, dst, dirs_exist_ok=True)

        except FileExistsError as e:
            warn(str(e))

    def _store_vector_db(self) -> None:
        self.database.to_csv(f"{self.storage_path}/vectordb/vectordb.csv")
        self.database.to_pickle(f"{self.storage_path}/vectordb/vectordb.pkl")
        return None

    def _load_db(self) -> None:
        self.database = pd.read_pickle(
            f"{self.storage_path}/vectordb/vectordb.pkl"
        )
        # self.database = self.database.drop(columns="Unnamed: 0")
        return None

    def embed_procedure(self, text):
        text = text.replace("\n", " ")
        return self.embed(text)

    def embed_db(self):
        data_path = self.storage_path / "procedures"
        for file in os.listdir(data_path):
            with open(f"{data_path}/{file}", "r") as f:
                text = f.read().strip()
            if not text:
                continue
            self.database = pd.concat(
                [
                    self.database,
                    pd.DataFrame(
                        [[np.array(self.embed_procedure(text)), file]],
                        columns=["embedding", "text"],
                    ),
                ],
                ignore_index=True,
            )
        self._store_vector_db()
        return None

    def add_procedure(self, title, text, xdl) -> None:
        embedding = self.embed_procedure(text)
        self.database = pd.concat(
            [
                self.database,
                pd.DataFrame(
                    [[embedding, title]],
                    columns=["embedding", "text"],
                ),
            ],
            ignore_index=True,
        )
        self._store_vector_db()
        for dir, content in zip(("xdls", "procedures"), (xdl, text)):
            with open(
                self.storage_path
                / Path(dir)
                / f"{title}.{self.folder_to_file[dir]}",
                "w",
            ) as f:
                f.write(content)
        return None

    def _column_to_np_array(self) -> np.array:
        return np.stack((self.database.embedding))

    def _cosine_similarity(self, query, k) -> None:
        self.database["distance"] = self.database.embedding.apply(
            lambda x: cosine(x, query)
        )

        indices = self.database.sort_values("distance").index[:k]
        return indices

    def retrieve_top_k(self, query, k=3):
        """
        Return list of retrieve_k_top most similar items,
        most similar is last.
        """
        k = min(self.database.shape[0], k)
        if k == 0:
            return []
        # print(f"Retrieving {k} most similar procedures")
        embedded_query = np.array(self.embed_procedure(query))
        indices = self._cosine_similarity(embedded_query, k)
        proced_id = self.database.iloc[indices].text.tolist()
        return proced_id

    def load_procd_id(self, proced_id):
        data_dirs = ["xdls", "procedures", "reaction_smiles", "graphs"]
        proced_id = proced_id.split(".")[0]
        data = {}
        for dir in data_dirs:
            if not os.path.exists(
                self.storage_path
                / Path(dir)
                / f"{proced_id}.{self.folder_to_file[dir]}"
            ):
                data[dir] = "Not available"
                continue

            with open(
                self.storage_path
                / Path(dir)
                / f"{proced_id}.{self.folder_to_file[dir]}",
                "r",
            ) as f:
                if dir == "graphs":
                    data[dir] = f.read()
                else:
                    data[dir] = f.read()
        return data

    def get_previous_amgibuities(self, procedure=None, k=2):
        # print("Retrieving ambiguities")
        if procedure is None:
            procedure = self.raw_procedure

        procedure_setences = re.split(
            r"\.\s", " ".join(procedure.splitlines()).strip()
        )
        relevant_amgibuities = []
        for sentence in procedure_setences:
            temp_ambiguity = self.ambiguity_agent.retrieve_top_k(sentence, k=2)
            for _, ambiguity in temp_ambiguity.iterrows():
                relevant_amgibuities.append(
                    ": ".join(ambiguity[["ambiguity", "resolution"]])
                )
        self.relevant_amgibuities = list(set(relevant_amgibuities))
        return self.relevant_amgibuities

    def format_prompt(
        self,
        new_procedure: str,
        iterative=False,
        generated_xdl=None,
        feedback=None,
        chemicals=None,
    ) -> str:
        if not iterative or self.examples is None:
            self.examples = ""
            closest_ids = self.retrieve_top_k(new_procedure, self.k)

            if not closest_ids:
                warn("No similar procedures found")
                self.examples = "No reactions performed yet"
            else:
                for j, idx in enumerate(closest_ids):
                    data = self.load_procd_id(idx)
                    self.examples += f"""EXAMPLE {j+1}\n
Procedure: {data['procedures']},\n
XDL: {data['xdls']},\n"""



        if iterative:
            self.prompt = self.iterative_prompt_template.format(
                steps_description=self.steps,
                ambiguities= ambiguity.common_ambiguities,
                examples=self.examples,
                new_procedure=new_procedure,
                old_xdl=generated_xdl,
                errors=feedback["errors"],
                error_mapping_format=xdl_prompt.error_mapping_format,
                chemicals=chemicals,
            )
        else:
            self.prompt = self.prompt_template.format(
                steps_description=self.steps,
                ambiguities=ambiguity.common_ambiguities,
                examples=self.examples,
                new_procedure=new_procedure,
                step_by_step_instructions=xdl_prompt.step_by_step_instructions,
                chemicals_response=xdl_prompt.chemicals,
                chemicals=chemicals,
            )
        return None

    def generate_xdl(
        self,
        new_procedure: str,
        iterative=False,
        generated_xdl=None,
        feedback=None,
        chemicals=None,
    ) -> tuple[str]:
        self.format_prompt(
            new_procedure=new_procedure,
            iterative=iterative,
            generated_xdl=generated_xdl,
            feedback=feedback,
            chemicals=chemicals,
        )

        response = self.chat(xdl_prompt.system_prompt
                    if not iterative
                    else xdl_prompt.iterative_system_prompt,
                    self.prompt
                    )
        
        self.responses.append(response)
        return response


class Procedure_Agent(BaseAgent):
    xmol_to_mol = {
        "mol": 1,
        "mmol": 1e-3,
        "umol": 1e-6,
        "μmol": 1e-6,
        "µmol": 1e-6,
        "nmol": 1e-9,
    }

    def __init__(
        self,
        procedure: str = None,
        model=PROCEDURE_AGENT.get("model", "gpt-4o"),
        embedding_model=PROCEDURE_AGENT.get(
            "embedding_model", "text-embedding-3-large"
        ),
        temperature=PROCEDURE_AGENT.get("temperature", 0),
        timeout=PROCEDURE_AGENT.get("timeout", 150),
        solvent_db=PROCEDURE_AGENT.get(
            "solvent_db", "../acra/data/solvent_db.xlsx"
        ),
        ambiguity_agent=None,
        paper_path=None,
        use_ambiguity=True,
    ):
        super().__init__(model, embedding_model, temperature, timeout)

        self.raw_procedure = procedure
        self.chemicals = []
        self.steps = []
        self.clean_procedure = ""
        self.procedure_prompt = (
            procedure_prompt.raw_procedure_prompt
            if True
            else procedure_prompt.raw_procedure_paper_prompt
        )
        self.clean_procedure_prompt = (
            procedure_prompt.clean_procedure_prompt
            if True
            else procedure_prompt.clean_procedure_prompt_paper
        )
        self.rxn_to_procedure_prompt = procedure_prompt.clean_procedure_prompt

        self.ambiguity_agent = ambiguity_agent
        self.relevant_amgibuities = None
        self.solvent_db = pd.read_excel(solvent_db)
        self.paper_path = paper_path
        self.retrieve_sections = []
        self.user_request = "No requests"
        self.forced_classification = None
        self.use_ambiguity = use_ambiguity

        if self.paper_path:
            self.database = pd.read_pickle(self.paper_path)


    def format_extraction_prompt(self):
        if self.relevant_amgibuities is None and self.use_ambiguity:
            self.get_previous_amgibuities()

        return self.procedure_prompt.format(
            procedure=self.raw_procedure,
            ambiguities=ambiguity.common_ambiguities_procedure,
            previous_ambiguities=self.relevant_amgibuities,
            response_format=procedure_prompt.response_format if not self.paper_path else procedure_prompt.response_format_paper,
        )

    def format_rxn_to_procedure_prompt(self, procedure_data: dict):
        """Format the prompt for the reaction to procedure task

        Args:
            procedure_data (dict): Expected keys: reaction_type, reagent_section, procedure_section

        Returns:
            str: prompt
        """
        return self.rxn_to_procedure_prompt.format(
            reaction_type=procedure_data.get("reaction_type", None),
            reagent_section=procedure_data.get("reagent_section", None),
            procedure_section=procedure_data.get("procedure_section", None),
            ambiguities=ambiguity.common_ambiguities_procedure,
            response_format=procedure_prompt.response_format,
        )

    def format_procedure_prompt(self, ambiguity_library):
        
        if not self.paper_path:
            return self.clean_procedure_prompt.format(
                procedure=self.raw_procedure,
                identified_chemicals=self.chemicals,
                identified_ambiguities=self.ambiguities,
                ambiguities=ambiguity.common_ambiguities_procedure,
                previous_ambiguities=self.relevant_amgibuities,
                ambiguity_library=ambiguity_library,
                user_request=self.user_request,
                response_format=procedure_prompt.response_format_clean_procedure,
            )
        else:
            return self.clean_procedure_prompt.format(
                procedure=self.raw_procedure,
                paperqa=self.retrieve_sections,
                identified_chemicals=self.chemicals,
                identified_ambiguities=self.ambiguities,
                ambiguities=ambiguity.common_ambiguities_procedure,
                previous_ambiguities=self.relevant_amgibuities,
                ambiguity_library=ambiguity_library,
                user_request=self.user_request,
                response_format=procedure_prompt.response_format_clean_procedure,
            )

    def _cosine_similarity(self, query, k) -> None:
        self.database["distance"] = self.database.embedding.apply(
            lambda x: cosine(x, query)
        )
        indices = self.database.sort_values("distance").index[:k]
        return indices

    def embed_procedure(self, text):
        text = text.replace("\n", " ")
        return self.embed(text)

    def get_paper_information(self):
        sections = {}
        for question in self.paper_questions:
            embedded_query = np.array(self.embed_procedure(question))
            indices = self._cosine_similarity(embedded_query, 1)
            section = self.database.iloc[indices].text
            sections[question] = section
        self.retrieve_sections = sections
        return None

    def get_previous_amgibuities(self, procedure=None, k=2):

        if procedure is None:
            procedure = self.raw_procedure

        procedure_setences = re.split(
            r"\.\s", " ".join(procedure.splitlines()).strip()
        )
        relevant_amgibuities = []
        for sentence in procedure_setences:
            temp_ambiguity = self.ambiguity_agent.retrieve_top_k(sentence, k=2)
            for index, ambiguity in temp_ambiguity.iterrows():
                relevant_amgibuities.append(
                    " ".join(
                        [
                            "AMBIGUITY:",
                            ambiguity["ambiguity"],
                            "RESOLUTION:",
                            ambiguity["resolution"],
                        ]
                    )
                )
        self.relevant_amgibuities = "\n ".join(list(set(relevant_amgibuities)))
        return self.relevant_amgibuities

    def parse_get_chemicals_response(self, response):
        response = response.replace("null", "None")
        try:
            response = json_utils.extract_json_from_response(response)
        except Exception as e:
            raise e
        self.response = response
        if "chemicals" in response:
            unique_chemicals = set([c[0] for c in response["chemicals"]])
            # only keep unique chemicals. Sometimes LLMs return duplicates
            for _, chemical in enumerate(unique_chemicals):
                self.chemicals.append(
                    [c for c in response["chemicals"] if c[0] == chemical][0]
                )

        else:
            raise ValueError("No chemicals were identified")
        if "ambiguities" in response:
            self.ambiguities = response["ambiguities"]
        else:
            raise ValueError("No ambiguities were identified")
        if "unresolved_ambiguities" in response:
            if len(response["unresolved_ambiguities"].keys()) == 0:
                self.unresolved_ambiguities = None
            else:
                self.unresolved_ambiguities = response[
                    "unresolved_ambiguities"
                ]
        else:
            self.unresolved_ambiguities = []

        if "paperqa" in response:
            self.paper_questions = response["paperqa"]
        else:
            self.paper_questions = []

        return response

    def parse_rxn_to_procedure_response(self, response):
        response = response.replace("null", "None")
        try:
            response = json_utils.extract_json_from_response(response)
        except Exception as e:
            raise e
        self.response = response
        if "chemicals" in response:
            unique_chemicals = set([c[0] for c in response["chemicals"]])
            # only keep unique chemicals. Sometimes LLMs return duplicates
            for _, chemical in enumerate(unique_chemicals):
                self.chemicals.append(
                    [c for c in response["chemicals"] if c[0] == chemical][0]
                )
        else:
            raise ValueError("No chemicals were identified")
        return response

    def get_solvent_properties(self, chemical_names):
        chemical_names = [c.lower() for c in chemical_names]

        if any(
            solvent in chemical_names
            for solvent in self.solvent_db["Compound"]
        ):
            solvent_properties = self.solvent_db[
                self.solvent_db["Compound"].isin(chemical_names)
            ]
            return solvent_properties
        return None

    def get_chemicals(self):
        prompt = self.format_extraction_prompt()
        self.get_chemicals_prompt = prompt
        response = self.prompt(prompt)
        response = self.parse_get_chemicals_response(response)

        if self.paper_path and self.paper_questions:
            self.get_paper_information()

        print("Getting Chemcial Information")
        for idx, chemical in enumerate(self.chemicals):
            self.get_checmical_information(chemical, idx)

        print("Chemical Information Retrieved")
        print(self.chemicals)
        return None
    
    def rxn_to_procedure(self, procedure_data):
        prompt = self.format_rxn_to_procedure_prompt(procedure_data)
        self.get_chemicals_prompt = prompt
        response = self.prompt(prompt)
        response = self.parse_rxn_to_procedure_response(response)
        print(f"Response: {response}")

        print("Getting Chemcial Information")
        for idx, chemical in enumerate(self.chemicals):
            self.get_checmical_information(chemical, idx)

        print("Chemical Information Retrieved")
        print(self.chemicals)
        return None
    
    def generate_clean_procedure(self, ambiguity_library=None):
        prompt = self.format_procedure_prompt(
            ambiguity_library=ambiguity_library
        )

        response = self.prompt(prompt, clean_procedure=True)
        self.response = response.replace("null", "None")
        try:
            self.response = json_utils.extract_json_from_response(
                self.response
            )
            if "classification" in self.response:
                self.classification = self.response["classification"].get(
                    "classification", "C"
                )
                if (
                    self.classification == "A"
                    and self.forced_classification == "B"
                ):
                    self.classification = "B"

                if self.classification not in ["A", "B", "C"]:
                    raise ValueError("Invalid classification")
            else:
                self.classification = None
            print("Classification", self.classification)

        except Exception as e:
            raise e
        self.clean_response = self.response

    def get_checmical_information(self, chemical, idx):
        if isinstance(chemical[-1], str):
            approx_search = (
                True
                if any(
                    s in chemical[-1].lower()
                    for s in ["mol", "mmol", "umol", "nmol", "eq", "ratio"]
                )
                else False
            )
        else:
            approx_search = False

        c = chemicals.Chemical(chemical[0], approx_search=approx_search)
        found = c.get_cid()
        if isinstance(chemical[-1], str):
            if not found and any(
                u in chemical[-1].lower()
                for u in ["mol", "mmol", "umol", "nmol", "eq", "ratio"]
            ):
                self.user_request = "Categroize this procedure as a Blueprint! Some information for the chemicals could not be found!"
                self.forced_classification = "B"
                return None
        elif not found:
            return None

        if c.cid:
            c.get_molecule_properties()
            if chemical[-1]:
                if any(
                    u in chemical[-1].lower()
                    for u in ["mol", "mmol", "umol", "nmol", "μmol", "µmol"]
                ):
                    conversion = self.xmol_to_mol.get(
                        chemical[-1].lower(), None
                    )
                    if conversion:
                        amount = float(chemical[1]) * conversion
                        amount_g = amount * float(c.molecular_weight)
                        self.chemicals[idx].extend([str(amount_g) + " g"])
            self.chemicals[idx].append(f"IUPAC name: {c.iupac_name}")
            try:
                solvent_properties = self.get_solvent_properties(
                    c.compound.synonyms
                )
                self.chemicals[idx].extend(
                    [
                        f"boiling point: {solvent_properties['bp'].values[0]} °C",
                        f"melting point: {solvent_properties['mp'].values[0]} °C",
                        f"density: {solvent_properties['density'].values[0]} g/mL",
                    ]
                )
            except Exception:
                ...

    def prompt(self, prompt, clean_procedure=False):
        response = self.chat(procedure_prompt.system_prompt
                    if not clean_procedure
                    else procedure_prompt.clean_system_prompt,
                    prompt
                    )
        return response
