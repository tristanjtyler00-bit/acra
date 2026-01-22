import os
from rich import print
from rich import print
import random
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from acra.agents.prompts import ambiguity_prompt
from acra.config import AMBIGUIQTY_AGENT
from acra.utils import json_utils
from .base_agent import BaseAgent


class AmbiguityDB(BaseAgent):
    def __init__(
        self,
        storage_path="../acra/data/ambiguity/",
        continuation: bool = True,
        temperature=AMBIGUIQTY_AGENT.get("temperature", 0),
        timeout=AMBIGUIQTY_AGENT.get("timeout", 150),
        embedding_model=AMBIGUIQTY_AGENT.get("embedding_model", "text-embedding-3-large"),
        model=AMBIGUIQTY_AGENT.get("model", "gpt-4o"),
        k=3,
        priming_prompt=ambiguity_prompt.ambiguity_prompt,
    ):
        super().__init__(model, embedding_model, temperature, timeout)

        self.storage_path = Path(storage_path)
        self._generate_folder_structure()
        self.k = k
        self.database = None
        self.priming_prompt = priming_prompt

        if not continuation:
            self._generate_folder_structure()
            self.database = pd.DataFrame(
                columns=["embedding", "ambiguity", "resolution"]
            )
            if not os.path.exists(f"{self.storage_path}/vectordb/vectordb.pkl"):
                self.database.to_pickle(
                    f"{self.storage_path}/vectordb/vectordb.pkl"
                )
        else:
            try:
                self.database = pd.read_pickle(
                    f"{self.storage_path}/vectordb/vectordb.pkl"
                )
            except FileNotFoundError as e:
                raise e

    def _generate_folder_structure(self) -> None:
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            os.makedirs(self.storage_path / "vectordb", exist_ok=True)
        except FileExistsError as e:
            warn(str(e))

    def _store_vector_db(self) -> None:
        self.database[["ambiguity", "resolution"]].to_csv(f"{self.storage_path}/vectordb/vectordb.csv")
        # store as pkl
        self.database[["embedding", "ambiguity", "resolution"]].to_pickle(f"{self.storage_path}/vectordb/vectordb.pkl")
        return None

    def _load_db(self) -> None:
        self.database = pd.read_pickle(
            f"{self.storage_path}/vectordb/vectordb.pkl"
        )
        # self.database = self.database.drop(columns="Unnamed: 0")
        return None

    def get_researcher_input(self, questions: dict):
        q_a_mapping = {}
        for key, qs in questions.items():
            for q in qs["questions"]:
                q_a_mapping[key + ": " + q] = input(key + ": " + q)
        return q_a_mapping

    def embed_procedure(self, text):
        text = text.replace("\n", " ")
        return self.embed(text)

    def embed_db(self, q_and_a):
        for question, answer in q_and_a.items():
            if not answer:
                continue
            self.database = pd.concat(
                [
                    self.database,
                    pd.DataFrame(
                        [[np.array(self.embed_procedure(question)), question, answer]],
                        columns=["embedding", "ambiguity", "resolution"],
                    ),
                ],
                ignore_index=True,
            )
        self._store_vector_db()

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
        embedded_query = np.array(self.embed_procedure(query))
        indices = self._cosine_similarity(embedded_query, k)
        proced_id = self.database.iloc[indices]
        return proced_id

    def format_priming_prompt(self, procedure):
        return self.priming_prompt.format(
            procedure=procedure, response_format=ambiguity_prompt.response_format
        )

    def get_questions(self, prompt):
        return self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": ambiguity_prompt.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        ).choices[0].message.content

    def prime_db(self, k: int | None = None):
        # load 10 procedures
        if k is None:
            k = self.k

        procedures = random.choices(
            os.listdir(self.storage_path.parent / "priming_procedures"), k=k
        )

        for procedure in procedures:
            with open(
                self.storage_path.parent / "priming_procedures" / procedure, "r"
            ) as f:
                text = f.read().strip()

            if not text:
                continue

            prompt = self.format_priming_prompt(text)
            print(text)
            print(text)

            # ask questions sentence for sentence and get reply from humans
            self.response = self.get_questions(prompt)
            questions = json_utils.extract_json_from_response(self.response)
            q_and_a = self.get_researcher_input(questions)
            self.embed_db(q_and_a)
