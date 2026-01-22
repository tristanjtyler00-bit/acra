import os
import re
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import requests
import wget
from mergedeep import Strategy, merge
from pypdf import PdfReader
from rich import print
from tqdm.auto import tqdm

from acra.config import PAPER_SCRAPER_AGENT
from acra.paperscraper.prompts import (iterative_procedure_scraper_prompt,
                                       procedure_scraper_prompt,
                                       response_format,
                                       response_format_iterative,
                                       system_prompt)
from acra.utils import json_utils, prompt_utils
from acra.utils.prompt_utils import get_num_tokens


def write_to_log(data, filename="log.txt"):
    with open(filename, "a") as f:
        f.write(data + "\n")
    return None


class PaperScraper:
    """
    Agent to Download and extract text from a paper and then prompt the user to
    extract procedures from the paper. The urls are the links to the papers that
    the agent will download and extract text from. The first URL is always the
    paper that the agent will prompt the user to extract procedures from.
    """

    prompt_header = f"{20 * '-'}\nPROMPT\n{20 * '-'}\n\n"
    response_header = f"{20 * '-'}\nRESPONSE\n{20 * '-'}\n\n"

    def __init__(
        self,
        urls: list[str] = None,
        trial_dir: str = "trial_0", #same as XDL_memory.trial_dir
        log_file="log.txt",
        mode="paper",  # mode can be "paper" or "thesis" to change the  default chunk size
        path: str = PAPER_SCRAPER_AGENT.get("path", "../data/downloaded_papers/"),
        paper_memory: str = PAPER_SCRAPER_AGENT.get("paper_memory", "../data/memory/"), #same as XDL_memroy.storage_path
        model= PAPER_SCRAPER_AGENT.get("model", "gpt-4o"),
        temperature= PAPER_SCRAPER_AGENT.get("temperature", 0),
        timeout= PAPER_SCRAPER_AGENT.get("timeout", 150),
        chunk_size = PAPER_SCRAPER_AGENT.get("chunk_size", 3084),  # in tokens
        embedding_model= PAPER_SCRAPER_AGENT.get("embedding_model", "text-embedding-3-large"),
    ):
        """
        Args:
        urls: list of urls to main+SI to download and extract text from
                either local or online. Determined by "http" in the string
                Order should be main paper first, then SI
        path: path to save the downloaded papers
                only used if the paper is downloaded from a URL
        model: openai model to use for extraction
        temperature: temperature for the model
        timeout: timeout for the model
        chunk_size: size of the chunks to send to the model (in tokens)
        log_file: file to log the prompts and responses
        mode: "paper" or "thesis" to change the default chunk size
        Returns:
        None
        """

        assert urls is not None
        self.fname = []
        self.path = path
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        #self.chunk_size = chunk_size * 3  # 3 tokens per character ish
        self.chunk_size = chunk_size
        self.response_json = [None]
        self.log_file = log_file
        self.mode = mode
        self.embedding_model = embedding_model
        self.database = pd.DataFrame(columns=["embedding", "text"])
        self.paper_memory = Path(Path(paper_memory) / trial_dir) / "papers"

        for url in urls:
            # try to download the paper
            if "http" in url:
                self.fname.append(self.path + url.split("/")[-1])
                if not os.path.exists(self.fname[-1]):
                    try:
                        self.download_paper(url, path)
                    except Exception as e:
                        _ = e
                        self.download_paper_large(url)
            # local file
            else:
                self.fname.append(url)

        # convert to text
        self.main_text_len = 0
        self.text = []
        for _, fname in enumerate(self.fname):
            if len(self.text) == 0:
                self.text.append(self.pdf_to_text(fname))
                self.main_text_len = len(self.text)
            else:
                self.text.append(
                    self.pdf_to_text(fname, remove_references=False)
                )
        get_num_tokens(" ".join(self.text))
        self.text_chunks = self._text_chunker()
        self.embedding_chunks = self._text_chunker(chunk_size=2048)
        assert len(self.text_chunks) > 0, "No text to process"
        self._setup_client()
        self.embed_paper()

    def _setup_client(self, api_key=None):
        try:
            self.client = openai.OpenAI(
                # This is the default and can be somitted
                api_key=api_key
                if api_key
                else os.environ.get("CHAT_API_KEY"),
            )
        except Exception as e:
            raise e
        
    def _remove_whitespace(self, text):
        return re.sub(r"\s+", " ", text)

    def _text_chunker(self, chunk_size=None):
        # split text into chunks of self.chunk_size tokens
        # if self.mode == "thesis" or len(self.fname) > 1:
        # self.chunk_size = 6144
        # else:
        #     ...
        # print(f"Main text length: {self.main_text_len}")
        # first_chunk = self.text[:self.main_text_len]
        # print(f"First chunk: {get_num_tokens(first_chunk)}")

        # chunks = [first_chunk] + prompt_utils.split_string_with_limit(
        #     self.text[self.main_text_len:],
        #     limit=self.chunk_size,
        # )
        chunks = []
        for idx, text in enumerate(self.text):
            chunks += prompt_utils.split_string_with_limit(
                text,
                limit=self.chunk_size if chunk_size is None else chunk_size,
            )
        return chunks

    def download_paper(self, link: str):
        wget.download(link, self.path)
        return None

    def download_paper_large(self, url):
        """adapted from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests/16696317#16696317"""

        local_filename = url.split("/")[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(self.path + local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename

    def pdf_to_text(self, fname: str, remove_references=False):
        reader = PdfReader(fname)
        text = ""
        for page in tqdm(reader.pages):
            text += page.extract_text() + "\n"
        if remove_references:
            for i, line in enumerate(text.split("\n")):
                if "references" in line.lower():
                    text = "\n".join(text.split("\n")[:i])
                    break
        return text

    def get_num_tokens(self):
        return get_num_tokens(self.text)

    def extract_json(self, response):
        return json_utils.extract_json_from_response(response)

    def format_prompt(self, chunk, iterative=False, previous_response={}):
        if not iterative:
            return procedure_scraper_prompt.format(
                paper=chunk, response_format=response_format
            )
        else:
            return iterative_procedure_scraper_prompt.format(
                paper=chunk,
                response_format=response_format_iterative,
                previous_response=list(set(previous_response.get(
                    "procedure_titles", ["None extracted yet"]
                ))),
            )

    def prompt(
        self,
        chunk,
        iterative=False,
        previous_response=None,
        frequency_penalty=0,
    ):
        prompt = self.format_prompt(
            chunk, iterative=iterative, previous_response=previous_response
        )
        write_to_log(f"{self.prompt_header} {prompt}", self.log_file)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            frequency_penalty=frequency_penalty,
        )
        write_to_log(
            f"{self.response_header} {response.choices[0].message.content}\n",
            self.log_file,
        )
        return response.choices[0].message.content

    def parse_paper(self):
        iterative = False
        response = None
        for chunk in tqdm(self.text_chunks, desc="Parsing paper"):
            try:
                response = self.prompt(
                    chunk,
                    iterative=iterative,
                    previous_response=self.response_json[-1],
                )
                try:
                    response_json = self.extract_json(
                        response.replace("null", "None")
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    _ = e
                    response = self.prompt(
                        f"SYSTEM NOTE: ```previously you did not respond in the correct format read the following and respond in the noted format. Never repeat the same information twice```\n: {chunk}",
                        iterative=iterative,
                        previous_response=self.response_json[-1],
                    )
                    try:
                        response_json = self.extract_json(
                            response.replace("null", "None")
                        )
                    except Exception:
                        response_json = None
            except Exception as _:
                ...
                        
            if not iterative:
                if response_json is not None:
                    self.response_json.append(response_json)
                iterative = True
            else:
                if response_json is not None:
                    self.response_json.append(response_json)
            self.response_json = [None] + [self.combine_to_responses()]
        response = self.combine_to_responses()
        return response
    
    def combine_to_responses(self):
        # target = self.response_json[1]
        for resp in self.response_json[2:]:
            merge(self.response_json[1], resp, strategy=Strategy.ADDITIVE)
        return self.response_json[1]

    def embed_chunk(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            .data[0]
            .embedding
        )
        
    def embed_paper(self):
        for chunk in self.embedding_chunks:
            self.database = pd.concat(
                [
                    self.database,
                    pd.DataFrame(
                        [[np.array(self.embed_chunk(chunk)), chunk]],
                        columns=["embedding", "text"],
                    ),
                ],
                ignore_index=True,
            )
        self._store_vector_db()
        return None
    
    def _store_vector_db(self):
        if not os.path.exists(self.paper_memory):
            os.makedirs(self.paper_memory)
            os.makedirs(self.paper_memory / "0")
            storage_path = self.paper_memory / "0/"
            self.storage_path = f"{storage_path}/paper_embed.pkl"
        else:
            sub_dirs = [int(x) for x in os.listdir(self.paper_memory)]
            sub_dirs.sort()
            new_dir = sub_dirs[-1] + 1
            os.makedirs(self.paper_memory / str(new_dir))
            storage_path = self.paper_memory / f"{str(new_dir)}/"
            self.storage_path = f"{storage_path}/paper_embed.pkl"
        
        self.database.to_pickle(self.storage_path)
        return None
    
    
    