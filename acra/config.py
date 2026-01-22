from pathlib import Path


ROOT_PATH = Path(__file__).parent.resolve()
REPO_PATH = Path(__file__).parent.parent.resolve()
CURR_PATH = Path.cwd().resolve()


# PATHS
MEMORY_PATH = f"{REPO_PATH}/data/memory/" # path to store the memory relative to run directory
DATA_PATH = f"{ROOT_PATH}/data/"
SOLVENT_DB_PATH = f"{DATA_PATH}/solvent_properties/solvent_properties.xlsx"
AMBIGUITY_DB_PATH = f"{DATA_PATH}/ambiguity_db/"
PAPER_PATH = "../data/downloaded_papers/"
XDL_SAVE_PATH = f"{REPO_PATH}/data/xdls/" # global storage independent of trial


# MODEL CONFIG
PAPER_SCRAPER_AGENT  = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "timeout": 150,
    "chunk_size": 4096, # Papers can be quite dense. Input is not the problem. Output is.
    "embedding_model": "text-embedding-3-large",
    "path": PAPER_PATH,
    "paper_memory": MEMORY_PATH,
}


CRITIQUE_AGENT = {
    "model": "gpt-4o",
    "temperature": 0,
    "timeout": 150,
}


XDL_AGENT = {
    "model": "gpt-4o",
    "temperature": 0,
    "timeout": 150,
    "embedding_model": "text-embedding-3-large",
    "k": 5,
    "data_path": DATA_PATH,
    "storage_path": MEMORY_PATH,
    "use_memory": True,
}


PROCEDURE_AGENT = {
    "model": "gpt-4o",
    "temperature": 0,
    "timeout": 150,
    "solvent_db": SOLVENT_DB_PATH,
}


AMBIGUIQTY_AGENT = {
    "model": "gpt-4o",
    "embedding_model": "text-embedding-3-large",
    "temperature": 0,
    "timeout": 150,
    "path": AMBIGUITY_DB_PATH,
}


# RUN CONFIG
RUN_CONFIG = {
    "iterations": 6,
    "filename": "procedure.xdl",
    "expand_memory": True,
    "log_file": "log.txt",
    "continuation": False,
    "trial_dir": "trial_0",
    "ask_ambiguous": False,
    "graph": None, # default template will be used if false
    "save_dir": XDL_SAVE_PATH,
}


# unambiguous identifiers in model names that are compatible with openai api.
# Note deepseek is callable over openai api
OPENAI_MODELS = ("gpt", "deepseek")
EMBEDDING_MODELS = ("text-embedding",)


PROVIDER_MAPPER = {
    "gpt": "openai",
    "deepseek": "deepseek",
    "text-embedding": "openai",
}


PROVIDER_URL = {
    "openai": None,
    "deepseek": "https://api.deepseek.com",
    "text-embedding": None,
}