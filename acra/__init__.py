import os
import warnings

from dotenv import load_dotenv
import openai

from acra.main import (
    check_xdl,
    paper_to_xdl,
    procedure_to_xdl,
)

try:
    from xdl import XDL
    import ChemputerAPI
    import SerialLabware
    import commanduino
    from chempiler import Chempiler
    from chemputerxdl import graphgen
    from chemputerxdl.platform import ChemputerPlatform
    os.environ["CHEMPU_AVAILABLE"] = "Y"
except ImportError:
    warnings.warn("Running without entire XDL/Chemputer stack")
    os.environ["CHEMPU_AVAILABLE"] = "N"

try:
    load_dotenv("../.env")
except Exception:
    warnings.warn("No .env file found")

try:
    openai.api_key = os.environ["CHAT_API_KEY"]
except Exception:
    warnings.warn("Set your CHAT_API_KEY before proceeding")

try:
    openai.api_key = os.environ["EMBEDDING_API_KEY"]
except Exception:
    warnings.warn("Set your EMBEDDING_API_KEY before proceeding")