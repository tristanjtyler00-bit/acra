import os

import openai
import numpy as np

from acra.config import OPENAI_MODELS, EMBEDDING_MODELS, PROVIDER_MAPPER, PROVIDER_URL


def get_provider(model: str) -> str:
    for m, provider in PROVIDER_MAPPER.items():
        # some of m should a substring of a model. e.g. gpt of gpt-4o-2024-08-06
        if m in model.lower(): 
            return provider
    
    raise ValueError("Model not supported")


def get_base_url(model: str) -> str:
    return PROVIDER_URL.get(get_provider(model))


class BaseAgent:
    method_map = {
        "openai": "chat_openai",
        "deepseek": "chat_openai",
    }

    # deepseek does not have an embedding model at this point
    embedding_method_map = {
        "openai": "embed_openai",
    }

    def __init__(
        self,
        model: None | str,
        embedding_model: None | str = None,
        temperature: int = 0,
        timeout: int = 150,
        response_format: dict = {"type": "json_object"},
    ) -> None:
        assert model is not None or embedding_model is not None, "Either model or embedding model must be provided"
        self.model = model
        self.embedding_model = embedding_model

        # abstract attributes
        self.temperature = temperature
        self.timeout = timeout
        self.response_format = response_format

        self.set_up_client()

    def set_up_client(self):
        """
        Seperated setup to be able to mix from different providers
        """

        # chat client
        try:
            if self.model is None:
                ...
            else:
                provider = get_provider(self.model)
                self.base_url = get_base_url(self.model)

                # should be fairly bullet proof. Checking for openai compatibility
                if any((model in self.model for model in OPENAI_MODELS)):
                    self.client = openai.OpenAI(
                        api_key=os.environ.get("CHAT_API_KEY"),
                        base_url=self.base_url
                    )
                    self.chat_method = self.method_map.get(provider)
                else:
                    raise ValueError("Model not supported")
        except Exception as e:
            raise e

        # embedding client
        try:
            if self.embedding_model is None:
                ...
            else:
                provider = get_provider(self.embedding_model)
                self.embed_base_url = get_base_url(self.embedding_model)

                # should be fairly bullet proof. Checking for openai compatibility
                if any((
                    model in self.embedding_model for model in EMBEDDING_MODELS
                )):
                    self.embed_client = openai.OpenAI(
                        api_key=os.environ.get("EMBEDDING_API_KEY"),
                        base_url=self.embed_base_url
                    )
                    self.embed_method = self.embedding_method_map.get(provider)
                else:
                    raise ValueError("Model not supported")
        except Exception as e:
            raise e
        return None


    def embed(self, text: str) -> np.ndarray:
        return getattr(self, self.embed_method)(text)

    def embed_openai(self, text: str) -> np.ndarray:
        return (
            self.embed_client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            .data[0]
            .embedding
        )


    def chat(self, system_prompt: str = '', prompt: str = '') -> str:
        return getattr(self, self.chat_method)(system_prompt, prompt)

    def chat_openai(self, system_prompt: str = '', prompt: str = '') -> str:
        return self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=self.response_format,
                ).choices[0].message.content
    
    def __str__(self):
        return self.__class__.__name__
    


