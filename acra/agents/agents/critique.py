import os

import openai

from acra.agents.prompts import critique_prompt, steps, steps_OT
from acra.config import CRITIQUE_AGENT
from acra.utils.json_utils import extract_json_from_response
from .base_agent import BaseAgent


steps_mapping = {
    "chemputer": steps.steps,
    "OT": steps_OT.steps,
}

class CritiqueAgent(BaseAgent):
    """
    Find inconsistencies in the XDL compared to the NLP synthesis instructions
    """

    def __init__(
        self,
        prompt_template: str = critique_prompt.critique_prompt,
        model: int = CRITIQUE_AGENT.get("model", "gpt-4o"),
        temperature: int = CRITIQUE_AGENT.get("temperature", 0),
        timeout: int = CRITIQUE_AGENT.get("timeout", 150),
        platform: str = "chemputer",
    ) -> None:
        super().__init__(model=model, temperature=temperature, timeout=timeout, response_format=None)

        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        if platform not in steps_mapping:
            platform = "chemputer"
            print("Platform not found, using default platform 'chemputer'")
        self.steps = steps_mapping[platform]

    def format_prompt(self, procedure, xdl_code, human_readable, chemicals) -> str:
        return self.prompt_template.format(
            steps_description=self.steps,
            procedure_description=procedure,
            code=xdl_code,
            human_readable=human_readable,
            response_format=critique_prompt.response_format,
            chemicals=chemicals
        )

    def format_inclusion_prompt(
        self, procedure, xdl_code, human_readable, missing_steps
    ) -> str:
        return critique_prompt.inclusion_prompt.format(
            # steps_description=steps.steps,
            procedure_description=procedure,
            code=xdl_code,
            human_readable=human_readable,
            requested_correction=missing_steps,
        )

    def _extract_suggestion(self, response: str) -> str:
        suggestion = extract_json_from_response(response)
        return suggestion

    def generate_suggestion(
        self,
        procedure,
        xdl_code,
        human_readable,
        chemicals,
    ) -> str:
        """
        Prompt LLM with the given task
        """
        self.prompt = self.format_prompt(
            procedure=procedure,
            xdl_code=xdl_code,
            human_readable=human_readable,
            chemicals=chemicals,
        )

        self.response = self.chat(critique_prompt.system_prompt, self.prompt)
        return self._extract_suggestion(self.response)

    def include_suggestion_in_xdl(
        self, procedure, xdl_code, human_readable, missing_steps
    ) -> str:
        """
        Prompt LLM with the given task
        """
        self.inclusion_prompt = self.format_inclusion_prompt(
            procedure=procedure,
            xdl_code=xdl_code,
            human_readable=human_readable,
            missing_steps=missing_steps,
        )

        response = self.chat(critique_prompt.inclusion_system_prompt, self.inclusion_prompt)
        return response
