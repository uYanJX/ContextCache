from typing import Optional

from openai import OpenAI as OpenAIClient

from vcache.context.dialogue import dialogue_to_chat_messages
from vcache.inference_engine.inference_engine import InferenceEngine


class OpenAIInferenceEngine(InferenceEngine):
    """
    OpenAI-compatible inference engine supporting both single-turn and multi-turn dialogues.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 1,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    @property
    def client(self) -> OpenAIClient:
        if self._client is None:
            kwargs = {"api_key": self.api_key}
            if self.base_url is not None:
                kwargs["base_url"] = self.base_url
            self._client = OpenAIClient(**kwargs)
        return self._client

    def create(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            messages = dialogue_to_chat_messages(prompt=prompt, system_prompt=system_prompt)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error creating completion from OpenAI: {e}")
