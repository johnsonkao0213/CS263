from google import genai
from google.genai import types


class GeminiWrapper:
    def __init__(self, api_key=None, model_name="gemini-3-flash-preview"):
        self.backend = "gemini"
        self.model_name = model_name
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def _extract_system_and_prompt(self, message):
        if isinstance(message, list):
            system_parts = []
            user_parts = []
            for item in message:
                role = item.get("role")
                content = item.get("content", "")
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(content)
            system_instruction = "\n".join([p for p in system_parts if p])
            prompt = "\n".join([p for p in user_parts if p])
        else:
            system_instruction = ""
            prompt = message.get("content", "") if isinstance(message, dict) else str(message)
        return system_instruction, prompt

    def generate(
        self,
        messages,
        max_new_tokens=512,
        temperature=1.0,
        top_p=0.9,
        stop_sequences=None,
    ):
        responses = []
        for message in messages:
            system_instruction, prompt = self._extract_system_and_prompt(message)
            config_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_new_tokens,
            }
            if stop_sequences:
                config_kwargs["stop_sequences"] = stop_sequences
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            config = types.GenerateContentConfig(**config_kwargs)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            responses.append(response.text or "")
        return responses
