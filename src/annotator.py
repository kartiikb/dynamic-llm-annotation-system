import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMAnnotator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables."
            )

        self.client = Groq(api_key=api_key)

        self.model = "openai/gpt-oss-120b"

    def annotate(self, prompt):
        """
        Send one annotation prompt to the LLM
        and return the parsed JSON response.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"LLM returned invalid JSON:\n{text}"
            )

        return result