#!/usr/bin/env python3

from openai import OpenAI
from typing import List
from openai.types.chat import ChatCompletion

class Client(OpenAI):
    def __init__(self, api_key):
        super().__init__(api_key=api_key)

    def transcribe(self, audio_file: bytes) -> str:
        transcript = self.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )
        return transcript
    
    def summarize(self, transcript: str, prompt_instructions: List[str], model: str = "gpt-4", temperature: float = 0.9) -> ChatCompletion:
        messages = [
            *({"role": "system", "content": prompt_instruction} for prompt_instruction in prompt_instructions),
            {"role": "system", "content": "Here is the transcript: " + transcript}
        ]
        response = self.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format="text",
        )
        return response

