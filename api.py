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
    
    def summarize(self, transcript: str, prompt_instructions: List[str], model: str = "gpt-4", temperature: float = 0.9) -> str:
        messages = [
            *({"role": "system", "content": prompt_instruction} for prompt_instruction in prompt_instructions),
            {"role": "system", "content": "Here is the transcript: " + transcript}
        ]
        response = self.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    print("This is a library and should not be run directly.")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = Client(OPENAI_KEY)
    transcript = client.transcribe(open("Corn.m4a", "rb"))
    summary = client.summarize(transcript, ["Summarize the audio file."])
    print(summary)