import ollama
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize the Gemini client 
# (It will automatically look for an environment variable named GEMINI_API_KEY)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# This dictionary stores previous LLM responses to avoid redundant API calls
_hint_cache = {}

def get_llm_hint(current_pos):
    pos_key = tuple(current_pos)

    # If we have asked the LLM this before, return the stored answer
    if pos_key in _hint_cache:
        return _hint_cache[pos_key]
    
    print(f"--- LLM Planning for position {pos_key} ---")
    # Prompt the LLM with the current state
    prompt =  (
        f"The agent is at {pos_key} in a 5x5 grid. The goal is at (4,4). "
        f"Provide exactly one word indicating the direction to move: up, down, left, or right."
    )

    # Ollama Response
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    hint = response["message"]["content"].lower()

    # # Call Gemini 1.5 Flash (free tier)
    # response = client.models.generate_content(
    #     model='gemini-2.5-flash',
    #     contents=prompt,
    # )
    # hint = response.text.lower()

    # Mapping LLM text to action integers
    mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

    selected_action = None # Return None if LLM is unclear
    for key in mapping:
        if key in hint:
            selected_action =  mapping[key]
            break
    
    # Save the result to cache before returning
    _hint_cache[pos_key] = selected_action
    return selected_action 