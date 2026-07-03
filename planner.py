import ollama

def get_llm_hint(current_pos):
    # Prompt the LLM with the current state
    prompt = f"The agent is at {current_pos} in a 5x5 grid. The goal is at (4,4). Provide a single direction (Up, Down, Left, Right) to move to get closer to the goal."

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    hint = response["message"]["content"].lower()

    # Mapping LLM text to action integers
    mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    for key in mapping:
        if key in hint:
            return mapping[key]
    
    return None # Return None if LLM is unclear