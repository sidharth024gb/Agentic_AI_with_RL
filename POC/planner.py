import ollama

# This dictionary stores previous LLM responses to avoid redundant API calls
_hint_cache = {}

def get_llm_hint(current_pos):
    pos_key = tuple(current_pos)

    # If we have asked the LLM this before, return the stored answer
    if pos_key in _hint_cache:
        return _hint_cache[pos_key]
    
    print(f"--- LLM Planning for position {pos_key} ---")
    # Prompt the LLM with the current state
    prompt = f"The agent is at {pos_key} in a 5x5 grid. The goal is at (4,4). Provide a single direction (Up, Down, Left, Right) to move to get closer to the goal."

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    hint = response["message"]["content"].lower()

    # Mapping LLM text to action integers
    mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

    selected_action = None # Return None if LLM is unclear
    for key in mapping:
        if key in hint:
            selected_action =  mapping[key]
            break
    
    # Save the result to cache before returning
    _hint_cache[pos_key] = selected_action
    return None 