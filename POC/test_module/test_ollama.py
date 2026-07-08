import ollama

def test_ollama():
    test_prompt = 'The agent is at (0,0) in a 5x5 grid. The goal is at (4,4). Provide a single direction (Up, Down, Left, Right) to move to get closer to the goal.'
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Hello!'}])
    print(response['message']['content'])