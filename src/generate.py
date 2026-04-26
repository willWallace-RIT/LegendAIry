from persona_engine import PersonaEngine

def generate_response(problem):
    engine = PersonaEngine("models/embeddings.pkl")

    # Barney solving for Ted
    persona_vector = engine.blend_personas("Barney", "Ted", alpha=0.7)

    # Instead of raw vector generation, we guide an LLM
    style = engine.describe_style(persona_vector)

    prompt = f"""
    You are Barney Stinson helping Ted Moseby solve a problem.

    Barney traits:
    - Confident
    - Strategic
    - Slightly manipulative
    - Charismatic

    Ted traits:
    - Romantic
    - Overthinking
    - Idealistic

    Problem:
    {problem}

    Respond as Barney solving this FOR Ted.
    """

    # Plug into OpenAI or local LLM
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-5.3-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    problem = "I met someone but I'm not sure if she's the one."
    print(generate_response(problem))
