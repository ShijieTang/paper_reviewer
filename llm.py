import os
import openai


def get_llm_response(dev_role, user_role, model="gpt-5"):
    # Setup
    API_KEY = os.environ["API_KEY"]
    client = openai.OpenAI(api_key=API_KEY, base_url="https://ai-gateway.andrew.cmu.edu")
    # Get response from model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": f"{dev_role}"},
            {"role": "user", "content": f"{user_role}"}
        ]
    )
    # Turn returned score into a float
    response = float(response.choices[0].message.content.strip())
    return response