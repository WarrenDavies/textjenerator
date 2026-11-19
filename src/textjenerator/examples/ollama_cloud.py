from textjenerator import registry

config = {
    # model
    "backend": "ollama_cloud",
    "model_path": "nemotron-3-super", # replace with your preferred model
    "api_key": "", # replace with your API Key
    "messages": [ 
        {"role": "system", "content": """You are Jenbot, an expert, helpful, and diligent assistant. You provide the user with accurate answers to their queries. You are polite, friendly, and a little sarcastic."""},
    ]
}

text_generator = registry.get_model_class(config)
text_generator.load()

while True:
    user_input = input("You: ")
    print("\n\n\n")
    user_message = {"role": "user", "content": f"{user_input}"}
    text_generator.config["messages"].append(user_message)
    output = text_generator.generate()
    response = output.batch[0].data
    assistant_message = {"role": "assistant", "content": response}
    text_generator.config["messages"].append(assistant_message)
    print("Assistant:", response, "\n\n\n")
