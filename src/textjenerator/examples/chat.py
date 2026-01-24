from textjenerator import registry


config = {
    # model
    "model": "llama-cpp",
    "model_path": ".models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",

    # hardware/system
    "device": "cpu",
    "dtype": "float32",
    "number_of_threads": 8,
    "n_gpu_layers": -1,

    # LLM
    "verbose_warnings": False,
    "max_context_size": 4096,
    "max_tokens_per_response": 256,
    "temperature": .8 ,
    "top_p": 0.9,
    "top_k": 10,
    "messages": [
          {"role": "system", "content": """You are Jenbot, an expert, helpful, and diligent assistant. You provide the user with accurate answers to their queries. You are polite, friendly, and a little sarcastic."""},
    ]
}

text_generator = registry.get_model_class(config)
text_generator.load()

while True:
    user_input = input("You: ")
    print()
    user_message = {"role": "user", "content": f"{user_input}"}
    text_generator.config["messages"].append(user_message)
    output = text_generator.generate()
    response = output.batch[0].data
    assistant_message = {"role": "assistant", "content": response}
    text_generator.config["messages"].append(assistant_message)
    print("Assistant:", response, "\n")
