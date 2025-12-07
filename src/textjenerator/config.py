config = {
    # model
    "model": "llama-cpp",
    "model_path": "../jenbot/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",

    # hardware/system
    "device": "cpu",
    "dtype": "float32",

    # LLM
    "max_context_size": 4096,
    "number_of_threads": 8 ,
    "verbose_warnings": False,
    "messages_to_keep_in_context": 4,
    "max_tokens_per_response": 256,
    "temperature": 0.7 ,
    "top_p": 0.9,
    "top_k": 50,
    "messages": [ 
          {"role": "system", "content": """You are Jenbot, an expert, helpful, and diligent assistant. You provide the user with accurate answers to their queries. You are polite, friendly, and a little sarcastic."""},
          {"role": "user", "content": """Hi, who are you?"""},
    ]
}