from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model_path = "./trained_distilgpt2_chatbot"

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

chatbot_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

query = "I need Customer Support"
response = chatbot_pipeline(query, max_length=50, num_return_sequences=1)

print("Chatbot Response:", response[0]["generated_text"])
