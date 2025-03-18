from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

model = T5ForConditionalGeneration.from_pretrained("./trained_distilgpt2_chatbot")
tokenizer = T5Tokenizer.from_pretrained("./trained_distilgpt2_chatbot")

chatbot_pipeline = pipeline("text2text-generation", model="trained_distilgpt2_chatbot", tokenizer="trained_distilgpt2_chatbot")

# Sample query
query = "How do I track my order?"
response = chatbot_pipeline("respond: " + query, max_length=50)
print("Chatbot Response:", response[0]["generated_text"])