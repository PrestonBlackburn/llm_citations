from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/Phi-3-mini-4k-instruct"
local_dir = "./models/phi-3"

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_dir)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(local_dir)
