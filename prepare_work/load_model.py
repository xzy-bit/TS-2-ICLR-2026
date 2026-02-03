from transformers import AutoTokenizer, AutoModelForCausalLM , AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from huggingface_hub import login
from huggingface_hub import snapshot_download

auth_token = ""

# Base Model
tokenizer_ = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",token=auth_token)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B",token=auth_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",token=auth_token)

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",token=auth_token)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct",token=auth_token)

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")

# Reward Model
tokenizEr = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
model_ = AutoModelForSequenceClassification.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")

# Sentence Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-large-nli-stsb-mean-tokens")
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2",token=auth_token)
model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-mpnet-base-v2",token=auth_token)