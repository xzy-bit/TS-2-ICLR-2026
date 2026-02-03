from datasets import load_dataset

load_dataset("HuggingFaceH4/ultrafeedback_binarized",trust_remote_code=True)
load_dataset("tatsu-lab/alpaca_eval",trust_remote_code=True)
load_dataset("openai_humaneval",trust_remote_code=True)