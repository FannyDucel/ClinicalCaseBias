#from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import os
from peft import PeftModel, PeftConfig
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help='Path to model.')
    parser.add_argument("--out-path", type=str, help="Path where the output files are saved")
    args = parser.parse_args()

    config = PeftConfig.from_pretrained(args.model_path)

    print("Loading the model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    len_tokenizer = len(tokenizer)

    print(len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    #model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.model_path)

    model = model.merge_and_unload()

    print("Done")

    print(f"Saving {type(model)}")

    model.save_pretrained(args.out_path)

    print("Done")
    print("Don't forget to copy the tokenizer's files")
