from vllm import LLM, SamplingParams
from utils import get_constraints
import os
from processors import data_processor
from peft import PeftModel, PeftConfig
import argparse
import torch
import json
import time
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help='Path to model.')
    parser.add_argument("--test-path", type=str, help="Path to test data.")
    parser.add_argument("--out-path", type=str, default=5, help="Path where the output files are saved")
    parser.add_argument("--candidates", type=int, help='Number of candidates for each constraint.')
    parser.add_argument("--template", type=str, help='Template of inputs.')
    args = parser.parse_args()

    constraints = get_constraints(args.test_path)
    print(f"{len(constraints)} jeux de contraintes.")

    print("Loading the model...")

    sampling_params = SamplingParams(
        temperature=1.0,
        best_of=3,
        top_p=0.9,
        #top_k=300,
        max_tokens=1300,
        #repetition_penalty=1.03,
        skip_special_tokens=False
    )

    llm = LLM(
	model=args.model_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=8,
        distributed_executor_backend="mp",
        enable_chunked_prefill=False,
        ## deux paramètres suivant à dé-commenter pour Llama-3.1
        #max_model_len=4096,
        #max_num_batched_tokens=65528,
    )


    print("Done")

    processor = data_processor()

    outputs = []

    print("Building prompts")

    prompts = []

    refs = []

    nb_candidats = args.candidates

    for constraint in constraints:

        refs.append((constraint[1], constraint[2], constraint[3], constraint[-1]))

        p = processor.get_inference_message(
            instruction=constraint[0],
            input_text=constraint[1],
            template=args.template,
        )

        if args.template == "llama-instruct":
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            input_ids = tokenizer.apply_chat_template(p, add_generation_prompt=True)
            p = tokenizer.decode(input_ids)

        for i in range(nb_candidats):
            prompts.append(p)

    print(f"{len(prompts)} prompts")

    print(prompts[0])

    print("Generating...")

    deb = time.time()

    outputs = llm.generate(prompts, sampling_params)

    fin = time.time()

    cpt = 0

    dic_res = []

    for i in range(len(refs)):
        dic_res.append({})
        dic_res[i]["fichier"] = refs[i][2]
        dic_res[i]["reference"] = refs[i][1]
        dic_res[i]["input"] = refs[i][0]
        dic_res[i]["constraints"] = refs[i][-1]
        dic_res[i]["candidats"] = []
        for j in range(i*nb_candidats, (i+1)*nb_candidats):
            dic_res[i]["candidats"].append(outputs[j].outputs[0].text)

    with open(f"{args.out_path}.json", "w", encoding="utf-8") as fout:
        fout.write(json.dumps(dic_res, ensure_ascii=False, indent=1))

    print("Done")

    print(f"{(i+1)*nb_candidats} files generated in {fin-deb}")
