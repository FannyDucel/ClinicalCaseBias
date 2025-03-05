from transformers import AutoTokenizer
from typing import Dict
import re

class data_processor():

    prefix_vigogne: str = "### System"
    instruction_prefix_vigogne: str = "### Instruction"
    output_prefix_vigogne: str = "### Response"
    system_message_vigogne: str = (
    "Ci-dessous se trouve une instruction qui décrit une tâche à accomplir. Rédigez une réponse qui répond de manière"
    " précise à la demande."
    )

    instruction_prefix_base: str = ""
    output_prefix_base: str = ""
    #system_message_base: str = (
    #    "Je suis un médecin inventant des cas cliniques."
    #)
    system_message_base: str = ""


    def get_inference_message(self, instruction: str, input_text: str, template: str):
        symbols_to_strip: str = "!,-.:;?~ "

        instruction_text = re.sub("[" + re.escape(symbols_to_strip) + "]+$", "", instruction)
        instr = instruction_text
        instruction_text = f"{instruction_text} :\n{input_text}"

        if template == "llama-instruct":
            inference_message = [
                {"role": "system", "content": f"{instr}"},
                {"role": "user", "content": f"{input_text}"}
            ]

        if template == "vigogne":
            inference_message = f"{self.prefix_vigogne}:\n{self.system_message_vigogne}"
            inference_message += "\n\n" + f"{self.instruction_prefix_vigogne}:\n{instruction_text}"
            inference_message += "\n\n" + f"{self.output_prefix_vigogne}:\n"

        if template == "base":
            inference_message = f"{self.system_message_base}"
            inference_message += f"{self.instruction_prefix_base}{instruction_text}"
            inference_message += "\n" + f"{self.output_prefix_base}"

        return inference_message

    def process_example(self, instuct: Dict, tokenizer: AutoTokenizer, mask_input: bool, template: str, added_end_token: str=None):

        inference_message = self.get_inference_message(instuct["instruction"], instuct["input"], template)

        if template == "llama-instruct":
            inputs = tokenizer.apply_chat_template(inference_message, add_generation_prompt=True)
            output_ids = tokenizer(instuct["output"]).input_ids[1:]
            train_ids = inputs + output_ids
            tokenized_outputs = {
                "input_ids": train_ids,
                "attention_mask": [1 for x in range(len(train_ids))]
            }

            len_prompt = len(inputs) # (+1 si on ajoute le bos token)


        else:
            train_message = inference_message + instuct["output"]
            tokenized_outputs = tokenizer(train_message)

            len_prompt = len(tokenizer(inference_message)["input_ids"]) # (+1 si on ajoute le bos token)


        # si ajout de bos token
        #tokenized_outputs["input_ids"].insert(0, tokenizer.bos_token_id)
        #tokenized_outputs["attention_mask"].insert(0, 1)

        # si ajout de eos token
        tokenized_outputs["input_ids"].append(tokenizer.eos_token_id)
        tokenized_outputs["attention_mask"].append(1)

        input_ids = tokenized_outputs["input_ids"]

        #len_prompt = len(tokenizer(inference_message)["input_ids"]) # (+1 si on a ajouté le bos token avant)

        if mask_input:
            labels = ([-100] * len_prompt) + input_ids[len_prompt:]
        else:
            labels = input_ids.copy()

        processed_example = {"input_ids": input_ids, "labels": labels}

        return processed_example
