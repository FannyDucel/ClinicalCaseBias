import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from data_class import DataProcess
from peft import LoraConfig, get_peft_model
import accelerate
import peft
import torch
import argparse
from utils import load_special_tokens
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help='Path to model.')
    parser.add_argument("--training-file", type=str, help='Path to training file.')
    parser.add_argument("--validation-file", type=str, help='Path to validation file.')
    parser.add_argument("--save-path", type=str, help='Path where to save the finetuned model.')
    parser.add_argument("--template", type=str, help='Instruction template. Choose between "vigogne" and "bloom".')
    parser.add_argument("--batch-size", type=int, default=4, help='int: Batch size for dataloaders.')
    parser.add_argument("--block-size", type=int, default=128, help='int: Block size for data process class.')
    parser.add_argument("--lr", type=float, default=2e-4, help='float: Learning rate for fine-tuning.')
    parser.add_argument("--max-epochs", type=int, default=10, help='int: maximum number of epochs of training.')
    args = parser.parse_args()

    data_process = DataProcess(args.model_path, block_size=args.block_size)

    print("Loading data")
    train_data, val_data, tokenizer = data_process.prepare_data_instruct(args.training_file, args.validation_file, template=args.template, batch_size=args.batch_size, hf_trainer=True)
    print("Loading model")
    print(len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    #model.resize_token_embeddings(len(tokenizer))
    print("Done")
    print(model.hf_device_map)
    print(model.dtype)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj"], # for Llama-based models
        #target_modules=["query_key_value"], # for Bloom models
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    callbacks = [
        transformers.EarlyStoppingCallback(early_stopping_patience=2)
    ]
    training_args = TrainingArguments(
        output_dir=args.save_path,
        eval_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.max_epochs,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        seed=42,
        #fp16=True,
        bf16=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=False,
        auto_find_batch_size=True,
        #per_device_train_batch_size=1,
        #per_device_eval_batch_size=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )

    deb = time.time()

    trainer.train()

    fin = time.time()

    print(f"Durée de l'entraînement : {fin-deb}")

    print(f"Modèle: {args.model_path}")
    print(config)

    print("Done")
