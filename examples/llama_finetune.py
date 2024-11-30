import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from diloco_sim import DilocoSimulator, DilocoSimulatorConfig
import json
import os
from typing import Dict, List
import torch.nn as nn

def modified_eval_model(trainer):
    trainer.model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(trainer.config.eval_iters):
            x, y = trainer._get_batch()
            output = trainer.model(x)
            loss = trainer.config.loss_fn(output, y)
            total_loss += loss.item()
    trainer.model.train()
    return total_loss / trainer.config.eval_iters

# Monkey patch the evaluation method
DilocoSimulator._eval_model = modified_eval_model

def identity_loss(x, _):
    return x.mean()

class AlpacaDatasetProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_instruction(self, instruction: str, input_text: str = "", output: str = "") -> str:
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    def process_function(self, examples):
        prompts = [
            self.format_instruction(
                instruction=instr,
                input_text=inp if inp else "",
                output=out
            )
            for instr, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
        ]
        
        encodings = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        encodings['labels'] = encodings['input_ids'].clone()
        return encodings

class LlamaAlpacaModel(nn.Module):
    def __init__(self, model_name: str, use_bfloat16: bool = True):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            device_map={"": 0}
        )
    
    def forward(self, x):
        device = next(self.model.parameters()).device
        batch_size = x.size(0)
        input_ids = x[:, 0, :].long().to(device)
        attention_mask = x[:, 1, :].long().to(device)
        labels = x[:, 2, :].long().to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss.view(1)

class AlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        labels = self.encodings['labels'][idx]
        
        model_inputs = torch.stack([input_ids, attention_mask, labels])
        target = torch.tensor([0.0])
        return model_inputs, target

    def __len__(self):
        return len(self.encodings['input_ids'])

def load_alpaca_dataset():
    local_path = "alpaca_data.json"
    
    if not os.path.exists(local_path):
        print("Downloading Alpaca dataset...")
        import requests
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        response = requests.get(url)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(local_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset_dict = {
        'instruction': [],
        'input': [],
        'output': []
    }
    
    for item in data:
        dataset_dict['instruction'].append(item['instruction'])
        dataset_dict['input'].append(item['input'])
        dataset_dict['output'].append(item['output'])
    
    return Dataset.from_dict(dataset_dict)

def prepare_datasets(tokenizer, max_length: int = 512):
    dataset = load_alpaca_dataset()
    dataset = dataset.train_test_split(test_size=0.1)
    
    processor = AlpacaDatasetProcessor(tokenizer, max_length)
    
    train_encodings = processor.process_function(dataset["train"])
    eval_encodings = processor.process_function(dataset["test"])
    
    train_dataset = AlpacaDataset(train_encodings)
    eval_dataset = AlpacaDataset(eval_encodings)
    
    return train_dataset, eval_dataset

class LlamaAlpacaConfig:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        save_dir: str = "./llama_alpaca_finetuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 512,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        eval_iters: int = 50,
        ckpt_interval: int = 500,
        use_bfloat16: bool = True,
        num_nodes: int = 2,
        diloco_interval: int = 500
    ):
        self.model_name = model_name
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_iters = eval_iters
        self.ckpt_interval = ckpt_interval
        self.use_bfloat16 = use_bfloat16
        self.num_nodes = num_nodes
        self.diloco_interval = diloco_interval

if __name__ == "__main__":
    torch.manual_seed(42)
    
    config = LlamaAlpacaConfig(
        model_name="nunoFcul/llama3.1_1B_adapted",
        num_epochs=3,
        batch_size=1,
        num_nodes=2
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, eval_dataset = prepare_datasets(tokenizer, config.max_length)
    
    diloco_config = DilocoSimulatorConfig(
        model_cls=LlamaAlpacaModel,
        model_kwargs={
            "model_name": config.model_name,
            "use_bfloat16": config.use_bfloat16
        },
        optimizer_kwargs={
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay
        },
        loss_fn=identity_loss,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir,
        eval_iters=config.eval_iters,
        ckpt_interval=config.ckpt_interval,
        num_nodes=config.num_nodes,
        diloco_interval=config.diloco_interval
    )
    
    trainer = DilocoSimulator(diloco_config)
    trainer.train()