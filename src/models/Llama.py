import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import _crop_past_key_values
from huggingface_hub import login
from .Model import Model
from src.utils import clean_str
import re
import string


class Llama(Model):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        login(token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float32).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"
        self.model.requires_grad_(False) # Freeze model weights
        
    
    def query(self, msg, top_tokens=100000):
        inputs = self.tokenizer(msg, padding=True, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                temperature=self.temperature,
                max_new_tokens=self.max_output_tokens,
                early_stopping=True,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=False,
            )

        out = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        result = out[len(msg):]

        generated_ids = outputs.sequences[:, input_ids.shape[1]:]
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        punctuation_set = set(string.punctuation)
        total_attention_values = None
        
        for i in range(len(generated_tokens) - 1):
            if i != 0 and total_attention_values is not None and generated_tokens[i] == '<0x0A>':
                break
            clean_token = generated_tokens[i]
            if not clean_token or clean_token in punctuation_set or clean_token == self.tokenizer.eos_token or clean_token == '<0x0A>':
                continue

            attention_scores = torch.stack(outputs.attentions[i + 1]).mean(dim=0)
            avg_attention = attention_scores.mean(dim=1)[0]

            token_attention = avg_attention[:, :input_ids.shape[1]]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_values = [token_attention[0][i].item() if input_tokens[i] not in punctuation_set else 0 for i in range(len(input_tokens))]

            if total_attention_values is None:
                total_attention_values = attention_values
            else:
                total_attention_values = [sum(x) for x in zip(total_attention_values, attention_values)]
        
        pattern = r'\[\d+\]\s+(.*?)(?=\n\[\d+\]|\n-{5,}|\Z)'
        matches = re.finditer(pattern, msg, re.DOTALL)

        passage_indices = []
        for match in matches:
            passage_indices.append((match.start(), match.end()))

        token_passage_indices = []
        msg_tokens = self.tokenizer(msg[:passage_indices[0][0]], return_tensors="pt").input_ids
        start = msg_tokens.shape[1]
        # computing the start and end token of each retrieved passage
        for passage_idx in passage_indices:
            msg_tokens = self.tokenizer(msg[:passage_idx[1]], return_tensors="pt").input_ids
            end = msg_tokens.shape[1]
            token_passage_indices.append((start, end))
            start = end 

        passage_scores = []
        passage_attention_values = []
        passage_tokens = []

        for token_idx in token_passage_indices:
            if total_attention_values is not None:
                attention_slice = total_attention_values[token_idx[0]: token_idx[1]]
                top_alpha = sorted(attention_slice, reverse=True)[:min(top_tokens, len(attention_slice))] # change this to conisder different top \aplha tokens from each passage
                score = sum(top_alpha)
            else:
                score = 1
            passage_scores.append(score)
        total_attention_score = sum(total_attention_values) if total_attention_values is not None else 10
        
        return result, passage_scores, total_attention_score
    
