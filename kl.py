from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import set_seed
import torch
import numpy as np
import logging
from argparse import ArgumentParser, Namespace
import calendar
import time
import json


parser = ArgumentParser()
parser.add_argument('--model1', type=str, required=True)
parser.add_argument('--model2', type=str, required=True)
parser.add_argument('--seq-length', type=int, required=True)
parser.add_argument('--num-samples', type=int, required=True)
parser.add_argument('--prompt1', type=str, default='', required=False)
parser.add_argument('--prompt2', type=str, default='', required=False)
parser.add_argument('--system1', type=str, default='', required=False)
parser.add_argument('--system2', type=str, default='', required=False)

LLAMA2_INSTR_TEMPLATE = """[INST] <<SYS>>
{} 
<</SYS>>

{} [/INST]""" 


def modify_prompt(model_name, prompt_text, system_text):
    model_name = model_name.lower()

    # add system prompt and instruction tokens if model supports instructions
    if 'llama-2' in model_name and 'chat' in model_name: 
        prompt_text = LLAMA2_INSTR_TEMPLATE.format(system_text, prompt_text)

    return prompt_text


def generate(model, num_seq, seq_length, prompt_text, system_text):
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path,
                                              device_map='auto',
                                              padding_side="left",
                                              # Needed for HF name change?
                                              tokenizer_type='llama' if 'llama' in model.name_or_path else None, 
                                              add_bos_token=True,
                                              legacy=False,
                                              use_fast=False)
    # check that bos token is added
    assert len(tokenizer.encode('')) == 1
    
    prompt_text = modify_prompt(model.name_or_path, prompt_text, system_text)
    print('Generation prompt:', prompt_text)

    prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    num_prompt_tokens = len(prompt['input_ids'][0])

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**prompt, do_sample=True, 
                                 min_new_tokens=seq_length, max_new_tokens=seq_length,
                                 num_return_sequences=num_seq)
        
    # checking that number of new generated tokens is equal to seq_length
    assert len(outputs[0]) == num_prompt_tokens + seq_length
        
    # removing prompt tokens and converting to text
    texts = [tokenizer.decode(o[num_prompt_tokens:], skip_special_tokens=True) for o in outputs]

    torch.cuda.empty_cache()

    return texts


def get_log_sum(model, texts, prompt_text, system_text):
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path,
                                              device_map='auto',
                                              padding_side="left",
                                              # Needed for HF name change?
                                              tokenizer_type='llama' if 'llama' in model.name_or_path else None, 
                                              add_bos_token=True,
                                              legacy=False,
                                              use_fast=False)
    # check that first token is bos
    assert len(tokenizer.encode('')) == 1
    
    prompt_text = modify_prompt(model.name_or_path, prompt_text, system_text)
    print('Log calculation prompt:', prompt_text)
    
    num_skip_tokens = len(tokenizer.encode(prompt_text))
    print('Skipping tokens:', num_skip_tokens)

    log_sums = np.zeros(len(texts))
    model.eval()
    with torch.no_grad():
        # computing one by one because different tokenizers may tokenize text into different number of tokens
        # padding or truncation may reduce the accuracy
        for i, text in enumerate(texts): 
            inputs = tokenizer(prompt_text + text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            for j, token in enumerate(inputs['input_ids'][0][num_skip_tokens:], start=num_skip_tokens): 
                log_sums[i] += log_probs[j-1][token]

            torch.cuda.empty_cache()

    return log_sums


def compute_kl(model1, model2, num_samples, seq_length, 
               prompt_text1, prompt_text2, 
               system_text1, system_text2):

    texts = generate(model1, 
                     prompt_text=prompt_text1, 
                     num_seq=num_samples, 
                     seq_length=seq_length,
                     system_text=system_text1) 
    kls = get_log_sum(model1, texts, prompt_text1, system_text1) - \
          get_log_sum(model2, texts, prompt_text2, system_text2)
    return kls


def main(args: Namespace):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model1 = AutoModelForCausalLM.from_pretrained(
        args.model1,
        quantization_config=quantization_config if 'llama' in args.model1.lower() else False,
        device_map='auto',
        # takes more time but requires less memory
        use_cache=False,
    )
    if args.model2 != args.model1:
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model2,
            quantization_config=quantization_config if 'llama' in args.model2.lower() else False,
            device_map='auto',
            # takes more time but requires less memory
            use_cache=False
        )
    else: 
        model2 = model1

    kls = compute_kl(model1, model2,
                     num_samples=args.num_samples,
                     seq_length=args.seq_length,
                     prompt_text1=args.prompt1, 
                     prompt_text2=args.prompt2,
                     system_text1=args.system1,
                     system_text2=args.system2)
    
    data = {
        'model1': args.model1,
        'model2': args.model2,
        'seq_length': args.seq_length,
        'num_samples': args.num_samples,
        'kls': kls.tolist(),
        'prompt1': args.prompt1,
        'prompt2': args.prompt2,
        'system1': args.system1,
        'system2': args.system2
    }

    current_GMT = time.gmtime()
    timestamp = calendar.timegm(current_GMT)
    with open(f"experiments/exp_{timestamp}.json", "w") as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    main(parser.parse_args())
