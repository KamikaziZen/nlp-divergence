from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from transformers import set_seed
import torch
import numpy as np
import logging
from argparse import ArgumentParser, Namespace
import calendar
import time
import json


logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('--model1', type=str, required=True)
parser.add_argument('--model2', type=str, required=True)
parser.add_argument('--seq-length', type=int, required=True)
parser.add_argument('--num-samples', type=int, required=True)
parser.add_argument('--prompt', type=str, default='', required=False)
parser.add_argument('--prompt2', type=str, default=None, required=False)


def generate(model, prompt_text, num_seq, seq_length):
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path,
                                              device_map='auto',
                                              padding_side="left",
                                              # Needed for HF name change
                                              tokenizer_type='llama' if 'llama' in model.name_or_path else None, 
                                              add_bos_token=True,
                                              legacy=False,
                                              use_fast=False)
    print('generation prompt text:', prompt_text)
    prompt = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**prompt, do_sample=True, 
                                 min_new_tokens=seq_length, max_new_tokens=seq_length,
                                 num_return_sequences=num_seq)
        
    # checking that number of generated tokens is equal to seq_length
    assert len(outputs[0]) == len(prompt['input_ids'][0]) + seq_length
        
    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return texts


def get_log_sum(model, texts, prompt_text):
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
    num_skip_tokens = len(tokenizer.encode(prompt_text))
    logger.info('skipping tokens:', num_skip_tokens)
    print('skipping tokens:', num_skip_tokens)

    log_sums = np.zeros(len(texts))
    model.eval()
    with torch.no_grad():
        # computing one by one because different tokenizers may tokenize text into different number of tokens
        # padding or truncation may reduce the accuracy
        for i, text in enumerate(texts): 
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            for j, token in enumerate(inputs['input_ids'][0][num_skip_tokens:], start=num_skip_tokens): 
                log_sums[i] += log_probs[j-1][token]

    return log_sums


def compute_kl(model1, model2, prompt_text, num_samples, seq_length, prompt_text2=None):

    texts = generate(model1, prompt_text, num_seq=num_samples, seq_length=seq_length) 
    kls = get_log_sum(model1, texts, prompt_text) - \
          get_log_sum(model2, texts, prompt_text2 if prompt_text2 is not None else prompt_text)
    return kls


def main(args: Namespace):
    model1 = AutoModelForCausalLM.from_pretrained(args.model1,
                                                  load_in_8bit=True if 'llama' in args.model1 else False,
                                                  device_map='auto')
    model2 = AutoModelForCausalLM.from_pretrained(args.model2,
                                                  load_in_8bit=True if 'llama' in args.model2 else False,
                                                  device_map='auto')

    kls = compute_kl(model1, model2,
                     prompt_text=args.prompt, 
                     prompt_text2=args.prompt2,
                     num_samples=args.num_samples,
                     seq_length=args.seq_length)
    
    data = {
        'model1': args.model1,
        'model2': args.model2,
        'seq_length': args.seq_length,
        'num_samples': args.num_samples,
        'kls': kls.tolist(),
        'prompt': args.prompt,
        'prompt2': args.prompt2
    }

    current_GMT = time.gmtime()
    timestamp = calendar.timegm(current_GMT)
    with open(f"experiments/exp_{timestamp}.json", "w") as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    main(parser.parse_args())
