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


LLAMA2_INSTR_TEMPLATE = """[INST] <<SYS>>
{}
<</SYS>>

{} [/INST]"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=False)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--prompt1', type=str, default='', required=False)
    parser.add_argument('--prompt2', type=str, default='', required=False)
    parser.add_argument('--system1', type=str, default='', required=False)
    parser.add_argument('--system2', type=str, default='', required=False)

    args = parser.parse_args()
    if args.metric == 'ent' and args.model2 is not None:
        raise ValueError('Entropy is calculated for one model '
                         'but argument <model2> was passed.')
    if args.metric not in ['kl', 'js', 'ent']:
        raise ValueError(f'Metric not foudn: {args.metric}')

    return args


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
                                 min_new_tokens=seq_length,
                                 max_new_tokens=seq_length,
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
        # TODO: do batch computation, mask padding tokens in attention mask? 
        # but computing one-by-one requires less memory...
        for i, text in enumerate(texts): 
            inputs = tokenizer(prompt_text + text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            for j, token in enumerate(inputs['input_ids'][0][num_skip_tokens:], start=num_skip_tokens): 
                log_sums[i] += log_probs[j-1][token]

            torch.cuda.empty_cache()

    return log_sums


def get_probs(model, texts, prompt_text, system_text):
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

    probs = []
    model.eval()
    with torch.no_grad():
        # computing one by one because different tokenizers may tokenize text into different number of tokens
        # padding or truncation may reduce the accuracy
        # TODO: do batch computation, mask padding tokens in attention mask? 
        # but computing one-by-one requires less memory...
        for i, text in enumerate(texts):
            probs_i = np.empty(0)
            inputs = tokenizer(prompt_text + text, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            logits = outputs.logits[0]
            sm = torch.softmax(logits, dim=-1)

            for j, token in enumerate(inputs['input_ids'][0][num_skip_tokens:], start=num_skip_tokens):
                probs_i = np.append(probs_i, sm[j-1][token].cpu().item())

            probs.append(probs_i)

            torch.cuda.empty_cache()

    return probs


def main(args: Namespace):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model1,
        quantization_config=quantization_config if 'llama' in args.model1.lower() else None,
        device_map='auto',
        # takes more time but requires less memory
        use_cache=False,
    )

    texts = generate(
        model,
        prompt_text=args.prompt1,
        num_seq=args.num_samples,
        seq_length=args.seq_length,
        system_text=args.system1)

    p_probs = get_probs(model, texts, args.prompt1, args.system1)
    # for each sample text take log of probs and summ them
    p_logsums = np.array([np.log(p).sum() for p in p_probs])

    if args.model2 is not None and args.model2 != args.model1:
        model = AutoModelForCausalLM.from_pretrained(
            args.model2,
            quantization_config=quantization_config if 'llama' in args.model2.lower() else None,
            device_map='auto',
            # takes more time but requires less memory
            use_cache=False
        )

    if args.metric == 'kl':
        q_probs = get_probs(model, texts, args.prompt2, args.system2)
        q_logsums = np.array([np.log(q).sum() for q in q_probs])

        kls = p_logsums - q_logsums
    elif args.metric == 'js':
        # or just use np.float128?
        # p_logsums_minus = p_logsums - p_logsums.max()
        # p_probs_minus = np.exp(p_logsums_minus)

        q_probs = get_probs(model, texts, args.prompt2, args.system2)
        q_logsums = np.array([np.log(q).sum() for q in q_probs])
        # q_logsums_minus = q_logsums - p_logsums.max()
        # q_probs_minus = np.exp(q_logsums_minus)
        
        # m_logsums = p_logsums.max() - np.log(2) + \
        #             np.log(p_probs_minus + q_probs_minus)

        # jss = p_logsums - m_logsums
        jss = - np.exp(q_logsums - p_logsums)

        texts = generate(
            model,
            prompt_text=args.prompt2,
            num_seq=args.num_samples,
            seq_length=args.seq_length,
            system_text=args.system2)

        q_probs = get_probs(model, texts, args.prompt2, args.system2)
        q_logsums = np.array([np.log(q).sum() for q in q_probs])
        # q_logsums_minus = q_logsums - q_logsums.max()
        # q_probs_minus = np.exp(q_logsums_minus)

        if args.model2 != args.model1:
            model = AutoModelForCausalLM.from_pretrained(
                args.model1,
                quantization_config=quantization_config if 'llama' in args.model1.lower() else None,
                device_map='auto',
                # takes more time but requires less memory
                use_cache=False
            )

        p_probs = get_probs(model, texts, args.prompt1, args.system1)
        p_logsums = np.array([np.log(p).sum() for p in p_probs])
        # p_logsums_minus = p_logsums - q_logsums.max()
        # p_probs_minus = np.exp(p_logsums_minus)

        # m_logsums = q_logsums.max() - np.log(2) + \
        #             np.log(p_probs_minus + q_probs_minus)
        
        # jss += q_logsums - m_logsums
        # jss -= 

        # jss *= 0.5

    data = {
        'model1': args.model1,
        'model2': args.model2,
        'seq_length': args.seq_length,
        'num_samples': args.num_samples,
        'kls': None if args.metric != 'kl' else kls.tolist(),
        'jss': None if args.metric != 'js' else jss.tolist(),
        'ent': None if args.metric != 'ent' else (-p_logsums).tolist(),
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
    main(parse_args())
