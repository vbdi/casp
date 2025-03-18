import argparse
import json
import math
import os
import random
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import datasets
import glog
import torch
from tqdm import tqdm

from lib.utils import gptq_data_utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llava_next_video import LlavaNextVideoForConditionalGeneration
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='Llama-2-7b-E8P-2Bit', type=str)
parser.add_argument('--seqlen', default=4096, type=int)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    datasets = ['c4', 'wikitext2' ]
    if 'llava' in args.hf_path.lower():
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(args.hf_path).cuda()
        model_str=args.hf_path
        print('@@@@@@@@@@@@@@@@@@',model_str)
    else:
        model, model_str = model_from_hf_path(
            args.hf_path,
            use_cuda_graph=not args.no_use_cuda_graph,
            use_flash_attn=not args.no_use_flash_attn)
        model_str=args.hf_path
        print('@@@@@@@@@@@@@@@@@@',model_str)

    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
            nsamples, args.seqlen)

        if not args.no_use_cuda_graph:
            model.reset()

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                           use_cache=False,
                           output_hidden_states=False,
                           output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            # print(loss.item())
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        glog.info(f'{dataset} perplexity: {ppl}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
