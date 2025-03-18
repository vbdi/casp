#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from utils.model_utils import *


current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
        

@torch.no_grad()
def profle_svd(model_name, model, calib_loader, dev):

    layers = model.language_model.model.layers
    model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.to(dev)
    model.language_model.model.norm = model.language_model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    model.to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.language_model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)

            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
 
    model.language_model.model.embed_tokens = model.language_model.model.embed_tokens.cpu()
    model.language_model.model.norm = model.language_model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']

    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2: 
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat
     
 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev,layers_cond):
    model.eval()
    layers = model.language_model.model.layers

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)

        #### Replace Wq and Wk with its low-rank verison ####

        svd_attn = SVD_LlamaAttention(config=model.language_model.config, ratio=ratio[:2],layers_cond=layers_cond,layer_idx=i)

        #### Replace Wq and Wk with its low-rank verison ####

        for name in subset:
            print(f'########## Layer{i}.{name}')

            if "q_proj" in name or "k_proj" in name:
                ratio=ratio
            else:
                continue

            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####

            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        
            svd_attn.q_u_proj.weight.data = svd_u
            svd_attn.q_v_proj.weight.data = svd_v

            svd_attn.k_u_proj.weight.data = svd_u
            svd_attn.k_v_proj.weight.data = svd_v
 

            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', nargs='*',type=float, default=[0.2,0.3,0.3],choices=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,0.8], help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params. Three rations shoud be given for [Q_proj and K_proj], [V_proj,O_proj], and MLPs, respectively.')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--layers_cond', type=str, default=None, help='only attention, only mlp, etc.')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--fused_qk',action='store_true',)
    
    args = parser.parse_args()
    print(args)

    args.ratio = 1- args.ratio

    if args.fused_qk:
        fused_qk='fused_qk'
    else:
        fused_qk=''
    model, tokenizer = get_model_from_huggingface(model_id=args.model,seq_len=args.model_seq_len)
    model = model.eval()
    if args.profiling_mat_path is None:
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        profiling_mat = profle_svd(args.model, model, cali_white_data, args.DEV)
        if args.save_path is not None:
            torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+fused_qk+ '.pt')
    else:
        profiling_mat = torch.load(args.profiling_mat_path)
    whitening(args.model, model, profiling_mat, args.ratio, args.DEV,args.layers_cond)
    if args.save_path is not None:
        torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + args.layers_cond+fused_qk+ '.pt')   # fp32
 