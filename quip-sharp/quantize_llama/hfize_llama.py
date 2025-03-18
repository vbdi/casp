import argparse
import os
import time

import glog
import torch
from transformers import AutoTokenizer
import os, sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from lib import codebook, utils
from lib.utils.unsafe_import import model_from_hf_path
from model.llama import LlamaForCausalLM
from lib.utils.model_version import MODEL_VERSION

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', type=str)
parser.add_argument('--hf_output_path', type=str)
parser.add_argument('--compressed_model', action='store_true')
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']
    # model_config['_attn_implementation']='svd'
    # model_config.update({"_attn_implementation": "svd"})


    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model_config.quip_params['model_version'] = MODEL_VERSION
    if args.compressed_model:
        attn_implementation="svd"
    else:
        attn_implementation="eager"
    
    # if args.compressed_model:
    #     model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
    #                                          torch_dtype='auto',
    #                                          low_cpu_mem_usage=True,
    #                                          config=model_config,
    #                                          attn_implementation=attn_implementation).half()
    # else:
    # print(model_config)
    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                            torch_dtype='auto',
                                            low_cpu_mem_usage=True,
                                            config=model_config,
                                            ).half()
    print(model)
    # print(model.model.layers[0])
    cpu = torch.device('cpu')
    if os.path.exists(f'{args.quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{args.quantized_path}/lmhead.pt',
                                 map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'])
        model.model.norm.weight.copy_(lmhead_data['norm'])

    for ii in range(len(model.model.layers)):

        codebook_id = codebook.get_id(model_config.quip_params['codebook'])
        codesz = model_config.quip_params['codesz']

        codebook_id_qk = codebook.get_id(model_config.quip_params[f'codebook_qk_{ii}'])
        codesz_qk = model_config.quip_params[f'codesz_qk_{ii}']

        codebook_id_ov = codebook.get_id(model_config.quip_params[f'codebook_ov_{ii}'])
        codesz_ov = model_config.quip_params[f'codesz_ov_{ii}']

        codebook_id_mlp = codebook.get_id(model_config.quip_params[f'codebook_mlp_{ii}'])
        codesz_mlp = model_config.quip_params[f'codesz_mlp_{ii}']

        layer = model.model.layers[ii]

        if os.path.exists(f'{args.quantized_path}/{ii}_layernorm.pt'):
            ln_data = torch.load(f'{args.quantized_path}/{ii}_layernorm.pt',
                                 map_location=cpu)
            layer.input_layernorm.weight.copy_(ln_data['input_layernorm'])
            layer.post_attention_layernorm.weight.copy_(
                ln_data['post_attention_layernorm'])

        if args.compressed_model:
            ## q_u
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_q_u.pt',map_location=cpu)
            # print('@@@@@@@@@@@@@@@@@@@@@@',saved_layer)
            utils.unpack_quip(layer.self_attn.q_u_proj, saved_layer, codebook_id_qk,codesz_qk)
            ## q_v
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_q_v.pt',map_location=cpu)
            utils.unpack_quip(layer.self_attn.q_v_proj, saved_layer, codebook_id_qk,codesz_qk)
            ## k_u
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_k_u.pt',map_location=cpu)
            utils.unpack_quip(layer.self_attn.k_u_proj, saved_layer, codebook_id_qk,codesz_qk)
            ## k_v
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_k_v.pt',map_location=cpu)
            utils.unpack_quip(layer.self_attn.k_v_proj, saved_layer, codebook_id_qk,codesz_qk)
            ## v
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_v.pt',map_location=cpu)
            utils.unpack_quip(layer.self_attn.v_proj, saved_layer, codebook_id_ov,codesz_ov)
        else:
            saved_layer = torch.load(f'{args.quantized_path}/{ii}_qkv.pt',
                                    map_location=cpu)
            for i in range(len(saved_layer['scales'])):
                layer.self_attn.qkv_proj.fuse_scales[i].copy_(
                    saved_layer['scales'][i])
            utils.unpack_quip(layer.self_attn.qkv_proj, saved_layer, codebook_id_qk,
                            codesz_qk)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id_ov,
                          codesz_ov)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt',
                                 map_location=cpu)
        for i in range(len(saved_layer['scales'])):
            layer.mlp.upgate_proj.fuse_scales[i].copy_(
                saved_layer['scales'][i])
        utils.unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id_mlp,
                          codesz_mlp)

        saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt',
                                 map_location=cpu)
        utils.unpack_quip(layer.mlp.down_proj, saved_layer, codebook_id_mlp,
                          codesz_mlp)
        glog.info(f'loaded layer {ii} down')

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(args.hf_output_path, use_cuda_graph=False,use_flash_attn=False)

    print(model)
    glog.info('successfully loaded hfized model')

    glog.info('generating some text...')

    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                             attention_mask=inputs['attention_mask'].cuda(),
                             max_new_tokens=64,
                             return_dict_in_generate=True)
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)
    glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
