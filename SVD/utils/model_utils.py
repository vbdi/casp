
import os
import sys
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


dev = torch.device("cuda")

def get_model_from_huggingface(model_id,seq_len):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlavaNextVideoForConditionalGeneration, LlamaForCausalLM 
    

    tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    if  "llava-next-video" in model_id.lower():
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_id, device_map="cpu",torch_dtype=torch.float16,attn_implementation="eager")
    elif  "llava-1.5" in model_id.lower():
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="cpu",torch_dtype=torch.float16,attn_implementation="eager")
    else:
        model = LlamaForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None,attn_implementation="eager")
    model.seqlen = seq_len
    return model, tokenizer

def get_model_from_local(model_id):
    print('###################',model_id)
    pruned_dict = torch.load(model_id, map_location='cpu')
    print('#################### Loaded')
    # 
    try:
        tokenizer =  pruned_dict['tokenizer']
        model = pruned_dict['model']
    except:
        from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
        model = pruned_dict
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res