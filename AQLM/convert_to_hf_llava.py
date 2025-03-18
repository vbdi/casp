import json
import os
import re
import shutil

import torch
from tqdm.auto import trange
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import safetensors
except ModuleNotFoundError:
    safetensors = None


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def pack_int_data(data: torch.IntTensor, nbits: int) -> torch.IntTensor:
    data[data >= 2 ** (nbits - 1)] -= 2**nbits
    # print('$$$$',(data >= 2 ** (nbits - 1)).sum())
    return data.to(get_int_dtype(nbits))


def get_num_layers(config) -> int:
    if config.model_type in ["llama" , "mistral" , "mixtral" , "gemma" , "phi3" , "qwen2"]:
            return config.num_hidden_layers
    else:
        raise NotImplementedError(f"Can't get number of layers for {config.model_type}")


def get_layers_prefix(config) -> str:
    if config.model_type in ["llama" , "mistral" , "mixtral" , "gemma" , "phi3" , "qwen2"]:
        return "model.layers"
    else:
        raise NotImplementedError(f"Can't get layers prefix for {config.model_type}")


def get_converted_state_dict(config, nbits: int, in_path: os.PathLike,args) -> [dict, list[str]]:
    state_dict = {}
    linear_weights_not_to_quantize = []

    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)

    for i in trange(num_layers):
        layer = torch.load(os.path.join(in_path, f"{i}.pth"))
        for name, p in layer.named_parameters():
            if args.svd:
                if '7b' in args.in_path.lower():
                    if 'k_u_proj' in name or 'k_v_proj' in name or 'q_u_proj' in name or 'q_v_proj' in name or i<7 or i==31:
                        nbits=16
                    else: 
                        nbits=16
            # print(f'###### layer{i}{name},nbits:{nbits}')
            if torch.is_floating_point(p.data):
                p.data = p.data.half()
                # print('@@@',p.data.shape)
            else:
                p.data = pack_int_data(p.data, nbits)
                # print('!!!',p.data.shape)
            if "quantized_weight." not in name:
                linear_weights_not_to_quantize.append(f"{layers_prefix}.{i}.{name}")
            else:
                name = re.sub("quantized_weight.", "", name)
            # print(f"########{layers_prefix}.{i}.{name}")
            state_dict[f"language_model.{layers_prefix}.{i}.{name}"] = p.data

    for key, value in torch.load(os.path.join(in_path, "not_quantized_weights.pt")).items():
        state_dict[key] = value.half()
        linear_weights_not_to_quantize.append(key)

    if "lm_head.weight" not in linear_weights_not_to_quantize:
        linear_weights_not_to_quantize.append("lm_head.weight")

    return state_dict, linear_weights_not_to_quantize


def get_metadata(in_path: os.PathLike) -> dict:
    quant_args = torch.load(os.path.join(in_path, "args.pt"))
    return {
        "nbits_per_codebook": quant_args["nbits_per_codebook"],
        "num_codebooks": quant_args["num_codebooks"],
        "out_group_size": quant_args["out_group_size"],
        "in_group_size": quant_args["in_group_size"],
    }


def update_config(config_dict: dict, aqlm_metadata: dict[str, int], linear_weights_not_to_quantize: list[str]):
    config_dict["quantization_config"] = {
        "quant_method": "aqlm",
        "nbits_per_codebook": aqlm_metadata["nbits_per_codebook"],
        "num_codebooks": aqlm_metadata["num_codebooks"],
        "out_group_size": aqlm_metadata["out_group_size"],
        "in_group_size": aqlm_metadata["in_group_size"],
        "linear_weights_not_to_quantize": linear_weights_not_to_quantize,
    }
    config_dict["torch_dtype"] = "float16"
    return config_dict

def update_config_svd(config_dict: dict, aqlm_metadata: dict[str, int], linear_weights_not_to_quantize: list[str]):
    config_dict["text_config"]["quantization_config"] = {
        "quant_method": "aqlm",
        "linear_weights_not_to_quantize": linear_weights_not_to_quantize,
    }

    # for ii in range(32):
    #     config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_qk_{ii}"]=8
    #     config_dict["text_config"]["quantization_config"][f"num_codebooks_qk_{ii}"]=2
    #     config_dict["text_config"]["quantization_config"][f"out_group_size_qk_{ii}"]=1
    #     config_dict["text_config"]["quantization_config"][f"in_group_size_qk_{ii}"]=8

    #     if ii<0 or ii==33:
    #         config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_ov_{ii}"]=12
    #     else:
    #         config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_ov_{ii}"]=8
    #     config_dict["text_config"]["quantization_config"][f"num_codebooks_ov_{ii}"]=2
    #     config_dict["text_config"]["quantization_config"][f"out_group_size_ov_{ii}"]=1
    #     config_dict["text_config"]["quantization_config"][f"in_group_size_ov_{ii}"]=8
        
    #     if ii<0 or ii==33:
    #         config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_mlp_{ii}"]=12
    #     else:
    #         config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_mlp_{ii}"]=8
    #     config_dict["text_config"]["quantization_config"][f"num_codebooks_mlp_{ii}"]=2
    #     config_dict["text_config"]["quantization_config"][f"out_group_size_mlp_{ii}"]=1
    #     config_dict["text_config"]["quantization_config"][f"in_group_size_mlp_{ii}"]=8


    for ii in range(32):
        config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_qk_{ii}"]=16
        config_dict["text_config"]["quantization_config"][f"num_codebooks_qk_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"out_group_size_qk_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"in_group_size_qk_{ii}"]=8

        config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_ov_{ii}"]=16
        config_dict["text_config"]["quantization_config"][f"num_codebooks_ov_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"out_group_size_ov_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"in_group_size_ov_{ii}"]=8
        

        config_dict["text_config"]["quantization_config"][f"nbits_per_codebook_mlp_{ii}"]=16
        config_dict["text_config"]["quantization_config"][f"num_codebooks_mlp_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"out_group_size_mlp_{ii}"]=1
        config_dict["text_config"]["quantization_config"][f"in_group_size_mlp_{ii}"]=8

    config_dict["torch_dtype"] = "float16"
    return config_dict

def add_inference_code(model_type: str, save_path: os.PathLike):
    if os.path.isdir(f"./transformers/{model_type}"):
        shutil.copytree(f"./transformers/{model_type}", save_path, dirs_exist_ok=True)
    else:
        print(f"No predefined PreTrainedModel exists for {model_type}. You'll have to copy-paste some code yourself.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model",
        type=str,
        help="Path to the model to base config on, as in AutoConfig.from_pretrained()",
    )
    parser.add_argument(
        "in_path",
        type=str,
        help="Path of the checkpoint to convert",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Path to save HF compatible checkpoint to",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Whether to save in safetensors format",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Whether to load model",
    )
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        help="Whether to save tokenizer",
    )
    parser.add_argument(
        "--svd",
        action="store_true",
        help="Whether attention is compressed with svd",
    )
    args = parser.parse_args()

    if  "llava-next-video" in args.model.lower() or "llava_next_video" in args.model.lower(): 
        if args.svd:
            from model.llava_next_video_svd import LlavaNextVideoForConditionalGeneration
            model = torch.load(args.model)['model'].half()
        else:
            
            from transformers import LlavaNextVideoForConditionalGeneration
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model, 
                                                                        torch_dtype=torch.float16, 
                                                                        use_flash_attention_2=False,
                                                                        low_cpu_mem_usage=True,
                                                                        )

    elif  "llava-1.5" in args.model.lower() or "llava_1.5" in args.model.lower():
        from transformers import AutoProcessor
        if args.svd:
            from component.llava import LlavaForConditionalGeneration
            model = torch.load(args.model)['model'].half()
        else:
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(args.model,
                                                                torch_dtype=torch.float16, 
                                                                use_flash_attention_2=False,
                                                                low_cpu_mem_usage=True,)
                                                                    
    elif  "llava-v1.6" in args.model.lower() or "llava_v1.6" in args.model.lower():
        from transformers import AutoProcessor
        if args.svd:
            from component.llava import LlavaNextForConditionalGeneration
            model = torch.load(args.model)['model'].half()
        else:
            from transformers import LlavaNextForConditionalGeneration
            model = LlavaNextForConditionalGeneration.from_pretrained(args.model,
                                                                torch_dtype=torch.float16, 
                                                                use_flash_attention_2=False,
                                                                low_cpu_mem_usage=True,)                                                            

    # print(old_config)
    # tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path, use_fast=True)

    # old_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    metadata = get_metadata(args.in_path)
    old_config=model.config 

    # load dummy model
    if args.load_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )

    state_dict, linear_weights_not_to_quantize = get_converted_state_dict(
        old_config.text_config, metadata["nbits_per_codebook"], args.in_path,args)
    # print(state_dict)
    torch.save(state_dict, os.path.join(args.out_path, "pytorch_model.bin"))
    
    if args.svd:
        new_config_dict = update_config_svd(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)
    else:
        new_config_dict = update_config_svd(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)
        # new_config_dict = update_config(old_config.to_diff_dict(), metadata, linear_weights_not_to_quantize)

    # print(new_config_dict)
    with open(os.path.join(args.out_path, "config.json"), "w") as config_file:
        json.dump(new_config_dict, config_file, indent=4)
    # print(model)
    # convert to safetensors
    if args.save_safetensors:
        assert safetensors
        if  "llava-next-video" in args.model.lower() or "llava_next_video" in args.model.lower(): 
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16)
            # print(model)
        elif  "llava-1.5" in args.model.lower() or "llava_1.5" in args.model.lower():     
            from transformers import AutoProcessor
            if args.svd:
                from component.llava import LlavaForConditionalGeneration  
            model = LlavaForConditionalGeneration.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16,ignore_mismatched_sizes=True) 
            # print(model.language_model.model.layers[0].self_attn.q_proj.num_codebooks)
        elif  "llava-v1.6" in args.model.lower() or "llava_v1.6" in args.model.lower():     
            from transformers import AutoProcessor
            if args.svd:
                from component.llava import LlavaNextForConditionalGeneration  
            model = LlavaNextForConditionalGeneration.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16,ignore_mismatched_sizes=True) 
            print(model.language_model.model.layers[0].self_attn.q_proj.num_codebooks)
        # model = AutoModelForCausalLM.from_pretrained(args.out_path, trust_remote_code=True, torch_dtype=torch.float16)
        # shutil.rmtree(args.out_path)
        model.save_pretrained(args.out_path)
        

    if args.save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(args.out_path)
