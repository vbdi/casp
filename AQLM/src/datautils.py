import os
import random
from itertools import chain
from typing import Optional, Sequence

import numpy as np
import torch
import torch.distributed
from datasets import load_dataset
from torch import nn
from tqdm import trange
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        # traindata = load_dataset(
        #     "allenai/c4",
        #     "default",
        #     data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        #     split="train",
        #     revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        # )
        traindata = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train',
        )

        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    else:
        # valdata = load_dataset(
        #     "allenai/c4",
        #     "default",
        #     data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        #     split="validation",
        #     revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        # )
        valdata = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        # traindata = load_dataset(
        #     "allenai/c4",
        #     "default",
        #     data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        #     split="train",
        #     revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        # )
        traindata = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train',
        )

        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        # valdata = load_dataset(
        #     "allenai/c4",
        #     "default",
        #     data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        #     split="validation",
        #     revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        # )
        valdata = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation'
        )

        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        return valenc


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def get_LLaVA_OneVision_Data(nsamples, seqlen):
    from datasets import load_dataset
    from transformers import AutoProcessor
    data=load_dataset('/fastdata/share/LLaVA-OneVision-Data/sharegpt4v_llava')['train']
    trainloader=[]
    for ii in tqdm(range(len(data))):    
        ii = int(ii)
        question=data[ii]['conversations'][0]['value']
        answer=data[ii]['conversations'][1]['value']
        prompt="USER:"+question+' ASSISTANT: '+answer
        image=data[ii]['image']

        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        inp = processor(images=image, text=prompt, return_tensors="pt",truncation=True, max_length=256)

        if inp['input_ids'].shape[1]!=256:
            continue
        
        trainloader.append(inp)
        print(len(trainloader),prompt)
        if len(trainloader)==nsamples:
            break
    return trainloader

def get_LLaVAVideo178K(nsamples, seqlen, eval_mode=False):
    import json
    import av
    from datasets import load_dataset
    from transformers import LlavaNextVideoProcessor
    with open('Datasets/LLaVA-Video-178K/0_30_s_academic_v0_1_cap_processed.json') as f:
        data=json.load(f)
    trainloader=[]
    for ii in tqdm(range(len(data))):    
        ii = int(ii)

        question=data[ii]['conversations'][0]['value'].replace('<image>',' <video> ')
        answer=data[ii]['conversations'][1]['value']
        prompt="USER:"+question+' ASSISTANT: '+answer
        video_path='/Datasets/LLaVA-Video-178K/'+data[ii]['video']    
        
        # skip if there if video files is not downloaded or text is short
        if not os.path.isfile(video_path) or len(prompt)<1300:
            continue   

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames>8:
            stride=total_frames / 8
        else:
            stride=1
            
        indices = np.arange(0, total_frames, stride).astype(int)
        video = read_video_pyav(container, indices)

        processor = LlavaNextVideoProcessor.from_pretrained('LLaVA-NeXT-Video-7B-hf') 
        
        inp = processor(text=prompt,videos=video,return_tensors="pt",truncation=True, max_length=250)
        if inp['input_ids'].shape[1]!=250:
            continue
        
        trainloader.append(inp)
        print(len(trainloader),prompt)
        if len(trainloader)==nsamples:
            break
    return trainloader


def get_LLaVA_Instruct_150K(nsamples, seqlen):
    ### Seq len with 256 text tokens would be 831: 575 (vision tokens) + 256 (text tokens)
    from datasets import load_dataset
    from transformers import AutoProcessor
    import json
    from PIL import Image
    file=open('/fastdata/share/LLaVA-Instruct-150K/llava_v1_5_mix665k.json')
    data=json.load(file)
    trainloader=[]
    for idx in tqdm(range(len(data))):    
        idx = int(idx)
        image = Image.open(f"/fastdata/share/LLaVA-Instruct-150K/{data[idx]['image']}")
        conversation=data[idx]['conversations']
        prompt='USER: <image>\n'+conversation[0]['value'].replace('<image>','').replace('\n','') #USER: 
        for ii in range(1,len(conversation)):
            if conversation[ii]['from']=='human':
                prompt+=" USER: "+conversation[ii]['value'] #
            elif conversation[ii]['from']=='gpt':
                prompt+=" ASSISTANT: "+conversation[ii]['value'] #

        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        inp = processor(images=image, text=prompt, return_tensors="pt",truncation=True, max_length=256)

        if inp['input_ids'].shape[1]!=256:
            continue
        
        trainloader.append(inp)
        print(len(trainloader),prompt)
        if len(trainloader)==nsamples:
            break
    return trainloader

def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
        use_fast_tokenizer: whether to use fast tokenizer
        trust_remote_code: whether to trust remote code
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
            )
        except:
            tokenizer = torch.load(model_path)['tokenizer']

        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name == "LLaVAVideo178K":
            data = get_LLaVAVideo178K(nsamples, seqlen, eval_mode=eval_mode)
        elif name == 'LLaVA-Instruct-150K':
            data = get_LLaVA_Instruct_150K(nsamples, seqlen,)
        elif name == 'LLaVA-OneVision-Data':
            return get_LLaVA_OneVision_Data(nsamples, seqlen)
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data


def split_long_texts(inputs: Sequence[str], split_max_length: int):
    """Split examples that exceed split_max_length into multiple sub-examples"""
    outputs = []
    for index, input_str in enumerate(inputs):
        while True:
            truncation_index = input_str.find("\n", split_max_length)
            if truncation_index == -1:
                outputs.append(input_str)
                break
            outputs.append(input_str[:truncation_index])
            input_str = input_str[truncation_index + 1 :]  # continue after \n
    return outputs


def group_texts(examples: Sequence[Sequence[int]], block_size: int, add_labels: bool = True):
    """Group tokenized examples together and split them into blocks of up to block_size tokens"""
    # based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }
    if add_labels:
        result["labels"] = result["input_ids"].copy()
    return result


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device, amp_dtype: Optional[torch.dtype] = None
) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    inps = [
        data[:, start : start + seqlen] for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
    ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    # pad sequences to be divisible by world_size for DDP/FSDP compatibility
    num_padding_sequences = -len(inps) % world_size
    inps.extend([inps[-1]] * num_padding_sequences)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity") if rank == 0 else inps):
        if sequence_index % world_size != rank:
            continue
        input_ids = input_ids.to(device)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            lm_logits = model(input_ids).logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()

    if world_size > 1:
        torch.distributed.all_reduce(total_nll_and_tokens, op=torch.distributed.ReduceOp.SUM)
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()
