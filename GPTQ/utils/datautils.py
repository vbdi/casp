import numpy as np
import torch
import os
import av
from tqdm import tqdm
import json
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


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

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = torch.load(model)['tokenizer']
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = torch.load(model)['tokenizer']
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
def get_LLaVAVideo178K(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import LlavaNextVideoProcessor
    with open('/Datasets/LLaVA-Video-178K/0_30_s_academic_v0_1_cap_processed.json') as f:
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
        if len(trainloader)==128:
            break
    return trainloader, None


def get_LLaVA_OneVision_Data(nsamples, seed, seqlen, model):
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
    return trainloader, None

def get_LLaVA_Instruct_150K(nsamples, seed, seqlen, model):
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
    return trainloader, None

def get_videochatgpt(nsamples, seed, seqlen, model):
    split='Generic'
    from datasets import load_dataset
    from transformers import LlavaNextVideoProcessor
    # videos=glog.glob('/Datasets/VideoChatGPT/Test_Videos/*.mp4').split('/')[-1].split('.')[-2]
    # trainvideos=videos[:100]
    # testvideos=videos[100:]
    data =  load_dataset('/Datasets/VideoChatGPT', split)['test']
    ii=0
    videos=[]
    prompts=[]
    trainvideos=[]
    trainloader=[]
    for ii in tqdm(range(len(data))):
        ii = int(ii)
        if data[ii]['video_name'] in trainvideos:
            continue
        trainvideos.append(data[ii]['video_name'])
        video_path='/Datasets/VideoChatGPT/Test_Videos/'+data[ii]['video_name']+'.mp4'
        print(video_path)
        if not os.path.isfile(video_path):
            continue
            # video_path='/Datasets/VideoChatGPT/Test_Videos/'+data[ii]['video_name']+'.mkv'

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames>16:
            stride=total_frames / 16
        else:
            stride=1
            
        indices = np.arange(0, total_frames, stride).astype(int)
        print(total_frames,stride,indices)
        # print('Total # of Frames:',len(indices))
        video = read_video_pyav(container, indices)

        if split=='Consistency':
            if data[ii]['question_1']=='None':
                question=data[ii]['question_2']
            elif data[ii]['question_2']=='None':
                question=data[ii]['question_1']  
        else:
            question=data[ii]['question']
        answer=data[ii]['answer']
        # # Tokenize the answer
        prompt="<video> \n"+question+answer
        processor = LlavaNextVideoProcessor.from_pretrained('LLaVA-NeXT-Video-7B-hf') 
        sample = processor(text=prompt,videos=video,return_tensors="pt").to("cuda")
        trainloader.append(sample)
        if len(trainloader)>2:
            break
    with open('/Datasets/VideoChatGPT/train_videos.json','w') as f:
        json.dump(trainvideos,f)

    return trainloader, None

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False)
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False)

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = torch.load(model)['tokenizer']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = torch.load(model)['tokenizer']
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = torch.load(model)['tokenizer']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)
    if 'videochatgpt' in name:
        return get_videochatgpt(nsamples, seed, seqlen, model)

    if 'LLaVAVideo178K' in name:
        return get_LLaVAVideo178K(nsamples, seed, seqlen, model)

    if 'LLaVA-OneVision-Data' in name:
        return get_LLaVA_OneVision_Data(nsamples, seed, seqlen, model)

    if 'LLaVA-Instruct-150K' in name:
        return get_LLaVA_Instruct_150K(nsamples, seed, seqlen, model)
