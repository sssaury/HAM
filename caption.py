from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, eval_model2
from llava.utils import disable_torch_init


# Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

# You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

# ``` python
import json
import os
import collections
import time

model_path = './checkpoint'
prompt = 'Describe the person in the image without referring to the background or any additional context. Focus on their physical appearance, including their hairstyle, clothing, shoes, any noticeable accessories they may be wearing, and any belongings they may be carrying.'
disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
	model_path, None, model_name
)

img_root = './PLIP_data/'
with open("./data/random100w_synth_paths.json",'r') as f:
    img_paths = json.load(f)
setA = set(img_paths)
caption_dir = './caption.json'

if os.path.exists(caption_dir):
    with open(caption_dir, 'r') as f:
        data = json.load(f)
    already = collections.defaultdict(list,data)
    already_path = list(already.keys())
else:
    already = collections.defaultdict(list)
    already_path = []
print("already:",len(already_path))
setB = set(already)
result = list(setA.symmetric_difference(setB))
print("remain:",len(result))

for i,name in enumerate(result):
    image_path = img_root + name
    start_time = time.time()
    args = type('Args', (), {
		"model_path": model_path,
		"model_base": None,
		"model_name": get_model_name_from_path(model_path),
		"query": prompt,
		"conv_mode": None,
		"image_file": image_path,
		"sep": ",",
		"temperature": 0,
		"top_p": None,
		"num_beams": 1,
		"max_new_tokens": 512})()
    for k in range(1):
        try:
            cap = eval_model2(args, tokenizer, model, image_processor, context_len, model_name)
            count = 0
            while len(cap.split(" "))>130 or len(cap.split(" "))<8 or '>' in cap or '<' in cap or '\n' in cap or '/n' in cap:
                cap = eval_model2(args, tokenizer, model, image_processor, context_len, model_name)
                count += 1
                if count > 3:
                    break
            already[name].append(cap)
        except Exception as e:
            print(f"-----------------{e}")
            continue
    if i%10 == 0:
        # print(cap)
        print(f'Process {i}/{len(result)} images, take {time.time()-start_time}s each to {caption_dir}')
        with open(caption_dir,'w') as f:
            json.dump(already,f)
with open(caption_dir,'w') as f:
    json.dump(already,f)
print(f'Process {i}/{len(result)} images')

