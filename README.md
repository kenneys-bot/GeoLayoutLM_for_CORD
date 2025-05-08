# GeoLayoutLM for CORD
The official PyTorch implementation of GeoLayoutLM (CVPR 2023 highlight).
## Preprocess
```
python preprocess_cord.py \
--inputpath /path/to/cord_data_path \
--outputpath /path/to/output_cord_geo_path \
--voca /path/to/tokenizer_path \
--model_type [model type] 
```

If you need CORD's dev dataset, here is another code:

``````
python preprocess_cord_v2.py \
--inputpath /path/to/cord_data_path \
--outputpath /path/to/output_cord_geo_path \
--voca /path/to/tokenizer_path \
--model_type [model type]
``````

Then, modify the code at the specified location below:

```
# geolayoutlm/utils/__init__.py: line51 cfg.model.n_classes = 2 * 22 + 1 -->
cfg.model.n_classes = 2 * 30 + 1
```
## Train(option)

If you want to use my train code, here is my python environment:

Python 3.8

Torch 1.8.1+cu111

Torchvision 0.9.1+cu111

**Install environment **

``````
pip install -r requriements.txt -c constraints.txt
``````

### requriements.txt

``````
nptyping==1.4.2
numpy
opencv-python-headless==4.5.4.60
pytorch-lightning==1.5.10
omegaconf
Pillow
six
tqdm
opencv-python==4.5.5.62
overrides==3.1.0
transformers
seqeval==1.2.2
imagesize
timm
tensorboard>=2.2.0
isort==5.9.3
black==21.9b0
``````

### constraints.txt

``````
torch==1.8.1+cu111
torchvision==0.9.1+cu111
``````

In order to migrate the GeoLayoutLM code base to the latest version, I have modified the library call declarations of some codes. If you have any questions, please submit an issue.

If you have any questions about the code implementation, please go to the original author to submit an issue:https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM