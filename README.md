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

Then, modify the code at the specified location below:
```
# geolayoutlm/utils/__init__.py: line51 cfg.model.n_classes = 2 * 22 + 1 -->
cfg.model.n_classes = 2 * 30 + 1
```
In order to migrate the GeoLayoutLM code base to the latest version, I have modified the library call declarations of some codes. If you have any questions, please submit an issue.

If you have any questions about the code implementation, please go to the original author to submit an issue:https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM