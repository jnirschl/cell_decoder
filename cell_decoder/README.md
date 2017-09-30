# Cell DECODER
## Cell DECODER: Cell DEep learning and COmputational DEscriptor toolbox.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](TODO--add_url/blob/master/LICENSE)

------------------
## Introduction
The Cell DEep learning and COmputational DEscriptoR (DECODER) toolbox is an API, written in Python, for training and applying neural networks to biological datasets. Our goal is to provide an easy interface to models pre-trained on biological datasets to improve image analysis and phenotypic profiling in biology. It is currently built on the [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/CNTK), but can be extended to support other deep learning packages.

------------------
## Installation
Cell DECODER requires CNTK as a backend engine. Please follow the [CNTK installation instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine).

Recommended dependencies:
* cuDNN 6 (GPU support)
* Holoviews
* Bokeh

Install from PyPi (todo)
```
sudo pip install cell_decoder
```

------------------

## Usage

### Deep feature extraction
```python
# import relevant modules
from cell_decoder import extract 
from cell_decoder.model_zoo import deep_cell_Res50

# Set user mapfile 
# Tab separated file with list of  image filepaths and labels
map_file = [PATH_TO_MAPFILE]
output_file = [PATH_TO_SAVE_OUTPUT]

cell_decoder.extract_features(deep_cell_Res50, map_file,
                              output_file)
```
------------------
## Acknowledgements
The authors gratefully acknowledge the NVIDIA Corporation's Academic Hardware Grant of a Titan-X GPU. Jeff Nirschl was supported by NINDS F30-NS092227.

The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
