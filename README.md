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
License and Citation
Images and data from the Human Protein Atlas are used with permission under the CC BY-NC-ND 4.0 License.

The Cell DECODER toolbox is released under the [MIT License](https://opensource.org/licenses/MIT).

If you use Cell DECODER for your research, please cite:
Nirschl JJ, Moore AM, Holzbaur ELF (YYYY). Cell DECODER: A Deep Learning
Package for Phenotypic Profiling in Biological Image Analysis.

Bibtex formatted reference:
@article{nirschl2017decoder,
    Author={Jeffrey J Nirschl and Andrew S Moore AM and  Erika LF Holzbaur},
    Journal={},
    Title={Cell DECODER: A Deep Learning Package for Phenotypic Profiling in Biological Image Analysis},
    Year{2017},
}


------------------
## Acknowledgements
The authors gratefully acknowledge the NVIDIA Corporation's Academic Hardware Grant of a Titan-X GPU. Jeff Nirschl was supported by NINDS F30-NS092227.

The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
