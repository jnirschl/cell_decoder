[<img src="https://raw.githubusercontent.com/jnirschl/cell_decoder/master/logo_lg.png">](https://github.com/jnirschl/cell_decoder)

# Cell DECODER: Cell DEep learning and COmputational DEscriptor toolbox.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](TODO--add_url/blob/master/LICENSE)

------------------
## Contents
* [Introduction](#introduction)
* [Installation guide](#installation-guide)
* [Example usage](#example-usage)
  * [Deep feature extraction](#deep-feature-extraction)
  * [Expected output](#expected-output)
* [License and citation](#license-and-citation)
* [Acknowledgements](#acknowledgements)

------------------
## Introduction
The Cell DEep learning and COmputational DEscriptoR (DECODER) toolbox is an API, implemented in Python, for training and applying neural networks as feature extractors in biological image datasets. Our goal is to provide an easy interface and models trained on biological datasets to allow phenotypic profiling using deep features. Cell DECODER is currently built on the [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/CNTK), but can be extended to support other deep learning packages. Cell DECODER is free and open-source software distributable under the MIT license.

------------------
## Installation guide
Cell DECODER uses the Python programming language. Previous experience with Python is helpful, but not strictly necessary. The installation time for a user with previous experience in Python user is less than 30 minutes.

### System requirements

||         System requirements           ||
| ----------          | :----------:      |
| Operating system    | Windows 8.1 or 10 |
|                     | Ubuntu 16.04      |
| Processor           | 2 GHz             |
| Memory (RAM)        | 2 GB              |
| Hard-drive space    | 50 MB             |

[Minimal system requirements for installing Cell DECODER. This does not include hard-drive space required for additional software packages or dependencies such as [Anaconda Python](https://conda.io/docs/user-guide/install/index.html) or [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine). A Graphics Processing Unit (GPU) is optional, but will accelerate performance. CUDA 8.0 and cuDNN 6 are required for GPU-acceleration. Refer to the online user manual for your GPU for more instructions.]

## 1. Install Python 3.5
We recommend the Anaconda Python distribution, which allows multiple Python environments with different configurations. Install Python 3.5 and refer to the [Anaconda Python user guide for installation instructions](https://conda.io/docs/user-guide/install/index.html)

## 2. Install CNTK
Cell DECODER uses CNTK as a backend for training and evaluating neural networks. Please follow the [CNTK installation instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine). 


## 3. Install Cell DECODER from PyPI
Install from PyPi
```
pip install cell_decoder
```

Open a Python and test the installation. The version should print during import if Cell DECODER is installed.
```python
>>> import cell_decoder
Running Cell DECODER version 0.1.0
```


## Example usage
Below, we provide example input and output for the deep feature extraction functionality of Cell DECODER. We also provide executable Jupyter Notebooks in the [Examples folder](https://github.com/jnirschl/cell_decoder/tree/master/examples). These notebooks contain tutorial text, source code, and example outputs.

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

### Expected output
```python
# 
cell_decoder.plot(TODO)
```


------------------
## License and Citation
Images and data from the Human Protein Atlas were used with permission under the [CC BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

The Cell DECODER toolbox is released under the [MIT License](https://opensource.org/licenses/MIT).

If you use Cell DECODER for your research, please cite:
Nirschl JJ, Moore AM, Holzbaur ELF. Cell DECODER: A Neral Network Toolbox for Phenotypic Profiling in Cell Biology.

Bibtex formatted reference:
```text
@article{nirschlCellDECODER,
    Author={Jeffrey J Nirschl and Andrew S Moore AM and  Erika LF Holzbaur},
    Journal={},
    Title={Cell DECODER: A Neral Network Toolbox for Phenotypic Profiling in Cell Biology},
    Year{submitted},
}
```

------------------
## Acknowledgements
The authors gratefully acknowledge the NVIDIA Corporation's Academic Hardware Grant of a Titan-X GPU. This research was supported by National Institutes of Health grants NINDS F30-NS092227 to Jeff Nirschl and NIH GM48661 to Erika L. F. Holzbaur.

The content is solely the responsibility of the authors and does not represent the official views of the National Institutes of Health or the NVIDIA corporation. The funders had no role in software development for Cell DECODER, study design, data collection and analysis, decision to publish, or preparation of this manuscript.
