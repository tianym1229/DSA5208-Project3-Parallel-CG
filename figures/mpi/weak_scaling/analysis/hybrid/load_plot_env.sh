#!/bin/bash

module load Python/3.13.1-GCCcore-14.2.0
module load matplotlib/3.10.3-gfbf-2025a

python3 -c "import numpy, matplotlib; print('NumPy:', numpy.__version__, '| Matplotlib:', matplotlib.__version__)"
