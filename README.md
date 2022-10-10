#Instruction

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

2. Specify '--dir_data' based on the HR and LR images path. 



### Begin to train

1. Cd to 'src', run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python3 main.py --chop --batch_size 16 --model DNLN --scale 2 --patch_size 96 --save DNLN_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models

    ```

## Test
### Quick start
1. Download benchmark datasets.

2. Cd to 'src', run the following scripts.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python3 main.py --model DNLN --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --n_feats 128 --depth 12 --pre_train ../models/model_x2.pt --save_results --test_only --chop
    ```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch). We thank the authors for sharing their codes.

