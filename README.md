# A Basic Look into Knowledge Distilling

|     NAME     | Student ID |
| :----------: | :--------: |
| Puquan Chen  |  z5405329  |
|   Song Lin   |  z5362555  |
|  Tianyu Pan  |  z5150836  |
| Zheyuan Shao |  z5334189  |
|  Haoyu Zang  |  z5326339  |



## Install environment

1. `conda create -n knowledge_distilling python=3.8`
2. `conda activate knowledge_distilling`
3. `git clone repo`
4. `cd repo`
5. `pip install -r requirements.txt`
6. `jupyter lab`

## Experimentation

- All Output files of the 2nd and the 3rd experiment can be accessed from this [link](https://drive.google.com/drive/folders/1K2OIiy2jOkLnDX9FfJmxFNWM8AQbO3ZF?usp=sharing), including saved models and csv files. 

####  MNIST - Image Classification - Fully Connected Neural Network (report section 4.1)
1. Run the code with command: `python3 experiment_4.1.py`.
2. You can try different distillation temperatures by editting line 21 on `experiment_4.1.py`.
3. The gereated output files for this experiment are stored in folder `experiment1_models`.
4. We did not plot any figures for this experiment.


#### CIFAR10 - Image Classification - ResNet (report section 4.2)
1. You can select the notebook file `experiment_4.2.ipynb`to run the code.
2. We have tested our notebooks both on Google Colab and local environment, but you may need to edit some codes for saving files.
3. The code will gererate multiple csv files to record training loss, training accuracy, testing loss and testing accuracy. We stored them in folder `data_record`.
4. You may need to [download](https://drive.google.com/file/d/17zYxp_FfcVrkRd3UJb-uakfc4ddME0gA/view?usp=sharing) our ResNet50 teacher model.
5. Run `python3 plot_resnet_distill.py` to get plots for this experiment. The plots are stored in the folder `figures`.

#### CIFAR10 - Image Classification - VGG16 (report section 4.3)
1. You can select the notebook file `experiment_4.3.ipynb`to run the code.
2. We have tested our notebooks both on Google Colab and local environment, but you may need to edit some codes for saving files.
3. The code will gererate multiple csv files to record training loss, training accuracy, testing loss and testing accuracy. We stored them in folder `data_record`.
4. You may need to [download](https://drive.google.com/file/d/1--RIvxWSFyx6MFc4HGIt96mLGDD19fC_/view?usp=sharing) our VGG16 teacher model.
5. Run `python3 plot_vgg_distill.py`and  `python3 plot_best.py` to get plots for this experiment. The plots are stored in the folder `figures`.


