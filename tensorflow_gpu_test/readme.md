# Train your own image classificator for rock paper and scissors game

### i used this dataset to train this classificator  from a video in tensorflow youtube channel

### link of the video 
https://www.youtube.com/watch?v=03NSQ7xQIWQ
### follow the colab
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb

### to run this code locally i used anaconda for easy installation of CUDA and cuDNN and i follow this steps:
1.- download and install  anaconda from: 
https://www.anaconda.com/distribution/

2.- open anaconda prompt:

3.- create a new environment and activate it: 

conda create -n tf-gpu 
conda activate tf-gpu 

4.- install a python kernel:

pip install ipykernel
python -m ipykernel install --user --name tf-gpu --display-name "tf-gpu"

5.- install tensorflow-gpu (it can take a few minutes): 

conda install tensorflow-gpu
