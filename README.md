# NCE ensemble water

## Environment

```
# check out the virtual env in anaconda 

conda info -e

# create virtual env "gc"

conda create -n gc -y python=3.6 jupyter

# activate virtual env named "gc"：

source activate gc

#install all the requirements in the vitual env

pip install -r requirements.txt

# add the kernal in anaconda python3.6 to jupyter notebook

https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook

# activate env "gc"
conda install nb_conda

# then the "gc" kernal will be added to jupyter notebook


# if you do not use virtual env (fix pip install `s permission denied problem):
# pip install --user [package]

```
 - Download: [data,pkl,npy](https://pan.baidu.com/s/1FhLJy5sCUykrdG14X8QljA)   
 - password: ukh2
 
```
.
+-- data
|   +-- xxx.csv
+-- examples
|   +-- xxx.py
+-- gcforest
|   +-- __init__.py
|   +-- ...
+-- _notebook
|   +-- .ipynb
+-- pkl
|   +-- xx.pkl
+-- npy
|   +-- xx.npy
+-- img
|   +-- xx.eps
+-- README.md
+-- requirements.txt
```