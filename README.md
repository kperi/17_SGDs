# Intro 

This repo contains the code for the paper
[Integrating the 17 SDGs into the European Green Deal, through Strategic and Financial Approaches](https://www.researchsquare.com/article/rs-2697240/v1).
 
### Scripts 

- train.py  : code to train the model 
```bash 
python train.py --device=cuda --lr=1e5
```
- example.ipynb : loading a model and doing inference 



### dependencies 

`pip install torch pandas transformers numpy matplotlib fire` 