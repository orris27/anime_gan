## Usage
1. Prepare images in data/faces folder, like 'data/faces/xx.jpg'. Dataset can be downloaded from [this](https://github.com/chenyuntc/pytorch-book/tree/master/chapter7-GAN%E7%94%9F%E6%88%90%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F)
```
ls
#-------------------------------------------------------------------------
# checkpoints/  data/  eval.py  model.py  README.md  train.py
#-------------------------------------------------------------------------

```
2. train
```
python train.py
```
3. eval
```
python eval.py
```

## Experience
I used to use
```
D_loss = - torch.mean(torch.log(real_pred)) - torch.mean(torch.log(1 - fake_pred))
G_loss = torch.mean(torch.log(1 - fake_pred))
```
as the loss of D and G, but they are incorrect with output of noises images.

So I changed to BCELoss and it works.
