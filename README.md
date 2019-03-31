## Usage
1. Prepare images in data/faces folder, like 'data/faces/xx.jpg'.
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
