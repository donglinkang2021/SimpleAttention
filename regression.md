# Regression

- heads model

```python
class Heads_Reg(BaseModel):
    """Regression model with multi-head between-heads attention"""
    def __init__(self, input_dim, n_embd, n_head, output_dim):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.embed = nn.Linear(input_dim, n_embd)
        self.pred = nn.Linear(n_embd, output_dim)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")
    
    def get_wei(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        wei = x @ x.transpose(-2, -1) / math.sqrt(self.n_embd)
        wei = F.softmax(wei, dim=-1)
        return wei

    def forward(self, x):
        x = self.embed(x)
        x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
        attention = F.scaled_dot_product_attention(x, x, x)
        attention = rearrange(attention, 'B nh hs -> B (nh hs)')
        return self.pred(attention)
```

- both MLP model and heads model has the same number of parameters

|n_embd|number of parameters|
|:---:|:---:|
|4|17|
|8|33|
|16|65|
|32|129|
|64|257|

## Regress Plane

- train config

```python
train_samples = 500
val_samples = 500
noise = 0.2 # a good para to test the model performance

batch_size = 64
num_epochs = 3
eval_interval = 1
learning_rate = 1e-2
```

- model config

```python
input_dim = 2 
n_embd = 8 # n_embd:param 8:33 16:65 32:129 64:257
n_head = 4
output_dim = 1
```

## Regress Gaussian

- train config

```python
train_samples = 500
val_samples = 500
noise = 0.01 # a good para to test the model performance

batch_size = 64
num_epochs = 30
eval_interval = 10
learning_rate = 3e-2
```

- model config

```python
input_dim = 2 
n_embd = 8 # n_embd:param 8:33 16:65 32:129 64:257
n_head = 4
output_dim = 1
```