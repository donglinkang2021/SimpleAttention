# README

Attention Here❗❗❗ This repository focus on how to use attention mechanism to apply on some simple task like regression and classification, hope to grasp some intuitions.

- run on CPU

```bash
pip install torch einops
```

## Datasets

- regression: gaussian, plane

<table>
    <tr>
        <td colspan="1" align=center>
            <img src="images\regression_plane.png" width="300px" alt="regression_plane">
        </td>
        <td colspan="1" align=center>
            <img src="images\regression_gaussian.png" width="300px" alt="regression_sin">
        </td>
    </tr>    
</table>

- classification: circle, spiral, two_gaussians, xor

<table>
    <tr>
        <td colspan="1" align=center>
            <img src="images\classification_circle.png" width="300px" alt="classification_circle">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_spiral.png" width="300px" alt="classification_spiral">
        </td>
    </tr>    
    <tr>
        <td colspan="1" align=center>
            <img src="images\classification_two_gaussians.png" width="300px" alt="classification_two_gaussians">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_xor.png" width="300px" alt="classification_xor">
        </td>
    </tr>
</table>

## Model

- take regress gaussian as example
- MLP: 2 hidden layer, 32 hidden units (105 para), tanh activation 
- Heads: 1 hidden layer, 8 hidden units (**33** para), **no activation** 

| Model | Final Output | Hidden |
| :---: | :---: | :---: |
| MLP | ![MLP](./visualize_train/linear/fin_16.gif) | ![MLP](./visualize_train/linear/hid_16.gif) |
| Heads | ![Heads](./visualize_train/heads/fin_16.gif) | ![Heads](./visualize_train/heads/hid_16.gif) |