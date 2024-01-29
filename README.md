# README

Attention Here❗❗❗ This repository focus on how to use attention mechanism to apply on some simple task like regression and classification, hope to grasp some intuitions.

- run on CPU

```bash
pip install torch einops
```

## Datasets

- regression: plane, gaussian
- classification: circle, spiral, two_gaussians, xor


<table>
    <tr>
        <th>plane</th>
        <th>gaussian</th>
        <th>circle</th>
        <th>spiral</th>
        <th>two_gaussians</th>
        <th>xor</th>
    </tr>
    <tr>
        <td colspan="1" align=center>
            <img src="images\regression_plane.png" width="300px" alt="regression_plane">
        </td>
        <td colspan="1" align=center>
            <img src="images\regression_gaussian.png" width="300px" alt="regression_sin">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_circle.png" width="300px" alt="classification_circle">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_spiral.png" width="300px" alt="classification_spiral">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_two_gaussians.png" width="300px" alt="classification_two_gaussians">
        </td>
        <td colspan="1" align=center>
            <img src="images\classification_xor.png" width="300px" alt="classification_xor">
        </td>
    </tr>    
</table>


## Model

Here we provide two model, MLP and Heads, the difference is as follows:

- MLP: 2 hidden layer, 8 hidden units (105 para), tanh activation 
- Heads: 1 hidden layer, 8 hidden units (**33** para), **no activation** 

## Visualize

We use the following gif to visualize the training process, the left is the final output, the right is the hidden state.

- We choose 2 relatively diffucult tasks in all 6 task, gaussian and spiral, to visualize the regression or classification training process.
- More details can be found in [visualize_train/README.md](visualize_train/README.md)

### regress_gaussian

| Model | Final Output | Hidden |
| :---: | :---: | :---: |
| MLP | ![MLP](./visualize_train/regression/linear/fin_16.gif) | ![MLP](./visualize_train/regression/linear/hid_16.gif) |
| Heads | ![Heads](./visualize_train/regression/heads/fin_16.gif) | ![Heads](./visualize_train/regression/heads/hid_16.gif) |

### classify_spiral_data

| Model | Final Output | Hidden |
| :---: | :---: | :---: |
| MLP | ![MLP](./visualize_train/classification/linear/fin_16.gif) | ![MLP](./visualize_train/classification/linear/hid_16.gif) |
| Heads | ![Heads](./visualize_train/classification/heads/fin_16.gif) | ![Heads](./visualize_train/classification/heads/hid_16.gif) |

## Result

See [visualize/README.md](visualize/README.md) for more details.

![Result](result.png)