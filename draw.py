import matplotlib.pyplot as plt

def show_attention(
        attention,
        xlabel = 'Keys',
        ylabel = 'Queries',
        title = 'Attention weights',
        figsize=(5, 5),
        cmap = 'Reds'
    ):
    """
    画出注意力权重图

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (m, n)
    """

    fig = plt.figure(figsize = figsize)

    pcm = plt.imshow(
        attention, 
        cmap = cmap
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.colorbar(pcm, shrink=0.7)
    plt.show()


def show_attention_batch(
        attention,
        xlabel='Keys',
        ylabel='Queries',
        title='Attention weights',
        figsize=(5, 5),
        cmap='Reds'
    ):
    """
    画出注意力权重图

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (b, m, n)
    """

    # Get the batch size
    batch_size, _, _ = attention.size()

    # Create subplots for each attention matrix in the batch
    fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * figsize[0], figsize[1]))

    # Iterate through each attention matrix in the batch
    for i in range(batch_size):
        axs[i].imshow(
            attention[i],
            cmap=cmap
        )
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(f'{title} - Batch {i+1}')

    plt.tight_layout()
    plt.show()