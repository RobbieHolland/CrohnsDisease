import matplotlib.pyplot as plt
from data.augmentation.augment_data import Augmentor
from data.handling.data_handler import DataHandler

def manipulate_example(fig, row, n_cols, example_image, manipulation):
    fig.add_subplot(row + 1, n_cols + 1, 1)
    plt.axis('off')
    plt.imshow(example_image)
    plt.title('Before')

    for i in range(n_manipulations):
        fig.add_subplot(row + 1, n_cols + 1, i + 2)
        plt.axis('off')
        plt.imshow(manipulation(example_image))
        plt.title('After #' + str(i + 1))

    return fig

if __name__ == "__main__":
    data_handler = DataHandler('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
    features_tr, _, _, _ = data_handler.load_dataset()

    n_examples = 2
    n_manipulations = 7

    figure_size = (n_examples, n_manipulations + 1)
    fig = plt.figure(figsize=figure_size)
    plt.axis('off')

    augmentor = Augmentor()

    for i in range(n_examples):
        fig = manipulate_example(fig, i, n_manipulations, features_tr[i], augmentor.augment)

    plt.show()
