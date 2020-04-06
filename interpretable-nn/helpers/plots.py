import matplotlib.pyplot as plt


# This function will plot images in the form of a grid with 1 row and 5 columns
# where images are placed in each column.

def plot_rgb_images(images_arr, num_imgs=5, titles=None):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for i, (img, ax) in enumerate(zip(images_arr, axes)):
        ax.imshow(img)
        ax.axis('off')
        if titles:
            txt_top = [l + '\n' for l in titles[i]]
            ax.set_title(''.join(txt_top),
                         horizontalalignment='left')
    plt.tight_layout()
    plt.show()


def plot_gray_images(images_arr, num_imgs=5):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = img.reshape((img.shape[0], img.shape[1]))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_analysis_images(images_arr, num_imgs=5):
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = img.reshape((img.shape[0], img.shape[1]))
        ax.imshow(img, cmap='seismic', clim=(-1, 1))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_rgb_decomposition(rgb_image):
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip([r, g, b], axes):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_image_grid(images,
                    grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    figsize=None,
                    dpi=224):
    n_rows = len(grid)
    n_cols = len(grid[0]) + 1
    if figsize is None:
        figsize = (n_cols, n_rows + 1)

    plt.clf()
    plt.rc("font", family="sans-serif")

    plt.figure(figsize=figsize)
    for r in range(n_rows):
        ax = plt.subplot2grid(shape=[n_rows + 1, n_cols], loc=[r + 1, 0])
        ax.imshow(images[r])
        ax.set_xticks([])
        ax.set_yticks([])

        # row labels

        if row_labels_left != []:
            txt_left = [l + '\n' for l in row_labels_left[r]]
            ax.set_ylabel(
                ''.join(txt_left),
                rotation=0,
                verticalalignment='center',
                horizontalalignment='right',
            )

        for c in range(0, n_cols - 1):
            ax = plt.subplot2grid(shape=[n_rows + 1, n_cols + 1],
                                  loc=[r + 1, c + 1])
            if grid[r][c] is not None:
                ax.imshow(grid[r][c], interpolation='none', cmap='seismic',
                          clim=(-1, 1))
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            if c == n_cols - 2:
                if row_labels_right != []:
                    txt_right = [l + '\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='left'
                    )

    if file_name is None:
        plt.show()
    else:
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi)