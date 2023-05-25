import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

class Plotter():
    def plot_retrieval_results(self, fldr, count, rank, query_img, similar_images: list):
        for img in similar_images:
            img = img.astype('float32')

        fig = plt.figure()
        fig.suptitle(f'Retrieval results with SIFT method')

        n_rows = int(np.ceil(rank / 5) + 1)
        gs = GridSpec(n_rows, 5)
        query_axis = fig.add_subplot(gs[0, 1])
        query_axis.set_title('Query image')
        query_axis.imshow(query_img)
        
        for i in np.arange(1, n_rows):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(similar_images[5 * (i-1) + 0])

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(similar_images[5 * (i-1) + 1])

            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(similar_images[5 * (i-1) + 2])
            ax3.set_title('Results:')

            ax4 = fig.add_subplot(gs[i, 3])
            ax4.imshow(similar_images[5 * (i-1) + 3])

            ax5 = fig.add_subplot(gs[i, 4])
            ax5.imshow(similar_images[5 * (i-1) + 4])

        #plt.savefig(fldr + '/res' + str(count) + '.png')
        plt.show()