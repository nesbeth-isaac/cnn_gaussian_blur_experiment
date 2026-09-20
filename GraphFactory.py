from matplotlib import pyplot as plt

class GraphFactory:
    """
        A factory class for plotting and saving comparative training metrics
        (Training Accuracy, Loss, and Test Accuracy) across multiple trained models.
        """

    def __init__(self, trained_models):

        """
                Initializes the factory with a collection of trained model instances
                and creates a 1x3 Matplotlib subplot grid.

                Parameters:
                - trained_models (list): A list of trained model objects, where each object
                                         contains metric arrays/tensors (e.g., training_accuracy,
                                         loss, test_accuracy, and model.model_name).
                """

        self.trained_models = trained_models
        self.list_of_figures = []

        self.colors = []

        self.fig, self.ax = plt.subplots(1, 3, figsize=(20, 10))

        self.graph_name = None


    def generate_colours(self):

        """
                Generates visually distinct colors across the 'turbo' colormap
                corresponding to the total number of models being plotted.
                """

        print("Setting Line Colours for Review Graphs")
        num_lines = len(self.trained_models)

        min_distance = 0.35  # Distance threshold in RGB space

        # Old (Deprecated):
        # cmap = plt.cm.get_cmap('turbo', num_lines)

        # Modern (Correct):
        cmap = plt.colormaps['turbo'].resampled(num_lines)
        self.colors = [cmap(i) for i in range(num_lines)]

    def generate_summary_graph(self):

        """
                Iterates through all trained models and plots their training accuracy,
                loss, and test accuracy curves onto their respective subplots,
                then formats axis titles, labels, and limits.
                """

        iterator = 0

        self.generate_colours()

        for trained_model in self.trained_models:
            self.ax[0].plot(trained_model.training_accuracy.detach(), label=trained_model.generated_model.model_name,
                       color=self.colors[iterator])

            self.ax[1].plot(trained_model.training_loss.numpy(), color=self.colors[iterator])

            self.ax[2].plot(trained_model.test_accuracy.detach(), color=self.colors[iterator])

            iterator += 1

        self.ax[0].set_title(f'Training Accuracy')
        self. ax[0].set_ylabel('Accuracy')
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylim([0, 110])

        self.ax[1].set_title('Training Loss')
        self.ax[1].set_ylabel('Loss')
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylim([0, 1])

        self.ax[2].set_title('Test Accuracy')
        self.ax[2].set_ylabel('Accuracy')
        self.ax[2].set_xlabel('Epoch')
        self.ax[2].set_ylim([0, 110])

    def plot_graph(self):

        """
                Executes the graph generation, attaches a shared figure legend,
                saves the figure to a PNG file, displays it, and returns the file path.

                Returns:
                - FILE_NAME (str): Name of the saved output image file.
                """

        GRAPH_NAME = self.graph_name + " Overall_Training_Graph"
        FILE_NAME = GRAPH_NAME + ".png"

        print("Plotting Graphs")

        self.generate_summary_graph()
        for axis in self.ax:
            axis.plot()

        self.fig.legend(loc="upper center",
                   bbox_to_anchor=(0.5, -0.02),  # Position below the figure (y=-0.02)
                   ncol=5)

        self.fig.savefig(GRAPH_NAME, bbox_inches='tight')

        print("File Saved Locally")

        self.fig.show()

        return FILE_NAME

    def set_graph_name(self, experiment_name):
        self.graph_name = experiment_name











