import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SOM:
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, radius_decay=0.98):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(x, y)
        self.radius_decay = radius_decay

        # Compute the mean of the input data
        data_mean = np.mean(data, axis=0)

        # Initialize weights to small random values centered around the mean of the input data
        self.weights = data_mean + np.random.randn(x, y, input_len) * 0.01

    def _rms(self, vector, neurons):
        return np.sqrt(np.sum((vector - neurons) ** 2, axis=2))

    def _get_2nd_bmu(self, vector):
        distances = self._rms(vector, self.weights)
        sorted_indices = np.argsort(distances, axis=None)
        first_bmu_idx = np.unravel_index(sorted_indices[0], distances.shape)
        second_bmu_idx = np.unravel_index(sorted_indices[1], distances.shape)
        first_bmu_distance = distances[first_bmu_idx]
        second_bmu_distance = distances[second_bmu_idx]

        return first_bmu_idx, second_bmu_idx, first_bmu_distance, second_bmu_distance

    def _update_weights(self, vector, bmu_idx, iteration, total_iterations):

        # Linear decay for radius and learning rate
        radius = self.radius * (1 - iteration / total_iterations)
        learning_rate = self.learning_rate * (1 - iteration / total_iterations)

        # Create a grid of indices
        x_indices, y_indices = np.indices((self.x, self.y))

        # Calculate distance from each neuron to the BMU
        dist_to_bmu = np.sqrt((x_indices - bmu_idx[0]) ** 2 + (y_indices - bmu_idx[1]) ** 2)

        # Neighborhood function (linear decay)
        influence = 1 - dist_to_bmu / radius
        influence[dist_to_bmu > radius] = 0  # Outside the radius, no influence

        # Update weights
        self.weights += learning_rate * influence[..., np.newaxis] * (vector - self.weights)


    def train(self, data, num_iterations):
        for iteration in range(num_iterations):
            for vector in data:
                bmu_idx = self._get_2nd_bmu(vector)[0]
                #print("found bmu")
                self._update_weights(vector, bmu_idx, iteration, num_iterations)
                #print("weight updated")


    def quantization_error(self, data, mapped):
        total_distance = 0
        for idx, mapped_neuron in enumerate(mapped):
            weighted_neuron = self.weights[mapped_neuron[0], mapped_neuron[1]]
            distance = np.sqrt(np.sum((data[idx] - weighted_neuron) ** 2))
            total_distance += distance
        return total_distance / len(data)

    def topological_error(self, data):
        bad_mappings = 0
        for vector in data:
            bmu_idx, second_bmu_idx, _, _ = self._get_2nd_bmu(vector)
            if not self._are_neighbors(bmu_idx, second_bmu_idx):
                bad_mappings += 1
        return bad_mappings / len(data)

    def _are_neighbors(self, idx1, idx2):
        return max(abs(idx1[0] - idx2[0]), abs(idx1[1] - idx2[1])) == 1


    def map_vects(self, data):
        mapped = np.array([self._get_2nd_bmu(vector)[0] for vector in data])
        return mapped

    def calculate_dominant_labels(self, mapped, labels):
        # Create a dictionary to count the occurrences of each label for each neuron
        neuron_label_count = {}
        for idx, (neuron, label) in enumerate(zip(mapped, labels)):
            neuron_tuple = tuple(neuron)  # Convert numpy array to tuple
            if neuron_tuple not in neuron_label_count:
                neuron_label_count[neuron_tuple] = {}
            if label not in neuron_label_count[neuron_tuple]:
                neuron_label_count[neuron_tuple][label] = 0
            neuron_label_count[neuron_tuple][label] += 1

        # Calculate the dominant label and its percentage for each neuron
        neuron_dominant_label = {}
        for neuron, label_counts in neuron_label_count.items():
            total_count = sum(label_counts.values())
            dominant_label = max(label_counts, key=label_counts.get)
            dominant_percentage = label_counts[dominant_label] / total_count
            neuron_dominant_label[neuron] = (
            dominant_label, dominant_percentage)

        return neuron_dominant_label

    def plot_neurons(self, neuron_dominant_label, data):
        # Create a figure
        fig, axes = plt.subplots(self.x, self.y, figsize=(10, 10))

        # Define colors for digits 0-9 (you can customize these colors as needed)
        digit_colors = {
            0: (0.12156863, 0.46666667, 0.70588235),  # Blue
            1: (1.0, 0.49803922, 0.05490196),  # Orange
            2: (0.17254902, 0.62745098, 0.17254902),  # Green
            3: (0.83921569, 0.15294118, 0.15686275),  # Red
            4: (0.58039216, 0.40392157, 0.74117647),  # Purple
            5: (0.54901961, 0.3372549, 0.29411765),  # Brown
            6: (0.89019608, 0.46666667, 0.76078431),  # Pink
            7: (0.49803922, 0.49803922, 0.49803922),  # Gray
            8: (0.7372549, 0.74117647, 0.13333333),  # Yellow
            9: (0.09019608, 0.74509804, 0.81176471)  # Cyan
        }

        # Plot each neuron
        for i in range(self.x):
            for j in range(self.y):
                neuron = (i, j)
                if neuron in neuron_dominant_label:
                    label, _ = neuron_dominant_label[neuron]
                    color = digit_colors[label]
                    # Create RGB image with colorized background
                    rgb_image = np.ones((28, 28, 3), dtype=np.float32)
                    rgb_image[:, :, 0] *= color[0]  # Red channel
                    rgb_image[:, :, 1] *= color[1]  # Green channel
                    rgb_image[:, :, 2] *= color[2]  # Blue channel

                    # Overlay SOM neuron weights
                    axes[i, j].imshow(rgb_image)
                    axes[i, j].imshow(self.weights[i, j].reshape(28, 28),
                                      cmap='gray',
                                      alpha=0.5)  # Overlay neuron weights with transparency
                    axes[i, j].set_title(f'Neuron {neuron} | Label: {label}',
                                         fontsize=8)
                    axes[i, j].axis('off')

                else:
                    axes[i, j].axis('off')

        # Adjust layout
        plt.tight_layout()
        plt.show()


# Read dataset
data_path = 'digits_test.csv'
data_df = pd.read_csv(data_path, header=None)
data = data_df.values / 255.0  # Normalize data to range [0, 1]


som = SOM(x=10, y=10, input_len=784, learning_rate=0.7)
som.train(data, num_iterations=50)

# Map digits to SOM
mapped = som.map_vects(data)

# Calculate dominant labels
labels_path = 'digits_keys.csv'
labels_df = pd.read_csv(labels_path, header=None)
labels = labels_df.values.flatten()
neuron_dominant_label = som.calculate_dominant_labels(mapped, labels)


# Calculate and print errors
quant_error = som.quantization_error(data, mapped)
topo_error = som.topological_error(data)

print(f"Quantization Error: {quant_error}")
print(f"Topological Error: {topo_error}")

total_percentage = 0

for neuron, (label, percentage) in neuron_dominant_label.items():
    total_percentage+= percentage

# Calculate and print the average percentage
avg_percentage = total_percentage / len(neuron_dominant_label)
print(f"Average Dominant Label Percentage: {avg_percentage:.2f}%")


som.plot_neurons(neuron_dominant_label, data)


