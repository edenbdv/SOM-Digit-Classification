 # SOM-Based Digit Recognition
 
## Self-Organizing Maps (SOM)
- Self-Organizing Maps (SOM) are a type of unsupervised artificial neural network that is used for clustering and visualization of high-dimensional data. They operate by mapping input data onto a lower-dimensional grid of neurons while preserving the topological properties of the input space.
- Each neuron competes to become the "best matching unit" (BMU) for a given input, based on similarity measures such as distance. As the network is trained, the weights of the neurons are adjusted, enabling them to capture patterns and structures within the data.
- This results in a spatial representation where similar inputs are positioned close together, making SOMs particularly effective for tasks like pattern recognition, data visualization, and exploratory data analysis.

## Overview
This project implements a Self-Organizing Map (SOM) for recognizing handwritten digits from a dataset. The goal is to visualize and analyze how well the SOM can represent and categorize different digit classes (0-9). The provided executable runs on Windows, presenting an interactive visualization of the learned digit representations.


### Visualization Methodology
- Each neuron in the SOM was represented by a visual image, labeled as “Neuron (x, y) | Label: z”, where:
  - **x and y** correspond to the neuron’s position in the SOM grid.
  - **z** represents the dominant label of the neuron, indicating the digit it most likely represents.
  
- Different colors were assigned to each digit to enhance visualization. This allows for easy identification of how neurons are clustered and their representation of various digits. For instance, neurons identifying the digit '2' are primarily located in the top left corner, while those for '0' are predominantly in the bottom right.

 ![som](https://github.com/user-attachments/assets/efc8a170-4a31-4259-bfc0-09751a10c7db)


## Initialization of Weights
- Weights of the neurons were initialized by calculating the average values of the input vectors and then adding small random values centered around this average. This approach helps the algorithm skip several iterations in finding the optimal solution, thus speeding up convergence.

## Neighborhood and Influence Calculation
- Two main variables were used:
  - **radius:** This indicates the maximum distance from the winning neuron (BMU) that will influence the weights of neighboring neurons. The radius starts large and decreases with more iterations, allowing the SOM to learn more intricate patterns later in the training process.
  
  - **influence:** Represents the effect on neighboring neurons relative to the BMU. Neurons closer to the BMU have a higher influence than those farther away.

## Error Metrics
- **Quantization Error:** Indicates the average RMS distance between input samples and their corresponding BMUs. A low value suggests a good mapping of inputs to similar areas in the SOM.
  
- **Topological Error:** Represents how well the SOM preserves the topology of the input space. A low value indicates that neighboring inputs in the original space remain close in the SOM.
  
- **Average Dominant Label Percentage:** Shows the average occurrence percentage of the dominant label for each neuron, reflecting how well the neurons represent their respective digits.


## How to Run the SOM Program

1. **Ensure the following files are in the working directory**:
   - `digits_test.csv`: The input data containing 10,000 grayscale images of digits (0-9), each represented as a 28x28 pixel matrix.
   - `digits_test_keys.csv`: The labels for each image (used for evaluation after training).

2. **Install dependencies**:
   Make sure you have the necessary Python libraries installed. You can install them using the following command:
   ```bash
   pip install numpy matplotlib pandas
   ```
3. **Run the program**:
   Execute the Python file from the command line:
   ```bash
   python main.py 
   ```
4. **Output**:
   - The program will train a Self-Organizing Map (SOM) on the input digit data.
   - After training, it will display the SOM grid with the dominant digit for each neuron and its percentage of occurrence.
   - The program will also show the error metrics that were mentioned before.
 



