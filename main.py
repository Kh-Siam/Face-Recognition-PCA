# To easily trace the code, 
# start from main right after reading
# the global variables.
 
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Define Relative Locations
DATASET_ARCHIVE      = "./archive"
FOLDERS_NAME         = "/s"

# Define Constants
NUMBER_OF_FOLDERS    = 40
NUMBER_OF_IMAGES     = 10
FORMAT_TYPE          = ".pgm"

# Givens
ALPHA                = [0.8, 0.85, 0.9, 0.95]
K                    = [  1,    3,   5,    7]

# This holds the entire dataset
dataset              = list()

# This holds the labels corresponding to the data
labels               = list()

# These are the training and testing sets as required
training_dataset     = list()
testing_dataset      = list()

# These are the labels divided accordingly
training_labels      = list()
testing_labels       = list()

# These are only for performance enhancing purposes in the PCA
eigenvalues          = list()
eigenvectors         = list()

# These two functions are just to avoid calculating the eigen values 
# and vectors with each run
def load_data(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return list()
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
            pickle.dump(data, f)

# Function to divide the database into training and testing sets
def split_dataset():
    global dataset
    global testing_dataset
    global training_dataset
    global labels
    global testing_labels
    global training_labels

    # Keep in mind that the even or odd here is based on indexing starting 
    # from 0. If you wish to deal with it starting from 1, then just swap
    # the new two lines.
    testing_dataset  = dataset[::2]
    testing_labels   = labels[::2]
    training_dataset = dataset[1::2]
    training_labels  = labels[1::2]

# Function to add images from archive folder into the dataset
# in the required format
def init_dataset():
    global dataset
    global labels

    for i in range(NUMBER_OF_FOLDERS):
        for j in range(NUMBER_OF_IMAGES):
            img = Image.open(DATASET_ARCHIVE + FOLDERS_NAME + str(i+1) + "/" + str(j+1) + FORMAT_TYPE)
            labels.append(i+1)
            dataset.append(np.array(img))
    dataset = np.resize(dataset, (NUMBER_OF_FOLDERS * NUMBER_OF_IMAGES, len(dataset[0]) * len(dataset[0][0])))
    split_dataset()

# This is a function that centers the dataset
def center_data(data):
    return data - data.mean(axis = 0)

# This is the main function that will run PCA on the dataset required
def pca(training_dataset, alpha):
    global eigenvalues
    global eigenvectors

    eigenvalues  = load_data("eigenvalues")
    eigenvectors = load_data("eigenvectors")
    if not (len(eigenvalues) and len(eigenvectors)):
        # Center the dataset
        training_dataset = center_data(training_dataset)
        # Compute the covariance matrix
        covariance_matrix = np.dot(training_dataset.T, training_dataset) / training_dataset.shape[0]
        # Retrieve the eigen values and vectors sorted in descending order
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigen values and vectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:,indices]
        save_data("eigenvalues", eigenvalues)
        save_data("eigenvectors", eigenvectors)
    # Calculate fraction of total variance and compare it to alpha
    total_sum_eigenvalues = sum(eigenvalues)
    for i in range(len(eigenvalues)):
        variance_fraction = sum(eigenvalues[:i]) / total_sum_eigenvalues
        if variance_fraction >= alpha:
            eigenvalue_index = i
            break
    # Based on the f(r), compute the new training set with the reduced basis
    projection_matrix = eigenvectors[:,:eigenvalue_index]
    return projection_matrix

# This function takes the projection_matrix and uses it to project the data on it
def project_dataset(data, projection_matrix):
    data = center_data(data)
    return (np.dot(projection_matrix.T, data.T)).T

# This function computes the first nearest neighbor classifier
def first_nearest_neighbor_classifier(training_dataset, training_labels, testing_dataset):
    predictions = list()
    for test_sample in testing_dataset:
        min_distance = np.inf
        min_index = -1
        for i, train_sample in enumerate(training_dataset):
            distance = np.linalg.norm(test_sample - train_sample)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        predictions.append(training_labels[min_index])
    return predictions

# This function computes the knn classifier. Notice that k is 3 by default.
def knn(training_dataset, training_labels, testing_dataset, k=3):
    predictions = list()
    for test_sample in testing_dataset:
        distances = [(i, np.linalg.norm(test_sample - train_sample)) for i, train_sample in enumerate(training_dataset)]
        k_nearest = sorted(distances, key=lambda x: x[1])[:k]
        predictions.append(max(set([training_labels[x[0]] for x in k_nearest]), key=[training_labels[x[0]] for x in k_nearest].count))
    return predictions

# This function simply calculates the accuracy of our predictions
def compute_accuracy(true_labels, predictions):
    return (sum(1 for x, y in zip(true_labels, predictions) if x == y) / len(true_labels)) * 100

if __name__=="__main__":
    print("Starting data initialization...")
    init_dataset()
    print("Data initialized successfully!")
    print("Shape of dataset: ", dataset.shape)
    print("Shape of training dataset: ", training_dataset.shape)
    print("Shape of testing  dataset: ", testing_dataset.shape)
    for alpha in ALPHA:
        print("\n\nAlpha: ", alpha*100, "%")
        print("\nComputing the projection matrix using PCA...")
        projection_matrix = pca(training_dataset, alpha)
        print("Projection matrix computed successfully!")
        print("Shape of projection matrix: ", projection_matrix.shape)
        print("\nProjecting both training and testing datasets...")
        new_training_dataset = project_dataset(training_dataset, projection_matrix)
        new_testing_dataset = project_dataset(testing_dataset, projection_matrix)
        print("Projection done!")
        print("Shape of training dataset after projection: ", new_training_dataset.shape)
        print("Shape of testing  dataset after projection: ", new_testing_dataset.shape)
        print("\nUsing first nearest neighbor classifier on the testing dataset...")
        predictions = first_nearest_neighbor_classifier(new_training_dataset, training_labels, new_testing_dataset)
        print("Using First Nearest Neighbor, Accuracy: ", compute_accuracy(testing_labels, predictions), "%")
        accuracy = []
        for k in K:
            predictions = knn(new_training_dataset, training_labels, new_testing_dataset, k)
            print("Using KNN with k=", k, ", Accuracy: ", compute_accuracy(testing_labels, predictions), "%")
            accuracy.append(compute_accuracy(testing_labels, predictions))

        # Create a scatter plot
        plt.scatter(K,accuracy, color='b', label='Data')

        # Perform linear regression to find the line of best fit
        coefficients = np.polyfit(K, accuracy, 1)  # Fit a first-degree (linear) polynomial
        poly = np.poly1d(coefficients)
        line_of_best_fit = poly(K)

        # Plot the line of best fit
        plt.plot(K, line_of_best_fit, color='r', label='Line of Best Fit')

        # Add labels and title
        plt.xlabel('K-number')
        plt.ylabel('Accuracy')
        plt.title('Scatter Plot with Line of Best Fit')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
    print("\nThere is a positive correlation between the classification accuracy and the ALPHA")
    print("\nDone!")