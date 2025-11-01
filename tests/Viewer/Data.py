from tensorflow.keras.datasets import mnist
import numpy as np


class DataManager:
    def __init__(self):
        (self.x_train, self.y_train), (_, _) = mnist.load_data()

    def get_images_by_digit(self, digit):
        return self.x_train[self.y_train == digit]

    def POD(self, images, num_modes:int = 10):
        # Reshape images to 2D array (num_samples, num_features)
        num_samples, height, width = images.shape
        reshaped_images = images.reshape(num_samples, height * width)

        # Compute the mean image
        mean_image = reshaped_images.mean(axis=0)

        # Center the images by subtracting the mean image
        centered_images = reshaped_images - mean_image

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_images, rowvar=False)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top 'num_modes' eigenvectors (POD modes)
        pod_modes = sorted_eigenvectors[:, :num_modes].reshape(height, width, num_modes)

        return pod_modes, mean_image.reshape(height, width)
    
    def eigenvector_basis(self, image, num_basis:int = 10, ordering:bool=True):
        eigenvalues, eigenvectors = np.linalg.eig(image)
        if ordering:
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
        
        num_eigenvalues = len(eigenvalues)
        basis = np.zeros((num_basis, image.shape[0], image.shape[1]), dtype=np.complex128)
        for i in range(num_basis if num_basis < num_eigenvalues else num_eigenvalues):
            basis[i, :, :] = np.outer(eigenvectors[:, i], eigenvectors[:, i].conj())
        
        return basis, eigenvalues[:num_basis]
    
    def reconstruct_image(self, basis, eigenvalues, basis_count:int=None):
        reconstructed_image = np.zeros(basis[0].shape, dtype=basis[0].dtype)
        for i in range(len(basis) if basis_count > len(basis) or basis_count is None else basis_count):
            reconstructed_image += eigenvalues[i] * basis[i]
        return np.real(reconstructed_image)