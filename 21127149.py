# Student Name: Huỳnh Minh Quang
# Student ID: 21127149
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Initialize centroids
def initial_Centroids(img, k_clusters, initCentroids): 
    if initCentroids == 'random':
        # Random initialization
        return np.random.randint(0, 256, size=(k_clusters, img.shape[1]))
    elif initCentroids == 'in_pixels':
        # Initialize centroids from random pixels in the image
        rand_indices = np.random.choice(img.shape[0], size=k_clusters, replace=False)
        return img[rand_indices]
    else:
        return None
# Update centroids
def update_Cetroids(img, k_clusters, labels, old_centroids):
   centroids=old_centroids
   for i in range(k_clusters):
        mask = labels == i
        if np.any(mask):
            centroids[i] = np.mean(img[mask], axis=0)
        return centroids
   
def label_Pixels(img, centroids):
    # Assign labels to pixels
    distances = np.linalg.norm(img[:, np.newaxis] - centroids, axis=-1)
    return np.argmin(distances, axis=1)

def kmeans(img_1d, k_clusters, max_iter, initCentroids):
    # Initialize cluster centroids and label
    centroids=initial_Centroids(img_1d, k_clusters, initCentroids)
    labels = label_Pixels(img_1d, centroids)
     # Run K-means
    for _ in range(max_iter):
        old_centroids=centroids.copy()
        centroids=update_Cetroids(img_1d, k_clusters, labels, old_centroids)
    # Check convergence
        if np.allclose(old_centroids, centroids, rtol=1e-3, equal_nan=False):
            break
    return centroids, labels

def image_Processing(image_File_Name, k):
    # Load the image
    image = Image.open(image_File_Name)
    # Convert the image to a numpy array
    img_array = np.array(image)   

    # Reshape the array to a 2D matrix
    img_2d = img_array.reshape(-1, img_array.shape[-1])
    # Init value
    max_iter = 1000
    initCentroids = 'random'

    centroids, labels = kmeans(img_2d, k, max_iter, initCentroids)
    # Update the pixel values based on the cluster centroids
    reduced_img_2d = centroids[labels]

    # Reshape the reduced image back to its original shape
    reduced_img_array = reduced_img_2d.reshape(img_array.shape)

    # Create a PIL image from the reduced image array
    reduce_Image=Image.fromarray(reduced_img_array.astype(np.uint8))
    return image, reduce_Image

# Display the original and reduced images
def Display(image, reduced_image, k):  
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(reduced_image)
    axes[1].set_title("Reduced Image (k = {})".format(k))
    axes[1].axis("off")
    plt.show()

# Save the reduced image
def Save_img(reduce_Image, k, input_File_Name): 
    # Prompt the user to choose the output format
    output_format = input("Enter the output format (png/pdf): ")
    fileName = input_File_Name.split('.')
    fileName = fileName[0] + '_k' + str(k) + '.' + output_format
    reduce_Image.save(fileName)

# Main
if __name__ == '__main__':
     # Prompt the user to enter the image file name
    input_File_Name= input("Enter the image file name: ")
    # Prompt the user to enter the
    k = int(input("Enter the number of colors to reduce to: "))

    init_Image, reduce_Image=image_Processing(input_File_Name, k)

    Display(init_Image, reduce_Image, k)

    Save_img( reduce_Image, k, input_File_Name)
