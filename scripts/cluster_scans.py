import logging
import os
import glob
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from visualization_app.utils import bbox_from_image_contours

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def cluster_features(image_paths, feature_vector_func, n_clusters=5):
    """Clusters images based on extracted features.

    Args:
        image_paths (list): List of image file paths.
        feature_vector_func (callable): Function which extract features from an image.
        n_clusters (int): Number of clusters to form.
    Returns:
        dict: Mapping {cluster: [image files]}
    """
    X = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        feature_vector = feature_vector_func(image)
        X.append(feature_vector)

    Z = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(16, Z.shape[1])).fit_transform(Z)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(Z)

    clusters = {i: [] for i in range(n_clusters)}
    for f, c in zip(files, labels):
        clusters[c].append(f)

    return clusters


def get_feature_vec_img_properties(image):
    """Extracts properties like histogram, edge density, laplacian variance,
    and color saturation from an image.

    Args:
        image (numpy.ndarray): Input image.
    Returns:
        numpy.ndarray: Feature vector
    """
    # bbox = bbox_from_image_contours(image)
    # image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # Crop to bounding box

    h, w = image.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0
    mean_i = gray.mean() / 255.0
    black_fr = (gray < 20).mean()
    lapv = np.tanh(cv2.Laplacian(gray, cv2.CV_32F).var() / 1000.0)
    mean_sat = hsv[..., 1].mean() / 255.0

    return np.concatenate([hist, [edge_density, mean_i, black_fr, lapv, mean_sat]])


def get_feature_vec_scan_position(image):
    """Extracts position and edge color features from an image.

    Args:
        image (numpy.ndarray): Input image.
    Returns:
        numpy.ndarray: Feature vector
    """
    bbox = bbox_from_image_contours(image)

    left_intensity = image[:, 0:30].mean() / 255.0
    right_intensity = image[:, -30:].mean() / 255.0

    return np.concatenate([bbox, [left_intensity, right_intensity]])


def load_images(folder):
    """Loads images from a specified folder.

    Args:
        folder (str): Path to the folder containing images.
    Returns:
        list: List of image file paths.
    """
    exts = ("*.jpg", "*.tif")
    files = sum([glob.glob(os.path.join(folder, e)) for e in exts], [])
    files.sort()

    LOGGER.info(f"Found {len(files)} images in {folder}")
    return files


def visualize(clusters, title):
    """Visualizes clustered images.

    Args:
        clusters (dict): Mapping {cluster: [image files]}.
        title (str): saved image title.
    """
    largest_cluster_size = max(len(images) for images in clusters.values())
    largest_cluster_size = min(
        largest_cluster_size, 20
    )  # Ensure maximally 20 images per row
    fig, axs = plt.subplots(
        len(clusters), largest_cluster_size, figsize=(10, len(clusters) * 2)
    )

    for ax_row, (label, images) in zip(axs, clusters.items()):
        for i, img_path in enumerate(images):
            if i == 0:
                ax_row[i].set_title(
                    f"Cluster {label} ({len(images)} images)", fontsize=10
                )
            if i == largest_cluster_size:
                break

            img = cv2.imread(img_path)
            ax_row[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        LOGGER.info(f"Cluster {label}: {images} ({len(images)} images)")

    # Plot images into a grid
    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(0)
    plt.savefig(f"{title}.png")
    plt.show()


if __name__ == "__main__":
    path = os.getenv("SCAN_DATA_PATH")
    files = load_images(path)

    bbox_labels = cluster_features(files, get_feature_vec_scan_position, n_clusters=2)
    feature_labels = cluster_features(
        files, get_feature_vec_img_properties, n_clusters=3
    )

    visualize(bbox_labels, "bbox_clusters")
    visualize(feature_labels, "feature_clusters")
