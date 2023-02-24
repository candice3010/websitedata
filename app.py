import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

logo = Image.open('ShopBag-removebg-preview.png')



# Define the function for the first tab
def presentation():
    #st.image("fashion-2019-1544479548-1.jpg", use_column_width=True)
    st.header("Welcome to K-Fashion a New Fashion Recommendation Web!")
    st.subheader("About Us")
    img = Image.open("IMG_8495.PNG")
    st.image(img, use_column_width=True)
    st.write("We are a team of fashion enthusiasts who have developed this website to help people find their perfect outfit.")
    st.subheader("How it Works")
    st.write("Our website uses machine learning algorithms to analyze images of different clothing items and group them into clusters based on their similarities. Users can generate a random image or filter the images by selecting specific clusters.")
    st.subheader("Contact Us")
    st.write("If you have any questions or feedback, please email us at info@productrecommendationweb.com")
    # Define the image URL and Instagram profile URL
    img_url = "https://www.logo.wine/a/logo/Instagram/Instagram-Logo.wine.svg"
    profile_url = "https://www.instagram.com/kfashion.uk/"
    # Create a button with the Instagram logo as its background image
    button_html = f'<a href="{profile_url}" target="_blank"><img src="{img_url}" style="width: 80px"></a>'
    st.markdown(button_html, unsafe_allow_html=True)


def generate_image():
    data = pd.read_csv('product_images_with_cluster.csv').to_numpy()

    images = [ np.reshape(row[0:-1], (28, 28)) for row in data]
    labels = [ row[-1] for row in data]

    # Create dictionary to store images for each cluster
    k = len(np.unique(labels))
    groups = {}
    for i in range(k):
        groups[i] = []
        for j in range(len(images)):
            if labels[j] == i:
                groups[i].append(images[j])

    st.title("Generate an Image")

    result = st.button("Click Here to generate an image")

    random_image_index = np.random.choice(range(9999))

    if result:
        st.subheader("Random product")
        st.image(images[random_image_index], width=400)

        # Find the cluster of the random image
        cluster = labels[random_image_index]

        # Find other images in the same cluster
        cluster_images = groups[cluster]

        for pixel in cluster_images:
            st.write("Recommended Product")
            st.image(pixel, width=200)


def filter_clusters():
    data = pd.read_csv('product_images_with_cluster.csv').to_numpy()

    images = [np.reshape(row[0:-1], (28, 28)) for row in data]
    labels = [row[-1] for row in data]

    st.header("Filter")

    k = len(np.unique(labels))

    groups = {}

    for i in range(k):
        groups[i] = []
        for j in range(len(images)):
            if labels[j] == i:
                groups[i].append(images[j])

    cluster_names = {0: "Vest", 1: "Long Sleeve", 2: "Dress", 3: "Shirt", 4:"Sneakers", 5:"Sandals", 6: "Short Dress", 7: "T-shirt", 8: "Pants", 9: "Boots", 10:"Pullover", 11: "Bag", 12: "Shoes", 13: "Big bag" }

    options = [f"{cluster_names.get(i, 'Cluster ' + str(i))}: {len(groups[i])} images" for i in range(k)]

    selected_clusters = st.multiselect("Select clusters", options=options, default=options)

    for i in range(k):
        if f"{cluster_names.get(i, 'Cluster ' + str(i))}: {len(groups[i])} images" in selected_clusters:
            for pixels in groups[i]:
                st.image(pixels, width=200)
            st.write(f"{cluster_names.get(i, 'Cluster ' + str(i))}: {len(groups[i])} images")

def recommendations():
    data = pd.read_csv('product_images_with_cluster.csv').to_numpy()

    # Extract images and labels
    images = [ np.reshape(row[0:-1], (28, 28)) for row in data]
    images_flat = np.array([img.flatten() for img in images])
    labels = [ row[-1] for row in data]

    # Fit KMeans clustering algorithm to flattened images
    kmeans = KMeans(n_clusters=len(np.unique(labels)))
    kmeans.fit(images_flat)

    # Calculate number of clusters
    k = len(np.unique(labels))

    # Create dictionary to store images for each cluster
    groups = {}
    for i in range(k):
        groups[i] = []
        for j in range(len(images)):
            if labels[j] == i:
                groups[i].append(images[j])

    # Create empty basket
    basket = []

    cluster_names = ['Long sleeves', 'T-shirts', 'dresses', 'shirts', 'sweats', 'shoes','dress', 'T-shirts', 'pants','boots','jackets','bags',"shoes","bags"]

    # Create dropdown menu to select cluster
    selected_cluster = st.sidebar.selectbox('Select a cluster:', cluster_names)

    # Get images in selected cluster
    selected_images = groups[cluster_names.index(selected_cluster)]

    # Create grid for images and basket
    col1, col2 = st.beta_columns([2, 1])

    # Show images in selected cluster
    with col1:
        st.header("Product Recommendation Web App")
        st.subheader("Images in Selected Cluster")
        for i in range(len(selected_images)):
            col_img, col_button = st.columns([2, 1])
            col_img.image(selected_images[i], width=100, use_column_width=True, caption=f"Image {i}")
            if col_button.button(f"Add to basket {i}", key=f"add_button_{i}"):
                basket.append(selected_images[i])

                # Get cluster number for added image
                added_image_cluster = None
                for j in range(len(images)):
                    if np.array_equal(images[j], selected_images[i]):
                        added_image_cluster = labels[j]
                        break

                # Get recommended images
                recommended_images = groups[added_image_cluster][:5]

                # Show recommended images
                st.subheader("Recommended Products")
                fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 2))
                for j in range(5):
                    ax[j].imshow(recommended_images[j], cmap='gray')
                    ax[j].axis('off')
                st.pyplot(fig)
                break

    # Show basket
    with col2:
        st.subheader("Basket")
        for i in range(len(basket)):
            st.image(basket[i], width=100, caption=f"Product {i}")
            

def accuracy_matrix():
    df1 = pd.read_csv('product_images_with_cluster.csv')
    df2 = pd.read_csv('true_label.csv')
    df_all_rows = pd.concat([df1, df2])

    # Load data and true labels
    data = df1
    true_labels = df2

    # Define input widgets
    n_components = st.slider('Select the number of PCA components:', min_value=2, max_value=data.shape[1], value=2)
    n_clusters = st.slider('Select the number of k-means clusters:', min_value=2, max_value=10, value=2)

    # Initialize lists to store results
    fmi_scores = []
    ari_scores = []
    nmi_scores = []

    # Calculate accuracy scores for given PCA components and k-means clusters
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    data_pca = np.nan_to_num(data_pca)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(data_pca)
    fmi_avg = fowlkes_mallows_score(true_labels["label"], cluster_labels)
    ari_avg = adjusted_rand_score(true_labels["label"], cluster_labels)
    nmi_avg = normalized_mutual_info_score(true_labels["label"], cluster_labels)

    # Append accuracy scores to lists
    fmi_scores.append(fmi_avg)
    ari_scores.append(ari_avg)
    nmi_scores.append(nmi_avg)

    # Print accuracy scores
    st.write(f"Fowlkes-Mallows Index: {fmi_scores[0]}")
    st.write(f"Adjusted Rand Index: {ari_scores[0]}")
    st.write(f"Normalized Mutual Information: {nmi_scores[0]}")

    # Display optimal results
    st.write("")
    st.write("Our optimal results were found using 14 clusters and 70 principal components.")
    st.write("")
    st.write("Cluster Matrix:")
    st.write(df_all_rows)





tabs = ['Presentation', 'Generate Image','reccommendations',"accuracy matrix",'filter clusters']
selected_tab = st.sidebar.selectbox('Select an option', tabs)

# Show the appropriate tab based on user selection
if selected_tab == 'Presentation':
    presentation()
elif selected_tab == 'Generate Image':
    generate_image()
elif selected_tab == 'reccommendations':
    recommendations()
elif selected_tab == 'accuracy matrix':
    accuracy_matrix()
else:
    filter_clusters()
