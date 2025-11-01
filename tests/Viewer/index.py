import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Data import DataManager as DM
from scipy.linalg import expm


D = DM()

st.markdown("""
<div style='text-align: center;'>
    <h1 style='color: #41649c;'>QPBL Viewer</h1>
    <p><strong>Author:</strong></p>
    <p> Gabriel Mejia, Eileen Kuhn &  Melvin Strobl</p>
    <h3>Abstract</h3>
    <p>
        Images with similarities must shared mathematical strucute in form of representaive basis. However each image can have different basis,
        which makes it difficult to find a common basis for a set of images. In this viewer we explore the MNIST dataset to find similarities
    </p>
</div>
""", unsafe_allow_html=True)


# Slider to select digit
digit = st.slider("Analyze a number", min_value=0, max_value=9, value=0)

np.random.seed(digit)

# Filter images of the selected digit
images = D.get_images_by_digit(digit)

# Randomly select 16 images
if images.shape[0] >= 16:
    idx = np.random.choice(images.shape[0], 16, replace=False)
    selected_images = images[idx]
else:
    selected_images = images


# Display images in a 4x4 grid
fig1, axes1 = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes1.flatten()):
    if i < selected_images.shape[0]:
        ax.imshow(selected_images[i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
st.pyplot(fig1)


st.markdown("""
<div style='text-align: center;'>
    <h3>POD</h3>
    <p>
        The modes of the Proper orthogonal decompotion are basis for the reconstruction of the images,
        which are more representative as the bigger of the eigenvalue.
    </p>
</div>
""", unsafe_allow_html=True)

# Perform POD
number_of_modes = st.slider("Number of POD modes", min_value=1, max_value=20, value=10)
modes, mean_image = D.POD(images, number_of_modes)

fig2, axes2 = plt.subplots(number_of_modes // 4 + 1, 4, figsize=(8, 8))
for i, ax in enumerate(axes2.flatten()):
    if i < number_of_modes:
        ax.imshow(modes[:,:,i], cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
st.pyplot(fig2)

st.markdown("""
<div style='text-align: center;'>
    <h3>EigenBasis</h3>
    <p>
        If we consider instead the basis of an individual image by eigenvalue decomposition,
    </p>
</div>
""", unsafe_allow_html=True)

st.latex(r"\ket{\text{Img}} = \sum_{j} \lambda_j\ket{q_j}\bra{q_j}")

# Eigenvector basis
n = st.slider("Select an image from the grid above", min_value=0, max_value=15, value=0)

fig3, axes3 = plt.subplots(figsize=(16, 16))
axes3.imshow(selected_images[n], cmap='gray')
axes3.axis('off')
st.pyplot(fig3)

EigenBasis, EigenValues = D.eigenvector_basis(selected_images[n], number_of_modes)

fig4, axes4 = plt.subplots(number_of_modes // 4 + 1, 4, figsize=(8, 8))
for i, ax in enumerate(axes4.flatten()):
    if i < number_of_modes:
        ax.imshow(np.real(EigenBasis[i]), cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
st.pyplot(fig4)

b = st.slider("Select the number of basis to reconstruct the image",
               min_value=0, max_value=len(EigenValues), value=3)


reconstructed_image = D.reconstruct_image(EigenBasis, EigenValues, b)

fig5, axes5 = plt.subplots(figsize=(16, 16))
axes5.imshow(reconstructed_image, cmap='gray')
axes5.axis('off')
st.pyplot(fig5)

ImgC = np.cov(selected_images[n])
EigenBasisC, EigenValuesC = D.eigenvector_basis(ImgC, number_of_modes)

fig8, axes8 = plt.subplots(figsize=(16, 16))
axes8.imshow(ImgC, cmap='gray')
axes8.axis('off')
st.pyplot(fig8)


fig6, axes6 = plt.subplots(number_of_modes // 4 + 1, 4, figsize=(8, 8))
for i, ax in enumerate(axes6.flatten()):
    if i < number_of_modes:
        ax.imshow(np.real(EigenBasisC[i]), cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
st.pyplot(fig6)

bC = st.slider("Select the number of basis to reconstruct the covariance image",
               min_value=0, max_value=len(EigenValuesC), value=3)


reconstructed_imageC = D.reconstruct_image(EigenBasisC, EigenValuesC, bC)

fig7, axes7 = plt.subplots(figsize=(16, 16))
axes7.imshow(reconstructed_imageC, cmap='gray')
axes7.axis('off')
st.pyplot(fig7)

ImgO = expm(1j*ImgC)

EigenBasisO, EigenValuesO = D.eigenvector_basis(ImgO, number_of_modes)

fig9, axes9 = plt.subplots(1, 3, figsize=(6, 6))
axes9[0].imshow(np.real(ImgO), cmap='gray')
axes9[0].axis('off')
axes9[1].imshow(np.imag(ImgO), cmap='gray')
axes9[1].axis('off')
axes9[2].imshow(np.abs(ImgO), cmap='gray')
axes9[2].axis('off')
st.pyplot(fig9)


fig10, axes10 = plt.subplots(number_of_modes // 4 + 1, 4, figsize=(8, 8))
for i, ax in enumerate(axes10.flatten()):
    if i < number_of_modes:
        ax.imshow(np.real(EigenBasisO[i]), cmap='gray')
        ax.axis('off')
    else:
        ax.axis('off')
st.pyplot(fig10)

# bC = st.slider("Select the number of basis to reconstruct the covariance image",
#                min_value=0, max_value=len(EigenValuesC), value=0)


# reconstructed_imageC = D.reconstruct_image(EigenBasisC, EigenValuesC, bC)

# fig7, axes7 = plt.subplots(figsize=(16, 16))
# axes7.imshow(reconstructed_imageC, cmap='gray')
# axes7.axis('off')
# st.pyplot(fig7)