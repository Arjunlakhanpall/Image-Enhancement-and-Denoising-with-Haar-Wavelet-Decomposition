# Image Fusion and Denoising using Haar Wavelet Transform

This project demonstrates image fusion and denoising techniques using the Haar wavelet transform and Non-Local Means denoising method. The goal is to merge two images and reduce noise while preserving important features.

## **Prerequisites**

Before running the code, make sure you have the following Python packages installed:

- `opencv-python` - for image processing  
- `scikit-image` - for image enhancement and restoration  
- `PyWavelets` - for wavelet decomposition and reconstruction  
- `matplotlib` - for visualizing images

You can install these dependencies by running:

```bash
pip install opencv-python scikit-image PyWavelets matplotlib
```

## **Steps Involved**

1. **Image Upload and Preprocessing:**
   - Two grayscale images are uploaded and preprocessed using histogram equalization to enhance their contrast.

2. **Wavelet Decomposition:**
   - Both images are decomposed into wavelet coefficients (LL, LH, HL, HH) using the 2D Haar wavelet transform.

3. **Image Fusion:**
   - The approximation coefficients (LL) are fused by averaging both images.
   - The detail coefficients (LH, HL, HH) are fused by taking the maximum value at each corresponding position.

4. **Reconstruction:**
   - The fused image is reconstructed by applying the inverse Haar wavelet transform to the fused coefficients.

5. **Denoising:**
   - The fused image is denoised using the Non-Local Means denoising method to reduce noise while preserving important image details.

6. **Visualization:**
   - The original images, enhanced images, fused image, and denoised image are displayed using `matplotlib`.

## **Running the Code**

1. Upload the two image files (`c08_1.tif` and `c08_2.tif`) to the environment.  
2. Run the code in a Python environment like Google Colab or Jupyter Notebook.  
3. View the results in the output, which includes:
   - Original Images  
   - Enhanced Images  
   - Fused Image  
   - Denoised Image

## **Example Code**

```python
import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import exposure, restoration

# Load images
img1 = cv2.imread('c08_1.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('c08_2.tif', cv2.IMREAD_GRAYSCALE)

# Enhance images
img1_enhanced = exposure.equalize_hist(img1.astype(np.float32) / 255.0)
img2_enhanced = exposure.equalize_hist(img2.astype(np.float32) / 255.0)

# Perform Haar wavelet transform
coeffs1 = pywt.dwt2(img1_enhanced, 'haar')
coeffs2 = pywt.dwt2(img2_enhanced, 'haar')

# Extract coefficients
LL1, (LH1, HL1, HH1) = coeffs1
LL2, (LH2, HL2, HH2) = coeffs2

# Fusion of coefficients
LL_fused = (LL1 + LL2) / 2
LH_fused = np.maximum(LH1, LH2)
HL_fused = np.maximum(HL1, HL2)
HH_fused = np.maximum(HH1, HH2)

# Reconstruct fused image
fused_img = pywt.idwt2((LL_fused, (LH_fused, HL_fused, HH_fused)), 'haar')

# Denoise the fused image
denoised_img = restoration.denoise_nl_means(fused_img, h=0.1, fast_mode=True)

# Display images
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('Original Image 1')
axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title('Original Image 2')
axs[0, 2].imshow(fused_img, cmap='gray')
axs[0, 2].set_title('Fused Image')
axs[1, 0].imshow(img1_enhanced, cmap='gray')
axs[1, 0].set_title('Enhanced Image 1')
axs[1, 1].imshow(img2_enhanced, cmap='gray')
axs[1, 1].set_title('Enhanced Image 2')
axs[1, 2].imshow(denoised_img, cmap='gray')
axs[1, 2].set_title('Denoised Image')

for ax in axs.ravel():
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## **Results**

- **Enhanced Images:** The images after histogram equalization, providing better contrast.  
- **Fused Image:** The combined image that merges features from both images using wavelet fusion.  
- **Denoised Image:** The image after noise reduction, making it clearer and more detailed.
  
![image](https://github.com/user-attachments/assets/0c15caaa-4260-4ba9-a086-5932fe17a700)



## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
