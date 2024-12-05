from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    return peak_signal_noise_ratio(img1, img2, data_range=img1.max() - img1.min())

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    return structural_similarity(img1, img2, multichannel=True, data_range=img1.max() - img1.min())
