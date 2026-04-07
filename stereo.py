import numpy as np
#==============No additional imports allowed ================================#

def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    HEIGHT, WIDTH, CHANNELS = len(img), len(img[0]), len(img[0, 0])
    normalized = np.zeros((HEIGHT, WIDTH, (CHANNELS * patchsize ** 2)), dtype=np.float32)
    
    # (1) take a patchsize x patchsize window around the pixel,
    patches = np.lib.stride_tricks.sliding_window_view(img, (patchsize, patchsize), axis=(0, 1))

    # (2) compute and subtract the mean for every channel
    means = np.mean(patches, axis=(3, 4), keepdims=True)
    descriptors = patches - means
    
    # (3) flatten it into a single vector
    descriptors = descriptors.reshape(HEIGHT - patchsize + 1, WIDTH - patchsize + 1, (CHANNELS * patchsize ** 2))

    # (4) normalize the vector by dividing by its L2 norm
    norms = np.linalg.norm(descriptors, axis=2, keepdims=True)
    results = np.where(norms >= 1e-6, descriptors / norms, 0)
    
    # (5) store it in the (i,j)th location in the output
    p = patchsize//2
    normalized[p : p + results.shape[0], p : p + results.shape[1], :] = results
    
    return normalized

def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    H, W, C = img_right.shape
    ncc_vol = np.zeros((dmax, H, W), dtype=np.float32)

    descriptors_right = get_ncc_descriptors(img_right, patchsize)
    descriptors_left = get_ncc_descriptors(img_left, patchsize)

    for d in range(dmax):
        ncc_vol[d, :, : W - d] = np.sum(descriptors_right[:, : W - d, :] * descriptors_left[:, d :, :], axis=2)
    return ncc_vol



def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    disparity = np.argmax(ncc_vol, axis=0)
    return disparity





    
