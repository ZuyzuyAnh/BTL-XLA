import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def myCEDnew():
    myalpha = 0.01   # Tăng giá trị myalpha để khuếch tán mạnh hơn
    sigma = 1.0      # Tăng giá trị sigma để bộ lọc Gaussian mạnh hơn
    T = 25           # Tăng số vòng lặp để khuếch tán lâu hơn
    rho = 6          # Tăng giá trị rho để bộ lọc khuếch tán theo hướng gradient mạnh hơn
    C = 1            # Để nguyên
    im = plt.imread('1.png')  # Đọc ảnh từ file
    if im.ndim == 3:  # Kiểm tra nếu ảnh có 3 kênh màu (RGB)
        im = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])  # Chuyển ảnh màu sang grayscale
    
    numrow, numcol = im.shape
    imorig = im.copy()  # Lưu ảnh gốc
    stepT = 0.15        # Giữ nguyên giá trị stepT
    t = 0
    im = im.astype(np.float64)  # Chuyển đổi kiểu ảnh thành kiểu số thực

    while t < (T - 0.001):
        t += stepT
        
        # 1. Bộ lọc Gaussian K_sigma
        limitX = np.arange(-int(2 * sigma), int(2 * sigma) + 1)
        kSigma = np.exp(-(limitX ** 2) / (2 * sigma ** 2))
        kSigma /= np.sum(kSigma)  # Chuẩn hóa bộ lọc
        usigma = ndimage.convolve(im, kSigma[:, None], mode='nearest')  # Lọc theo chiều dọc
        usigma = ndimage.convolve(usigma, kSigma[None, :], mode='nearest')  # Lọc theo chiều ngang

        # 2. Gradient
        uy, ux = np.gradient(usigma)

        # 3. Bộ lọc Gaussian K_rho
        limitXJ = np.arange(-int(3 * rho), int(3 * rho) + 1)
        kSigmaJ = np.exp(-(limitXJ ** 2) / (2 * rho ** 2))
        kSigmaJ /= np.sum(kSigmaJ)  # Chuẩn hóa bộ lọc
        Jxx = ndimage.convolve(ux ** 2, kSigmaJ[:, None], mode='nearest')
        Jxy = ndimage.convolve(ux * uy, kSigmaJ[:, None], mode='nearest')
        Jyy = ndimage.convolve(uy ** 2, kSigmaJ[:, None], mode='nearest')

        # 4. Biến đổi trục chính (Principal Axis Transformation)
        v2x = np.zeros_like(im)
        v2y = np.zeros_like(im)
        lambda1 = np.zeros_like(im)
        lambda2 = np.zeros_like(im)

        for i in range(numrow):
            for j in range(numcol):
                pixel = np.array([[Jxx[i, j], Jxy[i, j]], [Jxy[i, j], Jyy[i, j]]])
                eigvals, eigvecs = np.linalg.eig(pixel)
                v2x[i, j] = eigvecs[0, 1]
                v2y[i, j] = eigvecs[1, 1]
                lambda1[i, j] = eigvals[0]
                lambda2[i, j] = eigvals[1]
                norm = np.sqrt(v2x[i, j] ** 2 + v2y[i, j] ** 2)
                if norm != 0:
                    v2x[i, j] /= norm
                    v2y[i, j] /= norm

        v1x = -v2y
        v1y = v2x

        # 5. Tính toán ma trận khuếch tán
        di = lambda1 - lambda2
        lambda1 = myalpha + (1 - myalpha) * np.exp(-C / (di ** 2))
        lambda2 = myalpha

        Dxx = lambda1 * v1x ** 2 + lambda2 * v2x ** 2
        Dxy = lambda1 * v1x * v1y + lambda2 * v2x * v2y
        Dyy = lambda1 * v1y ** 2 + lambda2 * v2y ** 2

        # 6. Phi âm hóa ảnh (non-negativity discretization)
        im = non_negativity_discretization(im, Dxx, Dxy, Dyy, stepT)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imorig, cmap='gray')
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(im, cmap='gray')
    plt.title('Coherence Enhancing Diffusion Filtering')
    plt.show()

    plt.imsave('1CED.png', im, cmap='gray')


def non_negativity_discretization(im, Dxx, Dxy, Dyy, stepT):
    numrow, numcol = im.shape
    px = np.roll(im, -1, axis=0)
    nx = np.roll(im, 1, axis=0)
    py = np.roll(im, -1, axis=1)
    ny = np.roll(im, 1, axis=1)

    a = Dxx
    b = Dxy
    c = Dyy

    # Tính toán các trọng số stencil
    wbR1 = 0.25 * (np.abs(b) - b + np.abs(np.roll(b, -1, axis=1)) - np.roll(b, -1, axis=1))
    wtM2 = 0.5 * ((c + np.roll(c, -1, axis=0)) - (np.abs(b) + np.roll(np.abs(b), -1, axis=0)))
    wbL3 = 0.25 * (np.abs(np.roll(b, 1, axis=1)) - b + np.abs(b) - np.roll(b, 1, axis=1))
    wmR4 = 0.5 * ((a + np.roll(a, -1, axis=0)) - (np.abs(b) + np.roll(np.abs(b), -1, axis=0)))
    wmL6 = 0.5 * ((a + np.roll(a, 1, axis=0)) - (np.abs(b) + np.roll(np.abs(b), 1, axis=0)))
    wtR7 = 0.25 * (np.abs(np.roll(b, -1, axis=1)) - b + np.abs(b) - np.roll(b, -1, axis=1))
    wmB8 = 0.5 * ((c + np.roll(c, 1, axis=0)) - (np.abs(b) + np.roll(np.abs(b), 1, axis=0)))
    wtL9 = 0.25 * (np.abs(np.roll(b, 1, axis=1)) - b + np.abs(b) - np.roll(b, 1, axis=1))

    im += stepT * (
        wbR1 * (px - im) + wtM2 * (py - im) + wbL3 * (nx - im) + wmR4 * (np.roll(px, -1, axis=0) - im) +
        wmL6 * (np.roll(nx, 1, axis=0) - im) + wtR7 * (np.roll(px, -1, axis=1) - im) +
        wmB8 * (np.roll(py, -1, axis=1) - im) + wtL9 * (np.roll(ny, 1, axis=1) - im)
    )

    return im

# Gọi hàm để áp dụng thuật toán
myCEDnew()
