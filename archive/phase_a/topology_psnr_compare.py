import numpy as np, os

def mse(a,b): 
    d=a-b
    return float(np.mean(d*d))

def psnr(a,b, data_range=None):
    m = mse(a,b)
    if data_range is None:
        # range from GT
        data_range = float(b.max() - b.min())
    if m == 0:
        return float('inf')
    import math
    return 20*math.log10(data_range) - 10*math.log10(m)

for tag, d in [("GAN", "data_out/wind_mrhr_gan"), ("CNN","data_out/wind_mrhr_cnn")]:
    gt = np.load(os.path.join(d, "dataGT.npy"))
    sr = np.load(os.path.join(d, "dataSR.npy"))
    # compute over both channels
    m = mse(sr, gt)
    p = psnr(sr, gt)
    print(tag, "MSE:", m, "PSNR:", p)
