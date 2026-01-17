import numpy as np, os, math

def mse(a,b): 
    d=a-b
    return np.mean(d*d, axis=(1,2,3))

def psnr(a,b):
    # per-sample dynamic range from GT
    dr = (b.max(axis=(1,2,3)) - b.min(axis=(1,2,3)))
    m = mse(a,b)
    out=[]
    for i in range(len(m)):
        if m[i] == 0:
            out.append(float('inf'))
        else:
            out.append(20*math.log10(float(dr[i])) - 10*math.log10(float(m[i])))
    return np.array(out)

for tag, d in [("GAN", "data_out/wind_mrhr_gan"), ("CNN","data_out/wind_mrhr_cnn")]:
    gt = np.load(os.path.join(d, "dataGT.npy"))
    sr = np.load(os.path.join(d, "dataSR.npy"))
    m = mse(sr, gt)
    p = psnr(sr, gt)
    print("\n", tag)
    print("MSE per sample:", m)
    print("PSNR per sample:", p)
    print("Mean±Std PSNR:", float(p.mean()), float(p.std()))
