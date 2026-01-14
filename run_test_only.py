from PhIREGANs import PhIREGANs

# ---- Choose one config block ----
# WIND: LR -> MR (10x as 2x then 5x)
data_type = 'wind'
data_path = 'example_data/wind_LR-MR.tfrecord'
model_path = 'models/wind_lr-mr/trained_gan/gan'   # pretrained GAN
# model_path = 'models/wind_lr-mr/trained_cnn/cnn' # pretrained CNN baseline
r = [2, 5]
mu_sig = [[0.7684, -0.4575], [4.9491, 5.8441]]

if __name__ == '__main__':
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)

    # TEST / inference only (no training)
    phiregans.test(
        r=r,
        data_path=data_path,
        model_path=model_path,
        batch_size=1
    )
