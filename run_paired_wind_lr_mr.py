from PhIREGANs import PhIREGANs

data_type = 'wind'
data_path = 'example_data/wind_LR-MR.tfrecord'      # may already be paired; if not, see note below
model_path = 'models/wind_lr-mr/trained_gan/gan'
r = [2,5]
mu_sig = [[0.7684, -0.4575], [4.9491, 5.8441]]

if __name__ == '__main__':
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    phiregans.test_paired(r=r, data_path=data_path, model_path=model_path, batch_size=1)
