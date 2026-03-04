from PhIREGANs import PhIREGANs

data_type = 'wind'
data_path = 'example_data/wind_MR-HR.tfrecord'
model_path = 'models/wind_mr-hr/trained_cnn/cnn'   # CNN baseline checkpoint
r = [5]
mu_sig = [[0.7684, -0.4575], [5.02455, 5.9017]]

if __name__ == '__main__':
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    phiregans.test_paired(r=r, data_path=data_path, model_path=model_path, batch_size=1)
