from PhIREGANs import PhIREGANs

data_type = 'wind_nrel_original'
data_path = 'reference_nrel_example_data/wind_MR-HR.tfrecord'
model_path = 'models/wind_mr-hr/trained_gan/gan'
data_out_path = 'data_out_nrel_original/wind_mrhr_gan'
r = [5]
mu_sig = [[0.7684, -0.4575], [5.02455, 5.9017]]

if __name__ == '__main__':
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    phiregans.set_data_out_path(data_out_path)
    phiregans.test_paired(r=r, data_path=data_path, model_path=model_path, batch_size=1)
