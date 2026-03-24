from PhIREGANs import PhIREGANs

data_type = 'wind_speed_fixed'
data_path = 'example_data_fixed/wind_speed_MR-HR.tfrecord'
model_path = 'models_fixed/wind_speed_mr-hr/trained_cnn/cnn/cnn'
data_out_path = 'data_out_fixed/wind_speed_mrhr_cnn'
r = [5]

if __name__ == '__main__':
    probe = PhIREGANs(data_type=data_type, mu_sig=None)
    probe.set_mu_sig(data_path, batch_size=1)
    print('Using scalar mu_sig:', probe.mu_sig)

    phiregans = PhIREGANs(data_type=data_type, mu_sig=probe.mu_sig)
    phiregans.set_data_out_path(data_out_path)
    phiregans.test_paired(r=r, data_path=data_path, model_path=model_path, batch_size=1)
