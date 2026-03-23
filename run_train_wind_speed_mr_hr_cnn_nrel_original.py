from PhIREGANs import PhIREGANs

data_type = 'wind_speed_nrel_original'
data_path = 'reference_nrel_example_data/wind_speed_MR-HR.tfrecord'
model_name = 'models_nrel_original/wind_speed_mr-hr/trained_cnn'
r = [5]

if __name__ == '__main__':
    phiregans = PhIREGANs(data_type=data_type, mu_sig=None)
    phiregans.setModel_name(model_name)
    saved_model = phiregans.pretrain(r=r, data_path=data_path, batch_size=8)
    print('Saved model:', saved_model)
    print('Computed scalar mu_sig:', phiregans.mu_sig)
