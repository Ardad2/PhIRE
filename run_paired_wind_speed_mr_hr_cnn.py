from PhIREGANs import PhIREGANs

data_type  = "wind_speed"
data_path  = "example_data/wind_speed_MR-HR.tfrecord"
model_path = "models/wind_speed_mr-hr/trained_cnn/cnn/cnn"
r = [5]

mu_sig = [9.627784358464595, 4.296151331224189]

if __name__ == "__main__":
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    phiregans.set_data_out_path("data_out/wind_speed_mrhr_cnn")
    phiregans.test_paired(r=r, data_path=data_path, model_path=model_path, batch_size=1)
