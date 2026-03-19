from PhIREGANs import PhIREGANs

data_type = "wind_speed"
data_path = "example_data/wind_speed_MR-HR.tfrecord"
r = [5]

# Scalar speed stats should be estimated from the scalar TFRecord itself.
# Leave mu_sig=None for the first run so PhIREGANs computes them directly.
mu_sig = [9.627784358464595, 4.296151331224189]

if __name__ == "__main__":
    phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
    phiregans.setModel_name("models/wind_speed_mr-hr/trained_cnn")
    saved_model = phiregans.pretrain(r=r, data_path=data_path, batch_size=8)
    print("Saved model:", saved_model)
