from condPhIREGANs import *

data_type = '1D'
data_path = 'data/modelNN_kseTest_diversity.tfrecord'
model_path = 'epoch00078/gan'
mu_sig = [[1.7240014844838842, 0.20744504957106663], [-5.0281370042055505e-06, 1.3130419828853965]]

if __name__ == '__main__':


    g_n_resBlocks=16
    g_n_filters=32
    d_firstNFilt=8
    BATCH_SIZE =128
    N_Z =16


    phiregans = condPhIREGANs(data_type=data_type,mu_sig=mu_sig,
                              g_n_resBlocks=g_n_resBlocks, g_n_filters=g_n_filters,d_firstNFilt=d_firstNFilt)

    # TEST
    REP_SIZE = 32
    phiregans.test(data_path=data_path,
                   model_path=model_path,
                   n_z = N_Z,
                   batch_size=BATCH_SIZE,
                   rep_size=REP_SIZE)


