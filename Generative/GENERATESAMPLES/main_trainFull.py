from condPhIREGANs import *

data_type = '1D'
data_path = 'data/modelNN_kseTrain_diversity.tfrecord'

if __name__ == '__main__':

    g_n_resBlocks=16
    g_n_filters=32
    d_firstNFilt=8
    BATCH_SIZE = 64
    N_Z = 16

    phiregans = condPhIREGANs(data_type=data_type, learning_rate=1e-4,mu_sig=None, save_every=1, print_every=1,
                              g_n_resBlocks=g_n_resBlocks, g_n_filters=g_n_filters,d_firstNFilt=d_firstNFilt,idSim=15)


    # PRETRAIN
    REP_SIZE = 1
    phiregans.setNum_epochs(1)
    preTrainedModel_dir = phiregans.train(data_path=data_path,
                                model_path=None,
                                batch_size=BATCH_SIZE,
                                n_z=N_Z,
                                rep_size=REP_SIZE,
                                alpha=[10,0.1,0])

    # TRAIN
    REP_SIZE = 32
    phiregans.setNum_epochs(1000)
    phiregans.reset_run_id()
    model_dir = phiregans.train(data_path=data_path,
                                model_path=preTrainedModel_dir,
                                batch_size=BATCH_SIZE,
                                n_z=N_Z,
                                rep_size=REP_SIZE,
                                alpha=[1000,0.1,1])




