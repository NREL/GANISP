import os
import sys
sys.path.append('util')
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from time import strftime, time
from utils import plot_SR_data
from cgan_network import cGAN_NETWORK
from plotsUtil import *

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

class condPhIREGANs:
    # Network training meta-parameters
    DEFAULT_N_EPOCHS = 10 # Number of epochs of training
    DEFAULT_LEARNING_RATE = 1e-4 # Learning rate for gradient descent (may decrease to 1e-5 after initial training)
    DEFAULT_EPOCH_SHIFT = 0 # If reloading previously trained network, what epoch to start at
    DEFAULT_SAVE_EVERY = 10 # How frequently (in epochs) to save model weights
    DEFAULT_PRINT_EVERY = 2 # How frequently (in iterations) to write out performance

    def __init__(self, data_type, N_epochs=None, learning_rate=None, epoch_shift=None, save_every=None, print_every=None, mu_sig=None,g_n_resBlocks=None, g_n_filters=None,d_firstNFilt=None,idSim=None):

        self.N_epochs      = N_epochs if N_epochs is not None else self.DEFAULT_N_EPOCHS
        self.learning_rate = learning_rate if learning_rate is not None else self.DEFAULT_LEARNING_RATE
        self.epoch_shift   = epoch_shift if epoch_shift is not None else self.DEFAULT_EPOCH_SHIFT
        self.save_every    = save_every if save_every is not None else self.DEFAULT_SAVE_EVERY
        self.print_every   = print_every if print_every is not None else self.DEFAULT_PRINT_EVERY

        self.data_type = data_type
        self.mu_sig = mu_sig
        self.qoi_shape = None
        self.data_shape = None

        self.g_n_resBlocks = g_n_resBlocks
        self.g_n_filters = g_n_filters
        self.d_firstNFilt = d_firstNFilt

        # Set various paths for where to save data
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.idSim         = idSim
        self.model_name    = '/'.join(['models'+str(self.idSim), self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def setSave_every(self, in_save_every):
        self.save_every = in_save_every

    def setPrint_every(self, in_print_every):
        self.print_every = in_print_every

    def setEpochShift(self, shift):
        self.epoch_shift = shift

    def setNum_epochs(self, in_epochs):
        self.N_epochs = in_epochs

    def setLearnRate(self, learn_rate):
        self.learning_rate = learn_rate

    def setModel_name(self, in_model_name):
        self.model_name = in_model_name

    def set_data_out_path(self, in_data_path):
        self.data_out_path = in_data_path
    
    def reset_run_id(self):
        self.run_id        = '-'.join([self.data_type, strftime('%Y%m%d-%H%M%S')])
        self.model_name    = '/'.join(['models'+str(self.idSim), self.run_id])
        self.data_out_path = '/'.join(['data_out', self.run_id])

    def train_g(self,model,qoi,z,condMean_true,condVar_true,rep_size,opt):
        with tf.GradientTape() as tape:
            x_gen = model.generator([qoi,z],training=True)
            lossValue = model.generator_Loss(qoi,x_gen,condMean_true,condVar_true,rep_size)
        grads = tape.gradient(lossValue, (model.generator).trainable_weights)
        opt.apply_gradients(zip(grads, (model.generator).trainable_weights))

    def train_d(self,model,x_data,x_gen,opt):
        with tf.GradientTape() as tape:
            lossValue = model.discriminator_Loss(x_data,x_gen)
        grads = tape.gradient(lossValue, (model.discriminator).trainable_weights)
        opt.apply_gradients(zip(grads, (model.discriminator).trainable_weights))

    def train(self, data_path, model_path=None, batch_size=10, rep_size=10, n_z=16, alpha=[1., 0.01, 1.]):
        '''
            This method trains the generator using a disctiminator/adversarial training.

            inputs:
                data_path    - (string) path of training data file to load in
                model_path   - (string) path of previously pretrained or trained model to load
                batch_size   - (int) number of samples to grab per batch. Decrease if running out of memory
                rep_size     - (int) number of distinct realizations to generate from each conditional variable
                alpha        - (float array) scaling values for the content, adversarial, and diversity losses

            output:
                g_saved_model - (string) path to the trained generator model
        '''

        if self.mu_sig is None:
            self.set_mu_sig(data_path, batch_size)

        print(self.mu_sig)

        # Get shape of qoi and data
        self.set_qoi_shape(data_path)
        self.set_data_shape(data_path)
        n_qoi = self.qoi_shape[0]
        n_data = self.data_shape[0]

        print('Initializing network ...', end=' ')
        model = cGAN_NETWORK(n_qoi=n_qoi, n_data=n_data, n_z=n_z, rep_size=rep_size, alpha=alpha, mu_sig=self.mu_sig,
                             g_n_resBlocks=self.g_n_resBlocks, g_n_filters=self.g_n_filters,d_firstNFilt=self.d_firstNFilt)
        generator = model.generator
        discriminator = model.discriminator
   
        if model_path is not None:
            print('Loading previously trained network...', end=' ')
            generator.load_weights('G_'+model_path)
            discriminator.load_weights('D_'+model_path)
            print('Done.')


        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        print('Load data ...', end=' ')
        ds_train = tf.data.TFRecordDataset(data_path)
        batched_ds_train = ds_train.map(lambda xx: self._parse_train(xx, self.mu_sig)).shuffle(buffer_size=1024).batch(batch_size)
        print('Training network ...')

        # Make folders that contain the checkpoints
        g_model_dr  = 'G_'+self.model_name
        d_model_dr = 'D_'+self.model_name
        os.makedirs(g_model_dr,exist_ok=True)
        os.makedirs(d_model_dr,exist_ok=True)
  
        # Headers of log files
        f = open(g_model_dr+'/log.csv','a+')
        f.write('epoch;con_loss;adv_loss;div_loss;nStep\n')
        f.close()
        f = open(d_model_dr+'/log.csv','a+')
        f.write('epoch;adv_loss;nStep\n')
        f.close()


        # Start training
        iters = 0
        for epoch in range(self.epoch_shift+1, self.epoch_shift+self.N_epochs+1):
            print('Epoch: '+str(epoch))
            start_time = time()

            epoch_g_loss, epoch_d_loss, N = 0, 0, 0 

            # Loop through training data
            for element in batched_ds_train:
                batch_idx, batch_qoi, batch_data, batch_mean, batch_std = element
                batch_qoi, batch_data = np.repeat(batch_qoi, rep_size, axis=0), np.repeat(batch_data, rep_size, axis=0)
                N_batch = batch_qoi.shape[0]
                batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch, n_z, 1])

                # Initial training of the discriminator and generator
                batch_gen = model.generator([batch_qoi,batch_z],training=False)
                #print("batch_gen log 0 = ",tf.reduce_max(batch_gen))
                self.train_d(model,batch_data,batch_gen,optimizer)
                self.train_g(model,batch_qoi,batch_z,batch_mean,batch_std,rep_size,optimizer)
                
                #calculate current discriminator losses
                gl = model.generator_Loss(batch_qoi,batch_gen,batch_mean,batch_std,rep_size)
                dl = model.discriminator_Loss(batch_data,batch_gen)
                g_al = np.mean(model.generator_adversarialLoss(batch_gen))           
 
                #print("gl = ",gl)               
                #print("dl = ",dl)               
                #batch_gen = model.generator([batch_qoi,batch_z],training=False)
                #print("batch_gen log 1 = ",tf.reduce_max(batch_gen))
                #stop 
                gen_count = 1
                while (dl < 0.460) and gen_count < 10:
                    self.train_g(model,batch_qoi,batch_z,batch_mean,batch_std,rep_size,optimizer)
                    # Get losses
                    batch_gen = model.generator([batch_qoi,batch_z],training=False)
                    gl = model.generator_Loss(batch_qoi,batch_gen,batch_mean,batch_std,rep_size)
                    #g_al = np.mean(model.generator_adversarialLoss(batch_gen))
                    dl = model.discriminator_Loss(batch_data,batch_gen)
                    gen_count += 1

                dis_count = 1
                while (dl >= 0.6) and dis_count < 10:
                #while (dl >= 0.65) and dis_count < 10:
                    batch_gen = model.generator([batch_qoi,batch_z],training=False)
                    self.train_d(model,batch_data,batch_gen,optimizer)
                    # Get losses
                    gl = model.generator_Loss(batch_qoi,batch_gen,batch_mean,batch_std,rep_size)
                    dl = model.discriminator_Loss(batch_data,batch_gen)
                    dis_count += 1

                epoch_g_loss += gl*N_batch
                epoch_d_loss += dl*N_batch
                N += N_batch
                
                iters += 1
                if (iters % self.print_every) == 0:
                    gl = model.generator_Loss(batch_qoi,batch_gen,batch_mean,batch_std,rep_size)
                    dl = model.discriminator_Loss(batch_data,batch_gen)
                    g_cl = model.contentLoss(batch_qoi,batch_gen)
                    g_al = model.generator_adversarialLoss(batch_gen)
                    g_dl = model.diversityLoss(batch_qoi,batch_gen,batch_mean,batch_std,rep_size)

                    print('Number of G steps=%d, Number of D steps=%d' %(gen_count, dis_count))
                    print('G loss=%.5f, Content loss=%.5f, Adversarial loss=%.5f, Diversity loss=%.5f' %(gl, np.mean(g_cl), np.mean(g_al), np.mean(g_dl)))
                    print('D loss=%.5f' %(dl))
                    print('', flush=True)
                
                    # Log of training
                    f = open('G_'+self.model_name+'/log.csv','a+')
                    f.write(str(int(epoch))+';'+
                            str(np.mean(g_cl.numpy()))+';'+
                            str(np.mean(g_al.numpy()))+';'+
                            str(np.mean(g_dl.numpy()))+';'+
                            str(gen_count)+
                            '\n')
                    f.close()
                    f = open('D_'+self.model_name+'/log.csv','a+')
                    f.write(str(int(epoch))+';'+
                            str(np.mean(dl.numpy()))+';'+
                            str(dis_count)+
                            '\n')
                    f.close()

            
            if (epoch % self.save_every) == 0:
                g_model_dr  = '/'.join(['G_'+self.model_name, '/epoch{0:05d}'.format(epoch), 'gan'])
                d_model_dr = '/'.join(['D_'+self.model_name, '/epoch{0:05d}'.format(epoch), 'gan'])
                os.makedirs('G_'+self.model_name+'/epoch{0:05d}'.format(epoch),exist_ok=True)
                os.makedirs('D_'+self.model_name+'/epoch{0:05d}'.format(epoch),exist_ok=True)
                (model.generator).save_weights(g_model_dr)
                (model.discriminator).save_weights(d_model_dr)

                
          

            g_loss = epoch_g_loss/N
            d_loss = epoch_d_loss/N

            print('Epoch generator training loss=%.5f, discriminator training loss=%.5f' %(g_loss, d_loss))
            print('Epoch took %.2f seconds\n' %(time() - start_time), flush=True)

        g_model_dr  = '/'.join(['G_'+self.model_name, '/final', 'gan'])
        d_model_dr = '/'.join(['D_'+self.model_name, '/final', 'gan'])
        os.makedirs('G_'+self.model_name+'/final',exist_ok=True)
        os.makedirs('D_'+self.model_name+'/final',exist_ok=True)
        (model.generator).save_weights(g_model_dr)
        (model.discriminator).save_weights(d_model_dr)

        print('Done.')

        return self.model_name+'/final/gan'

    def test(self, data_path, model_path, n_z=16, batch_size=10, rep_size=10, plot_data=False):
        '''
            This method loads a previously trained model and runs it on test data

            inputs:
                r          - (int array) should be array of prime factorization of amount of super-resolution to perform
                data_path  - (string) path of test data file to load in
                model_path - (string) path of model to load in
                batch_size - (int) number of images to grab per batch. decrease if running out of memory
                rep_size   - (int) number of distinct realizations to generate from each LR field
                plot_data  - (bool) flag for whether or not to plot LR and SR images
        '''
        #print('MAKE SURE NO DATA RESCALING IS DONE')
        if self.mu_sig is None:
            print('mu_sig needs to be defined')
            sys.exit()
        #    self.set_mu_sig(data_path, batch_size)

        # Get shape of qoi and data
        self.set_qoi_shape(data_path)
        self.set_data_shape(data_path)
        n_qoi = self.qoi_shape[0]
        n_data = self.data_shape[0]

        print('Initializing network ...', end=' ')
        model = cGAN_NETWORK(n_qoi=n_qoi, n_data=n_data, n_z=n_z, rep_size=rep_size, mu_sig=self.mu_sig,
                             g_n_resBlocks=self.g_n_resBlocks, g_n_filters=self.g_n_filters,d_firstNFilt=self.d_firstNFilt)
        generator = model.generator
        discriminator = model.discriminator
  
        print('Loading previously trained network...', end=' ')
        generator.load_weights('G_'+model_path)
        print('Done.')


        print('Load data ...', end=' ')
        ds_test = tf.data.TFRecordDataset(data_path)
        batched_ds_test = ds_test.map(lambda xx: self._parse_train(xx, self.mu_sig)).batch(batch_size)

        # Start testing
        iters = 0
        start_time = time()

        # Get number of testing point
        nSnapTest = 0
        for element in batched_ds_test:
            nSnapTest += element[0].shape[0]


        # Loop through training data
        qoi_out =	np.empty(shape=(0,n_qoi,1))
        gen_out = 	np.empty(shape=(0,rep_size,n_data,1))
        data_out = 	np.empty(shape=(0,n_data,1))
        condMean_out = 	np.empty(shape=(0,n_data,1))
        condStd_out = 	np.empty(shape=(0,n_data,1))
        condMean_true = np.empty(shape=(0,n_data,1))
        condStd_true =	np.empty(shape=(0,n_data,1))
        for element in batched_ds_test:

            batch_idx, batch_qoi, batch_data, batch_mean, batch_std = element
            N_batch = batch_qoi.shape[0]
            batch_qoi_repeated = np.repeat(batch_qoi, rep_size, axis=0)
            batch_z = np.random.uniform(low=-1.0, high=1.0, size=[N_batch*rep_size, n_z, 1])

            # Initial training of the discriminator and generator
            batch_gen = model.generator([batch_qoi_repeated,batch_z],training=False)
            batch_gen = np.reshape(batch_gen,(N_batch, rep_size, n_data, 1))    
            batch_gen = batch_gen*self.mu_sig[1][1] + self.mu_sig[1][0]
            batch_condMean, batch_condStd = np.mean(batch_gen, axis=1), np.std(batch_gen, axis=1)
        
            qoi_out = 		np.concatenate((qoi_out, batch_qoi*self.mu_sig[0][1] + self.mu_sig[0][0]))
            gen_out = 		np.concatenate((gen_out, batch_gen))
            data_out = 		np.concatenate((data_out, batch_data*self.mu_sig[1][1] + self.mu_sig[1][0]))
            condMean_out = 	np.concatenate((condMean_out,batch_condMean))
            condStd_out = 	np.concatenate((condStd_out,batch_condStd))
            condMean_true = 	np.concatenate((condMean_true,batch_mean*self.mu_sig[1][1]))
            condStd_true = 	np.concatenate((condStd_true,batch_std*self.mu_sig[1][1]))
             
        os.makedirs(self.data_out_path)
        np.save(self.data_out_path+'/test_qoi.npy', qoi_out)
        np.save(self.data_out_path+'/test_data.npy',data_out)
        np.save(self.data_out_path+'/test_gen.npy',gen_out)
        np.save(self.data_out_path+'/test_condMean_gen.npy',condMean_out)
        np.save(self.data_out_path+'/test_condStd_gen.npy',condStd_out)
        np.save(self.data_out_path+'/test_condMean_true.npy',condMean_true)
        np.save(self.data_out_path+'/test_condStd_true.npy',condStd_true)

        print(self.data_out_path)
        print('Done.')


    def _bytes_feature(self,value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
          value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_feature(self,value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    
    
    def _int64_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _parse_train(self,example_proto,mu_sig=None):
        """
        Dictionary for reading LR and HR, and conditional moments
        """
        image_feature_description = {
            'index': tf.io.FixedLenFeature([], tf.int64),
            'qoi': tf.io.FixedLenFeature([], tf.string),
            'n_qoi': tf.io.FixedLenFeature([], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string),
            'n_data': tf.io.FixedLenFeature([], tf.int64),
            'mean': tf.io.FixedLenFeature([], tf.string),
            'std': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example_proto, image_feature_description)
        idx = example['index']
        n_qoi = example['n_qoi']
        n_data = example['n_data']
        mean = tf.cast(tf.io.decode_raw(example['mean'], tf.float64), tf.float32)
        mean = tf.reshape(mean, (n_data, 1))
        std = tf.cast(tf.io.decode_raw(example['std'], tf.float64), tf.float32)
        std = tf.reshape(std, (n_data, 1))
        data = tf.cast(tf.io.decode_raw(example['data'], tf.float64), tf.float32)
        data = tf.reshape(data, (n_data, 1))
        qoi = tf.cast(tf.io.decode_raw(example['qoi'], tf.float64), tf.float32)
        qoi = tf.reshape(qoi, (n_qoi, 1))
        if mu_sig is not None:
            qoi = (qoi - mu_sig[0][0])/mu_sig[0][1]
            data = (data - mu_sig[1][0])/mu_sig[1][1]
            mean = mean/mu_sig[1][1]
            std = std/mu_sig[1][1]
        #if mu_sig is not None:
        #    qoi = qoi/(mu_sig[1][1]**2)
        #    data = data/mu_sig[1][1]
        #    mean = mean/mu_sig[1][1]
        #    std = std/mu_sig[1][1]

        return idx, qoi, data, mean, std

    def _parse_test_(self, serialized_example, mu_sig=None):
        feature = {
          'index'    : tf.FixedLenFeature([], tf.int64),
          'data_LR'  : tf.FixedLenFeature([], tf.string),
          'h_LR'     : tf.FixedLenFeature([], tf.int64),
          'w_LR'     : tf.FixedLenFeature([], tf.int64),
          'c'        : tf.FixedLenFeature([], tf.int64)}

        example = tf.parse_single_example(serialized_example, feature)

        idx = example['index']

        h_LR, w_LR = example['h_LR'], example['w_LR']

        c = example['c']

        data_LR = tf.decode_raw(example['data_LR'], tf.float64)

        data_LR = tf.reshape(data_LR, (h_LR, w_LR, c))

        if mu_sig is not None:
            data_LR = (data_LR - mu_sig[0])/mu_sig[1]

        return idx, data_LR

    def set_mu_sig(self, data_path, batch_size):
        '''
            Compute mu, sigma 
            inputs:
                data_path - (string) should be the path to the TRAINING DATA since mu and sigma are
                            calculated based on the trainind data regardless of if pretraining,
                            training, or testing.
                batch_size - number of samples to grab each iteration. will be passed in directly
                             from pretrain, train, or test method by default.
            outputs:
                sets self.mu_sig
        '''
        print('Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train).batch(batch_size)
    
        N, mu_d, sigma_d, mu_q, sigma_q = 0, 0, 0, 0, 0
        for _, qoi, data, _, _  in dataset:
            N_batch  = data.shape[0]
            N_new = N + N_batch

            mu_d_batch = np.mean(data, axis=(0, 1, 2)) 
            sigma_d_batch = np.var(data, axis=(0, 1, 2)) 
            mu_q_batch = np.mean(qoi, axis=(0, 1, 2)) 
            sigma_q_batch = np.var(qoi, axis=(0, 1, 2)) 

            sigma_d = (N/N_new)*sigma_d + (N_batch/N_new)*sigma_d_batch + (N*N_batch/N_new**2)*(mu_d - mu_d_batch)**2
            mu_d = (N/N_new)*mu_d + (N_batch/N_new)*mu_d_batch

            sigma_q = (N/N_new)*sigma_q + (N_batch/N_new)*sigma_q_batch + (N*N_batch/N_new**2)*(mu_q - mu_q_batch)**2
            mu_q = (N/N_new)*mu_q + (N_batch/N_new)*mu_q_batch

            N = N_new

        self.mu_sig = [[mu_q, np.sqrt(sigma_q)],[mu_d,np.sqrt(sigma_d)]]
        #self.mu_sig = [[0.0, 1.0],[mu_d,np.sqrt(sigma_d)]]

        print('Done.')

    def set_qoi_shape(self, data_path):
        '''
            Get the shape of the qoi
            inputs:
                data_path - (string) path to training or testing data
            outputs:
                sets self.qoi_shape
        '''
        print('Set qoi shape: Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train)
        for _, qoiExample, _, _, _  in dataset:
            self.qoi_shape = qoiExample.shape   
            break 
        
        print('Done.')

    def set_data_shape(self, data_path):
        '''
            Get the shape of the data
            inputs:
                data_path - (string) path to training or testing data
            outputs:
                sets self.data_shape
        '''
        print('Set data shape: Loading data ...', end=' ')
        dataset = tf.data.TFRecordDataset(data_path)
        dataset = dataset.map(self._parse_train)
        for _, _, dataExample, _, _  in dataset:
            self.data_shape = dataExample.shape   
            break 

        print('Done.')
