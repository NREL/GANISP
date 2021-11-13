import sys
sys.path.append('util')
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflowUtils as tfu
from myProgressBar import printProgressBar

# Filenames to write
filenameToWriteTrain = 'kseTrain.tfrecord'
filenameToWriteTest = 'kseTest.tfrecord'
Result = np.load('KSDATA.npz')


shapeuu = Result['uu'].shape
shapeqoi = Result['qoi'].shape
uu = np.reshape(np.rollaxis(Result['uu'],2,1),(shapeuu[0]*shapeuu[2],shapeuu[1]))
qoi = np.reshape(np.rollaxis(Result['qoi'],2,1),(shapeqoi[0]*shapeqoi[2],shapeqoi[1]))



print('CHECK DIMS AND PLOTS')
print(qoi.shape)
print(uu.shape)
plt.plot(uu[0,:])
plt.show()

print('SPLIT TRAIN AND TEST')
qoi_train, qoi_test, data_train, data_test = train_test_split(qoi, uu, test_size=0.01, random_state=42, shuffle=True)


n_qoi =  qoi_train.shape[1]
n_data = data_train.shape[1]

n_train = qoi_train.shape[0]
n_test = qoi_test.shape[0]


print('WRITE TRAIN')
with tf.io.TFRecordWriter(filenameToWriteTrain) as writer:
    counter=0
    printProgressBar(0, n_train, prefix = 'Output snapshot Train ' + str(0) + ' / ' +str(n_train),suffix = 'Complete', length = 50)
    for i in range(n_train):

        # ~~~~ Prepare the data
        qoi_snapshot =   qoi_train[i]
        data_snapshot = data_train[i]

        # ~~~~ Write the data
        tf_example = tfu.data_example(counter,n_data,n_qoi,bytes(qoi_snapshot),bytes(data_snapshot))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, n_train, prefix = 'Output snapshot Train ' + str(counter) + ' / ' +str(n_train),suffix = 'Complete', length = 50)

print('WRITE TEST')
with tf.io.TFRecordWriter(filenameToWriteTest) as writer:
    counter=0
    printProgressBar(0, n_test, prefix = 'Output snapshot Test ' + str(0) + ' / ' +str(n_test),suffix = 'Complete', length = 50)
    for i in range(n_test):

        # ~~~~ Prepare the data
        qoi_snapshot =   qoi_test[i]
        data_snapshot = data_test[i]

        # ~~~~ Write the data
        tf_example = tfu.data_example(counter,n_data,n_qoi,bytes(qoi_snapshot),bytes(data_snapshot))
        writer.write(tf_example.SerializeToString())

        counter += 1
        printProgressBar(counter, n_test, prefix = 'Output snapshot Test ' + str(counter) + ' / ' +str(n_test),suffix = 'Complete', length = 50)

