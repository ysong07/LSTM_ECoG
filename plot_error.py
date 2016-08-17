import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import pdb 
import scipy.io as scipy_io
import cPickle as pkl
import numpy.matlib
#abc = pkl.load(open('test_cell_features_signal','r'))
#pdb.set_trace()
#plot_matrix_hidden1 = np.zeros([20,6000])
#plot_matrix_hidden2 = np.zeros([20,6000])
#for i in range(20):
#    plot_matrix_hidden1[i,:] = abc[i][15,0:6000]
#    plot_matrix_hidden2[i,:] = abc[i][15,6000:12000] 
#   
#
#for i in range(6):
#    plt.imshow(plot_matrix_hidden2[:,i*1000:(i)*1000+100].T, cmap=plt.get_cmap('Greys'))
#    plt.draw()
#    img_name =  "state_gate_"+str(i)+".pdf"
#    plt.savefig(img_name) 
#


"""error plot against time """
#file1 = h5py.File('error_file_4041_ensemble1','r')
#file2 = h5py.File('error_file_4041_ensemble2','r')
#file3 = h5py.File('error_file_4041_ensemble3','r')
#file4 = h5py.File('error_file_4041_ensemble4','r')
file1 = h5py.File('error_file_ensemble0_trained','r')
file3 = h5py.File('error_file_ensemble0_original','r')
file2 = h5py.File('error_file_ensemble1_trained','r')
file4 = h5py.File('error_file_ensemble1_original','r')
file5=  h5py.File('error_file0','r')
file6 = h5py.File('error_file1','r')
#plot_start_seg = 
#plot_end_seg=  
#prediction = (np.asarray(file1['prediction'])+np.asarray(file2['prediction'])+np.asarray(file3['prediction'])+np.asarray(file4['prediction']))/4
prediction = file1['prediction']
truth = file1['truth']
mask = file1['mask']
#plt.plot(truth[:,0,0].squeeze())
#plt.draw()
#plt.savefig('interval.pdf')
tmp1 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)
#abc=  np.sum(tmp,axis=0)
plt.figure()
plt.clf()
#plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'rs-')
#plt.draw()
#plt.ylim([15,60])

prediction = file2['prediction']
truth = file2['truth']
mask = file2['mask']
#plt.plot(truth[:,0,0].squeeze())
#plt.draw()
#plt.savefig('interval.pdf')
tmp2 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)
mask = np.matlib.repmat(np.asarray((tmp1.sum(axis=1) < tmp2.sum(axis=1))),40,1).T
abc=  np.sum(tmp1*mask+tmp2*(1-mask),axis=0)
plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
plt.draw()
plt.ylim([15,60])



prediction = file3['prediction']
truth = file3['truth']
mask = file3['mask']
#plt.plot(truth[:,0,0].squeeze())
#plt.draw()
#plt.savefig('interval.pdf')
tmp3 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)

#plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'b.-')
#plt.draw()
#plt.ylim([15,60])

prediction = file4['prediction']
truth = file4['truth']
mask = file4['mask']
tmp4 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)

mask = np.matlib.repmat(np.asarray((tmp3.sum(axis=1) < tmp4.sum(axis=1))),40,1).T
abc=  np.sum(tmp3*mask+tmp4*(1-mask),axis=0)
plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'k.-')
plt.draw()
plt.ylim([15,60])


prediction = file5['prediction']
truth = file5['truth']
mask = file5['mask']
#plt.plot(truth[:,0,0].squeeze())
#plt.draw()
#plt.savefig('interval.pdf')
tmp5 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)

#plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'b.-')
#plt.draw()
#plt.ylim([15,60])

prediction = file6['prediction']
truth = file6['truth']
mask = file6['mask']
tmp6 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)

mask = np.matlib.repmat(np.asarray((tmp5.sum(axis=1) < tmp6.sum(axis=1))),40,1).T
abc=  np.sum(tmp5*mask+tmp6*(1-mask),axis=0)
plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'b.-')
plt.draw()
plt.ylim([15,60])



plt.legend(['trained','inter','original'])

plt.savefig('error_2.pdf')
#
#
