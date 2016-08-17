"""Displays reconstruction and future predictions for trained models."""

from data_handler import *
from lstm_combo import *
import lstm

def main():
  model = ReadModelProto(sys.argv[1])
  lstm_autoencoder = LSTMCombo(model)
  data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
  
  # batch_id 
  batch_id = int(sys.argv[4])
  #lstm_autoencoder.Show(data, output_dir='./imgs/mnist_1layer_example.pdf')
  output_dir = './imgs_35/Ecog'
  #lstm_autoencoder.Show(data, output_dir=output_dir) #display with random order
  #lstm_autoencoder.Show_fix_range(data,10000,output_dir=output_dir) #assign a batch id to dataset 36 valid
  #lstm_autoencoder.Show_fix_range(data,batch_id,output_dir=output_dir) #assign a batch id to dataset 36 valid
  lstm_autoencoder.Show_fix_range(data,1000,output_dir=output_dir)#assign a batch id to dataset 41
  # lstm_autoencoder.Show_fix_range(data,1500,output_dir=output_dir) # assign a batch id to 33 36
  #lstm_autoencoder.Show_fix_range(data,0,output_dir=output_dir) #gaussian noise sequence
  #lstm_autoencoder.Get_Prediction_error(data,'error_file_ensemble1_trained',data.indices_.max())

  #lstm_autoencoder.Extract_feature(data,"./test_feature")
if __name__ == '__main__':
  # Set the board
  board_id = int(sys.argv[3])
  board = LockGPU(board=board_id)
  print 'Using board', board

  #cm.CUDAMatrix.init_random(42)
  #np.random.seed(42)
  main()
