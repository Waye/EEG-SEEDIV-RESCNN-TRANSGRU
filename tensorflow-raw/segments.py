# segements
import numpy as np

X89 = np.load("D:/BigData/SEED_IV/SEED_IV/DE0.5s/X89.npy")
print('{}-{}'.format('x_train shape', X89.shape))#x_train shape-(203970, 8, 9, 5)
img_rows, img_cols, num_chan = 8, 9, 5
falx = X89
falx = falx.reshape((15, int(X89.shape[0] / 15), img_rows, img_cols, num_chan)) #falx shape-(15, 13598, 8, 9, 5)
print('{}-{}'.format('falx shape', falx.shape))
t = 6 #(0.5s ->3s segement  6 pieces)

def segment_data(falx, t, lengths, labels):
  """Segments data into fixed-length segments with corresponding labels.

  Args:
    falx: The input data array.
    t: The length of each segment.
    lengths: A list of lengths for each segment.
    labels: A list of labels corresponding to each segment.

  Returns:
    new_x: The segmented data array.
    new_y: The corresponding label array.
  """

  # Calculate boundaries from lengths
  boundaries = np.cumsum(lengths)
  print('{}-{}'.format('boundaries', boundaries))

  # Calculate the total number of segments needed
  #total_segments = sum(np.ceil(np.array(lengths) / t).astype(int))
  total_segments = 3348
  print('{}-{}'.format('total_segments', total_segments))

  # Pre-allocate new_x with correct dimensions
  new_x = np.empty([falx.shape[0], total_segments, t, 8, 9, 5])  
  new_y = np.array([])

  for nb in range(falx.shape[0]):
    z = 0
    i = 0
    for j, bound in enumerate(boundaries):
      while i + t <= bound:
        # Assign segments directly, taking all 6 time steps at once
        new_x[nb, z] = falx[nb, i:i + t]  # Assign to the first t indices of the segment
        new_y = np.append(new_y, labels[j])
        i = i + t
        z = z + 1
      i = bound
    print('{}-{}'.format(nb, z))
  return new_x, new_y

lengths =[336,190,398,260,176,324,306,418,290,338,100,220,434,338,518,282,136,358,280,96,224,224,350,274,
          442, 202, 278,292,428,222,278, 368, 276, 166, 480, 100, 292, 216, 352, 122, 374, 392, 364, 86, 298, 352, 196, 152,
          340, 260, 184, 364, 386, 212, 516, 186, 208, 128, 414, 330, 314, 154, 230, 354, 114, 140, 366, 178, 318, 310, 330, 312]
all_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3,
             2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1,
             1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
new_x, new_y = segment_data(falx, 6, lengths, all_label)

print('{}-{}'.format('new_x shape', new_x.shape))
print('{}-{}'.format('new_y shape', new_y.shape))

np.save('D:/BigData/SEED_IV/SEED_IV/DE0.5s/t'+str(t)+'x_89.npy', new_x)#new_x shape-(15, 2247, 6, 8, 9, 5)
np.save('D:/BigData/SEED_IV/SEED_IV/DE0.5s/t'+str(t)+'y_89.npy', new_y)#new_y shape-(33705,)
