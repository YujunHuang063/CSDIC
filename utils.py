import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow_compression.python.layers import parameterizers
import numpy as np
import pdb

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
_CLASS_NUM = 19

def cal_loss(logits, y, loss_weight=1.0):
    '''
    raw_prediction = tf.reshape(logits, [-1, CLASSES])
    raw_gt = tf.reshape(y, [-1])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, CLASSES - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    '''

    y = tf.reshape(y, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(y,
                                               255)) * loss_weight
    one_hot_labels = tf.one_hot(
        y, _CLASS_NUM, on_value=1.0, off_value=0.0)
    logits = tf.reshape(logits, shape=[-1, _CLASS_NUM])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, weights=not_ignore_mask)

    return tf.reduce_mean(loss)

def compute_mean_iou(total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    mask = (sum_over_col!=0)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag
    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.min((np.sum((denominator != 0).astype(float)), np.sum((sum_over_col != 0).astype(float))))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag[mask] / denominator[mask]

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
      print('    class {}: {:.4f}'.format(i, iou))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    m_iou = float(m_iou)
    print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
    return float(m_iou)

def compute_mean_iou2(total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    mask = (sum_over_col!=0)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag
    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.min((np.sum((denominator != 0).astype(float)), np.sum((sum_over_col != 0).astype(float))))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag[mask] / denominator[mask]

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    m_iou = float(m_iou)
    #print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
    return float(m_iou)    

def compute_accuracy(total_cm):
    """Compute the accuracy via the confusion matrix."""
    denominator = total_cm.sum().astype(float)
    cm_diag_sum = np.diagonal(total_cm).sum().astype(float)
    # If the number of valid entries is 0 (no classes) we return 0.
    accuracy = np.where(
        denominator > 0,
        cm_diag_sum / denominator,
        0)
    accuracy = float(accuracy)
    print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))    

def cal_abc(pred, gt, classes_num):
    """

    :param pred: [batch, height, width]
    :param gt: [batch, height, width]
    :param classes_num:
    :return:
    """
    IoU_0 = []
    IoU = []
    eps = 1e-6

    pred_flatten = np.reshape(pred, -1)
    gt_flatten = np.reshape(gt, -1)
    #print(pred_flatten.shape, gt_flatten.shape)
    aa, bb, cc = [], [], []
    for i in range(0, classes_num):
        a = [pred_flatten == i, gt_flatten != 255]
        a = np.sum(np.all(a, 0))
        aa.append(a)
        b = np.sum(gt_flatten == i)
        bb.append(b)
        c = [pred_flatten == i, gt_flatten == i]
        c = np.sum(np.all(c, 0))
        cc.append(c)
    aa, bb, cc = np.array(aa), np.array(bb), np.array(cc)    
    return aa, bb, cc    

def cal_batch_mIoU(pred, gt, classes_num):
    """

    :param pred: [batch, height, width]
    :param gt: [batch, height, width]
    :param classes_num:
    :return:
    """
    IoU_0 = []
    IoU = []
    eps = 1e-6

    pred_flatten = np.reshape(pred, -1)
    gt_flatten = np.reshape(gt, -1)
    #print(pred_flatten.shape, gt_flatten.shape)

    for i in range(0, classes_num):
        a = [pred_flatten == i, gt_flatten != 255]
        a = np.sum(np.all(a, 0))
        b = np.sum(gt_flatten == i)
        c = [pred_flatten == i, gt_flatten == i]
        c = np.sum(np.all(c, 0))
        iou = c / (a + b - c + eps)
        if b != 0:
            IoU.append(iou)
        IoU_0.append(round(iou, 2))

    IoU_0 = dict(zip(CLASS_NAMES[0:], IoU_0))
    mIoU = np.mean(IoU)
    return mIoU, IoU_0   

def tofloat(x):
  return tf.cast(x, tf.float32)

def toint(x):
  return tf.cast(x, tf.int32)
    
def Padding(x, block_width, mode='SYMMETRIC', rgb_path='sta/Kitti_rgb_mean.npy'):
  if mode=="CONSTANT":
    rgb_mean = np.load(rgb_path)[::-1]
  x_shape = tf.shape(x)
  y_pad = tf.stack([0, tf.cast(tf.ceil(x_shape[1]/block_width)*block_width, tf.int32)-x_shape[1]])
  x_pad = tf.stack([0, tf.cast(tf.ceil(x_shape[2]/block_width)*block_width, tf.int32)-x_shape[2]])
  paddings = tf.stack([[0,0], y_pad, x_pad, [0,0]], axis=0)
  if mode=="CONSTANT":
    xr = tf.pad(x[:, :, :, :1], paddings, constant_values = rgb_mean[0]/255.)
    xg = tf.pad(x[:, :, :, 1:2], paddings, constant_values = rgb_mean[1]/255.)
    xb = tf.pad(x[:, :, :, 2:], paddings, constant_values = rgb_mean[2]/255.)
    x = tf.concat([xr, xg, xb], axis=3)
  else:
    x = tf.pad(x, paddings, mode=mode)  
  return x
  
def Squeeze(x, block_width):
  shape = tf.shape(x)
  x = tf.reshape(tf.transpose(tf.reshape(x, (shape[0], shape[1]//block_width, block_width, shape[2]//block_width, block_width, shape[3])), [0, 1, 3, 5, 2, 4]), (shape[0], shape[1]//block_width, shape[2]//block_width, shape[3]*block_width**2))
  return x
  
def Random_crop(x, block_width):
  x_shape = tf.shape(x)
  width = toint(tf.random_uniform((1, ),minval=tofloat(x_shape[1]-block_width),maxval=tofloat(x_shape[1]+1))) 
  begin = toint(tf.random_uniform((1, ),minval=tofloat(tf.constant([0.])),maxval=tofloat(x_shape[1]-width))) 
  x = tf.slice(x, [0, begin[0], begin[0], 0], [-1, width[0], width[0], -1])
  return x

class RGB2YUV(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, *args, **kwargs):
    super(RGB2YUV, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    z1 = 0.5092
    z2 = 0.19516
    WR = z1/(1+z1+z2)
    WG = 1/(1+z1+z2)
    WB = z2/(1+z1+z2)
    C = np.array([[WR, WG, WB], [-WR/(1-WB)/2, -WG/(1-WB)/2, 0.50000], [0.50000, -WG/(1-WR)/2, -WB/(1-WR)/2]], np.float32)
    self.trans_m = tf.convert_to_tensor(C, dtype=self.dtype)
    self.inv_trans_m = tf.convert_to_tensor(np.linalg.inv(C), dtype=self.dtype)
    super(RGB2YUV, self).build(RGB2YUV)

  def call(self, tensor, reverse=False):
    if not reverse:
      tensor = tf.matmul(tensor, tf.transpose(self.trans_m))
    else:
      tensor = tf.matmul(tensor, tf.transpose(self.inv_trans_m))  
    return tensor 

class Learn_RGB2YUV(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, *args, **kwargs):
    self.create_weight = parameterizers.NonnegativeParameterizer(
    minimum=1e-2)
    super(Learn_RGB2YUV, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.weight = self.create_weight(
          name="weight", shape=[2], dtype=self.dtype,
          getter=self.add_variable, initializer=tf.initializers.ones()) 
    self.one = tf.ones((1)) 
    self.RGB_weight = tf.concat([self.weight, self.one], 0)   
    line1 = self.RGB_weight/tf.reduce_sum(self.RGB_weight) 
    e2 = tf.constant([0, 0, 1], dtype = self.dtype) 
    line2 = (e2-line1)/(1-line1[2])/2 
    e0 = tf.constant([1, 0, 0], dtype = self.dtype) 
    line3 = (e0-line1)/(1-line1[0])/2 
    self.trans_m = tf.stack([line1, line2, line3], axis=0) 
    self.inv_trans_m = tf.linalg.inv(self.trans_m) 

    super(Learn_RGB2YUV, self).build(Learn_RGB2YUV)    

  def call(self, tensor, reverse=False):
    if not reverse:
      tensor = tf.matmul(tensor, tf.transpose(self.trans_m))
    else:
      tensor = tf.matmul(tensor, tf.transpose(self.inv_trans_m))  
    return tensor   

class DCT(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, N, y_low_filter_num, uv_low_filter_num, *args, **kwargs):
    self.N = N
    self.y_low_num = y_low_filter_num
    self.uv_low_num = uv_low_filter_num
    
    N = self.N
    B = np.zeros((N,N), np.float32)

    for i in range(N):
      for j in range(N):
        B[j, i] = np.cos((2*j+1)*np.pi/2/N*i)
        if i==0:
          B[j, i]*=np.sqrt(1/N)
        else:
          B[j, i]*=np.sqrt(2/N)  
    self.Ay = tf.convert_to_tensor(B[:, :self.y_low_num], dtype=tf.float32)
    self.Auv = tf.convert_to_tensor(B[:, :self.uv_low_num], dtype=tf.float32)
    
    N = self.y_low_num
    B = np.zeros((N,N), np.float32)
    for i in range(N):
      for j in range(N):
        B[j, i] = np.cos((2*j+1)*np.pi/2/N*i)
        if i==0:
          B[j, i]*=np.sqrt(1/N)
        else:
          B[j, i]*=np.sqrt(2/N)  
    self.Ayy = tf.convert_to_tensor(B[:, :self.y_low_num], dtype=tf.float32)
    self.Ayuv = tf.convert_to_tensor(B[:, :self.uv_low_num], dtype=tf.float32)

    N = self.uv_low_num
    B = np.zeros((N,N), np.float32)
    for i in range(N):
      for j in range(N):
        B[j, i] = np.cos((2*j+1)*np.pi/2/N*i)
        if i==0:
          B[j, i]*=np.sqrt(1/N)
        else:
          B[j, i]*=np.sqrt(2/N) 
    self.Auvuv = tf.convert_to_tensor(B[:, :self.uv_low_num], dtype=tf.float32)      
    super(DCT, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(DCT, self).build(DCT)

  def ori_to_sub(self, tensor, N, low_num, A, B, reverse = False, img_shape=False):
    shape = tf.shape(tensor)
    tensor = tf.transpose(tf.reshape(tensor, (shape[0], shape[1]//N, N, shape[2]//N, N, shape[3])), [0, 1, 3, 5, 2, 4])
    tensor = tf.matmul(tf.transpose(A), tf.matmul(tensor, A))
    B_t = tf.transpose(B)
    tensor = tf.matmul(tf.transpose(B_t), tf.matmul(tensor, B_t))
    if not img_shape:
      tensor = tf.reshape(tensor, (shape[0], shape[1]//N, shape[2]//N, shape[3]*low_num**2))
    else:
      tensor =tf.reshape(tf.transpose(tensor, [0, 1, 4, 2, 5, 3]), [shape[0], shape[1]//N*low_num, shape[2]//N*low_num, shape[3]])  
    '''
    if not reverse:
      tensor = tf.transpose(tf.reshape(tensor, (shape[0], shape[1]//self.N, self.N, shape[2]//self.N, self.N, shape[3])), [0, 1, 3, 5, 2, 4])
      tensor = tf.matmul(tf.transpose(self.A), tf.matmul(tensor, self.A))
      tensor = tf.reshape(tensor, (shape[0], shape[1]//self.N, shape[2]//self.N, shape[3]*self.low_num**2))
      #tensor =tf.reshape(tf.transpose(tensor, [0, 1, 4, 2, 5, 3]), [shape[0], shape[1]//self.N*self.low_num, shape[2]//self.N*self.low_num, 3])  
    else:
      #tensor = tf.transpose(tf.reshape(tensor, (shape[0], shape[1]//self.low_num, self.low_num, shape[2]//self.low_num, self.low_num, shape[3])), [0, 1, 3, 5, 2, 4])
      tensor = tf.reshape(tensor, [shape[0], shape[1], shape[2], shape[3]//self.low_num**2, self.low_num, self.low_num])
      tensor = tf.matmul(self.A, tf.matmul(tensor, tf.transpose(self.A)))
      tensor =tf.reshape(tf.transpose(tensor, [0, 1, 4, 2, 5, 3]), [shape[0], shape[1]*self.N, shape[2]*self.N, 1]) 
    '''   
    return tensor

  def image_reshape(self, tensor, reverse=False):
    shape = tf.shape(tensor)
    if not reverse:
      tensor1 = tensor[:, :, :, :self.y_low_num**2]
      tensor1 = self.reshape(tensor1, self.y_low_num)
      tensor2 = tensor[:, :, :, self.y_low_num**2:self.y_low_num**2+self.uv_low_num**2]
      tensor2 = self.reshape(tensor2, self.uv_low_num)
      tensor2 = self.ori_to_sub(tensor2, self.uv_low_num, self.y_low_num, self.Auvuv, self.Ayuv, False, img_shape=True)
      tensor3 = tensor[:, :, :, self.y_low_num**2+self.uv_low_num**2:]
      tensor3 = self.reshape(tensor3, self.uv_low_num)
      tensor3 = self.ori_to_sub(tensor3, self.uv_low_num, self.y_low_num, self.Auvuv, self.Ayuv, False, img_shape=True)
      tensor = tf.concat([tensor1, tensor2, tensor3], 3)
      tensor.set_shape([None, None, None, 3])
      return tensor
    else:
      tensor1 = tensor[:, :, :, :1]
      tensor1 = self.reshape(tensor1, self.y_low_num, True)
      tensor2 = tensor[:, :, :, 1:2]
      tensor2 = self.ori_to_sub(tensor2, self.y_low_num, self.uv_low_num, self.Ayuv, self.Auvuv)
      #tensor2 = self.reshape(tensor2, self.uv_low_num, True)
      tensor3 = tensor[:, :, :, 2:]
      tensor3 = self.ori_to_sub(tensor3, self.y_low_num, self.uv_low_num, self.Ayuv, self.Auvuv)
      #tensor3 = self.reshape(tensor3, self.uv_low_num, True)
      tensor = tf.concat([tensor1, tensor2, tensor3], 3)
      tensor.set_shape([None, None, None, self.y_low_num**2+2*self.uv_low_num**2])
      return tensor

  def reshape(self, tensor, low_num, reverse=False):
    shape = tf.shape(tensor)
    if not reverse:
      return tf.reshape(tf.transpose(tf.reshape(tensor, (shape[0], shape[1], shape[2], shape[3]//low_num**2, low_num, low_num)), [0, 1, 4, 2, 5, 3]), (shape[0], shape[1]*low_num, shape[2]*low_num, shape[3]//low_num**2))
    else:
      return tf.reshape(tf.transpose(tf.reshape(tensor, (shape[0], shape[1]//low_num, low_num, shape[2]//low_num, low_num, shape[3])), [0, 1, 3, 5, 2, 4]), (shape[0], shape[1]//low_num, shape[2]//low_num, shape[3]*low_num**2))    

  def call(self, tensor, reverse=False):
    if not reverse:
      tensor1 = tensor[:, :, :, :1]
      tensor1 = self.ori_to_sub(tensor1, self.N, self.y_low_num, self.Ay, self.Ayy)  
      tensor2 = tensor[:, :, :, 1:2]
      tensor2 = self.ori_to_sub(tensor2, self.N, self.uv_low_num, self.Auv, self.Auvuv)
      tensor3 = tensor[:, :, :, 2:3]
      tensor3 = self.ori_to_sub(tensor3, self.N, self.uv_low_num, self.Auv, self.Auvuv)
      tensor = tf.concat([tensor1, tensor2, tensor3], 3)
    else:
      tensor1 = tensor[:, :, :, :1]
      tensor1 = self.ori_to_sub(tensor1, self.y_low_num, self.N, self.Ayy, self.Ay, True, img_shape=True)  
      tensor2 = tensor[:, :, :, 1:2]
      tensor2 = self.ori_to_sub(tensor2, self.y_low_num, self.N, self.Ayuv, self.Auv, True, img_shape=True)
      tensor3 = tensor[:, :, :, 2:]
      tensor3 = self.ori_to_sub(tensor3, self.y_low_num, self.N, self.Ayuv, self.Auv, True, img_shape=True)
      tensor = tf.concat([tensor1, tensor2, tensor3], 3)  
    return tensor
'''
  def freq_to_sub_img(self, tensor, y_low_num, uv_low_num, uv_low_num_, reverse=False):
    dctyy = Dct(y_low_num, y_low_num)
    dctuvy = Dct(uv_low_num_, uv_low_num)
    if not reverse:
      tensor1 = tensor[:, :, :, :y_low_num**2]
      tensor1 = dctyy(tensor1, True)
      tensor2 = tensor[:, :, :, y_low_num**2:y_low_num**2+uv_low_num**2]
      tensor2 = dctuvy(tensor2, True)
      tensor3 = tensor[:, :, :, y_low_num**2+uv_low_num**2:]
      tensor3 = dctuvy(tensor3, True)
      if uv_low_num_==y_low_num:
        tensor = tf.concat([tensor1, tensor2, tensor3], 3)  
        return tensor
      else:
        
        return tensor1, tensor2, tensor3  
    else:
      tensor1 = tensor[:, :, :, :1]
      tensor1 = dctyy(tensor1)  
      tensor2 = tensor[:, :, :, 1:2]
      tensor2 = dctuvy(tensor2)
      tensor3 = tensor[:, :, :, 2:3]
      tensor3 = dctuvy(tensor3)
      tensor = tf.concat([tensor1, tensor2, tensor3], 3)
    return tensor    
'''      

class Add_Mean(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, idx, *args, **kwargs):
    self.idx = idx
    super(Add_Mean, self).__init__(*args, **kwargs)

  def build(self, input_shape):

    self.mean = self.add_variable(
        "mean", shape= (1, 1, 1, _NUM - BASE_NUM), dtype=self.dtype,
        initializer=tf.truncated_normal_initializer(stddev=0.2))
    super(Add_Mean, self).build(input_shape)

  def call(self, tensor):
    #pad
    zero = tf.zeros_like(tensor)
    mean = zero[:, :, :, :_NUM-BASE_NUM-self.idx[0]] + self.mean[:, :, :, self.idx[0]:]
    tensor = tf.concat([tensor, mean], axis=3)
    return tensor

class Quantize(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    self._default_quantize_param = parameterizers.NonnegativeParameterizer()
    super(Quantize, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.quantize = self._default_quantize_param(
          name="gamma", shape=[1, 1, 1, self.num_filters], dtype=self.dtype,
          getter=self.add_variable,
          initializer=tf.constant_initializer(1.0)) 
    super(Quantize, self).build(input_shape)

  def call(self, tensor, quantized = True):
    if quantized:
      tensor = tensor / self.quantize
    else:
      tensor = tensor * self.quantize
    return tensor  

def Relu_iden_grad(x):
  g = tf.get_default_graph()
  with g.gradient_override_map({"Relu": "Identity"}):
    x = tf.nn.relu(x)
  return x
     
def read_png(filename1):
  """Loads a PNG image file."""
  string = tf.read_file(filename1)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  
  return image
  
def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

class AverageMeter(object):
  def __init__(self, max_len):
    self.reset()
    self.max_len = max_len

  def reset(self):
    self.val = 0
    self.avg = 0
    self.record = []

  def update(self, val):
    self.record.append(val)
    if len(self.record)>self.max_len:
      del(self.record[0])
    self.val = val
    self.avg = np.sum(self.record) / len(self.record)     