from PIL import Image
import numpy as np
def calc_mean_RGB(train_data_path):
  total_r = 0
  total_g = 0
  total_b = 0
  pixel_count = 0

  for filename in os.listdir(train_data_path):
    if filename.endswith(".jpg",".png",".jpeg"):
      img_path = os.path.join(train_data_path,filename)
      with Image.open(img_path) as img:
          img_array = np.array(img)
          total_r +=np.sum(img_array[:,:,0])
          total_g += np.sum(img_array[:,:,1])
          total_b += np.sum(img_array[:,:,2])
          pixel_count += img_array.shape[0]*img_array.shape[1]

  mean_r = total_r/pixel_count
  mean_g = total_g/pixel_count
  mean_b = total_b/pixel_count

  return mean_r, mean_g, mean_b

# Usage
dataset_path = '/content/sample_data/mnist_train_small.csv'
mean_r, mean_g, mean_b = calc_mean_RGB(dataset_path)

print(f"Mean R: {mean_r}")
print(f"Mean G: {mean_g}")
print(f"Mean B: {mean_b}")
