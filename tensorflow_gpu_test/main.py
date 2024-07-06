from rock_paper_scissors import  full_process
from datetime import datetime



if __name__ == '__main__':
    import tensorflow as tf
    print(f"device: {tf.test.gpu_device_name()}")

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
        
    print("start: ", datetime.now())
    full_process()
    print("end: ", datetime.now())