import io
import os
import argparse
import logging

import tensorflow.compat.v1 as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import parse_frame, int64_feature, int64_list_feature, bytes_feature
from utils import bytes_list_feature, float_list_feature


def create_tf_example(filename, encoded_jpeg, annotations):
    """
    convert to tensorflow object detection API format
    args:
    - filename [str]: name of the image
    - encoded_jpeg [bytes-likes]: encoded image
    - annotations [list]: bboxes and classes
    returns:
    - tf_example [tf.Example]
    """
    #Load image and fetch dimensions
    image = Image.open(io.BytesIO(encoded_jpeg))
    w, h = image.size
    
    #Map classes to ID, based on dataset Proto
    class_mapping = {
        1:'vehicle',
        2:'pedestrian',
        3: 'sign',
        4:'cyclist'
    }
    #Format for TFRecord
    img_format = b'jpg'
    
    #For bounding boxes
    min_xs, min_ys, max_xs, max_ys, text_classes, classes = [],[],[],[],[],[]
    
    #Encode filename
    filename = filename.encode('utf8')
    
    #Convert all annotations
    for annotation in annotations:
        min_x, min_y = annotation.box.center_x - 0.5 * annotation.box.length, annotation.box.center_y - 0.5 * annotation.box.width
        max_x, max_y = annotation.box.center_x + 0.5 * annotation.box.length, annotation.box.center_y + 0.5 * annotation.box.width
        min_xs.append(min_x / w)
        max_xs.append(max_x / w)
        min_ys.append(min_y / h)
        max_ys.append(max_y / h)    
        text_classes.append(class_mapping[annotation.type].encode('utf8'))
        classes.append(annotation.type)
        

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(h),
        'image/width': int64_feature(w),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(img_format),
        'image/object/bbox/min_x': float_list_feature(min_xs),
        'image/object/bbox/max_x': float_list_feature(max_xs),
        'image/object/bbox/min_y': float_list_feature(min_ys),
        'image/object/bbox/max_y': float_list_feature(max_ys),
        'image/object/class/text': bytes_list_feature(text_classes),
        'image/object/class/label': int64_list_feature(classes),
    }))
                      
    return tf_example


def process_tfr(path):
    """
    process a waymo tf record into a tf api tf record
    """
    # create processed data dir
    file_name = os.path.basename(path)

    logging.info(f'Processing {path}')
    writer = tf.python_io.TFRecordWriter(f'output/{file_name}')
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        encoded_jpeg, annotations = parse_frame(frame)
        filename = file_name.replace('.tfrecord', f'_{idx}.tfrecord')
        tf_example = create_tf_example(filename, encoded_jpeg, annotations)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        help='Waymo Open dataset tf record')
    args = parser.parse_args()  
    process_tfr(args.path)