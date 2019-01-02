# -*- coding: utf-8 -*-
import os
import requests
from io import BytesIO
import numpy as np
import tensorflow as tf
from flask import Flask
from flask import request
from flask import jsonify
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/detect', methods=['POST'])
def upload():
    image_src = request.form.get('image_src')
    detecotr = TOD()
    try:
        count, data = detecotr.save_object_detection_result(image_src)
        status = 1
    except Exception as e:
        print(e)
        status = 2
        count = 0
        data = {}
    output = {'count': count, 'data': data, 'status': status}
    return jsonify(output)


class TOD(object):
    def __init__(self):
        # 修改1
        self.MODEL_NAME = os.getcwd()
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = self.MODEL_NAME + '/ai_label_map.pbtxt'
        self.NUM_CLASSES = 7
        self.PATH_TO_RESULTS = "/home/yq2/Pictures/results/"

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def save_object_detection_result(self, img_src):
        IMAGE_SIZE = (8, 6)
        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # loading ckpt file to graph
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        # Loading label map
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        # Helper code
        with detection_graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                # 根据image_src读取图片
                response = requests.get(img_src)
                image = Image.open(BytesIO(response.content))
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = self.load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                final_score = np.squeeze(scores)
                count = 0
                for i in range(100):
                    if scores is None or final_score[i] > 0.5:
                        count = count + 1
                print('************************************')
                (im_width, im_height) = image.size
                items = []
                for i in range(count):
                    item = {}
                    y_min = boxes[0][i][0] * im_height
                    x_min = boxes[0][i][1] * im_width
                    y_max = boxes[0][i][2] * im_height
                    x_max = boxes[0][i][3] * im_width
                    item['category'] = category_index[classes[0][i]]['name']
                    item['score'] = float(scores[0][i])
                    item['x_min'] = float(x_min)
                    item['y_min'] = float(y_min)
                    item['x_max'] = float(x_max)
                    item['y_max'] = float(y_max)
                    items.append(item)
                return count, items


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
    # app.run()

