import os
import cv2
import tensorflow as tf


class GenTFRecords:
    def __init__(self, images_path, output_path):
        """
        构造函数
        :param images_path: 需要写进文件的图片集合所在路径
        :param output_path: 输出文件的指定路径
        """
        self.input_dir = images_path
        self.output_dir = output_path

    @staticmethod
    def read_file(filename):
        """
        读入图片
        :param filename: 文件路径
        :return: ndarray 格式的文件内容和文件数组形状
        """
        img = cv2.imread(filename)
        img_shape = img.shape
        return img, img_shape

    def gen_tfrecords(self,
                      name='train.tfrecords',
                      gen_class_label=False,
                      gen_img_label=False,
                      img_label_path=None):
        """
        生成指定名字的 tfrecords 文件
        :param name: 输出文件名字
        :param gen_class_label: 是否生成表示类别的标签，用数字表示，标签来源于图片命名，类似 "_label.jpg"
        :param gen_img_label: 是否生成图片表示的标签
        :param img_label_path: 如果生成图片标签，那么这些图片标签对应的路径
        """
        # 列举需要写入文件的文件名
        images = os.listdir(self.input_dir)
        if gen_img_label:
            # 作为标签的图片的名字必须和对应训练样本相同
            self._gen_label_tfrecords(images, img_label_path)

        # 输出文件路径和名字
        output_tfrecords_name = os.path.join(self.output_dir, name)
        writer = tf.python_io.TFRecordWriter(output_tfrecords_name)  # 这个需要放在 os.listdir() 后面，防止误读

        # 开始写入
        for image in images:
            img_path = os.path.join(self.input_dir, image)
            img, img_shape = self.read_file(img_path)
            bytes_data = img.tobytes()

            if gen_class_label:
                # label = int(image.split('_')[-1].split('.')[0])  # 这里有个很奇怪的错误提示（虽然并不会错，为了美观，见下）
                label = int(str(image.split('_')[-1]).split('.')[0])  # 那这样呢...
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={  # 创建了一个字典，读的时候按照字典键读取对应的值
                            'img_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=list(img_shape)  # 值必须是列表
                                )
                            ),
                            'img_data': tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[bytes_data]
                                )
                            ),
                            'img_label': tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[label]
                                )
                            )
                        }
                    )
                )
            else:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'img_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=list(img_shape)
                                )
                            ),
                            'img_data': tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[bytes_data]
                                )
                            )
                        }
                    )
                )

            writer.write(example.SerializeToString())  # 将 example 序列化
        writer.close()

    def _gen_label_tfrecords(self, files, label_path):
        output_label_tfrecords_path = os.path.join(self.output_dir, 'label.tfrecords')

        writer = tf.python_io.TFRecordWriter(output_label_tfrecords_path)
        for image in files:
            img_name = os.path.join(label_path, image)
            img, img_shape = self.read_file(img_name)
            bytes_data = img.tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=list(img_shape)
                            )
                        ),
                        'img_data': tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[bytes_data]
                            )
                        )
                    }
                )
            )

            writer.write(example.SerializeToString())
        writer.close()
