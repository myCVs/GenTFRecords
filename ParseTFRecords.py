import tensorflow as tf


class ImageDataReading:
    def __init__(self, file_path, batch_size=1, epochs=1, shuffle=False):
        """
        :param file_path: tfrecords 文件地址
        :param batch_size: 清晰明了
        :param epochs: 同上
        :param shuffle: 混洗数据
        """
        self._path = file_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    @staticmethod
    def _parse_tfrecords_file(example_proto, fixed=True, features=None):
        """
        解析 tfrecords 文件结构，数据集格式固定，按照该固定格式进行解析即可
        :param example_proto: protocol buffer message
        :param fixed: 使用固定解析格式？
        :param features: features
                         features={
                            key1: tf.FixedLenFeature(list_shape, dtype)
                            key2: ...
                            ...
                         }
        :return: 解析后的 batch 列表，列表按顺序是 features 中的键值顺序
        """
        if fixed:
            features = {
                'img_shape': tf.FixedLenFeature((3,), dtype=tf.int64),
                'img_data': tf.FixedLenFeature((), dtype=tf.string)
            }
            parsed_dict = tf.parse_single_example(example_proto, features=features)
            image_data = tf.decode_raw(parsed_dict['img_data'], out_type=tf.uint8)

            return parsed_dict['img_shape'], image_data
        else:
            parsed_dict = tf.parse_single_example(example_proto, features=features)

            val = []
            keys = list(features.keys())
            for key in keys:
                if key == 'img_data':
                    image_data = tf.decode_raw(parsed_dict['img_data'], out_type=tf.uint8)
                    val.append(image_data)
                else:
                    val.append(parsed_dict[key])

            return val

    def _gen_dataset(self):
        """
        创建一个 Dataset
        :return: 返回 Dataset
        """
        data_set = tf.data.TFRecordDataset(self._path).map(self._parse_tfrecords_file)
        if self.shuffle:
            data_set = data_set.shuffle(1000).repeat(self.epochs).batch(self.batch_size, drop_remainder=True)
            return data_set
        else:
            data_set = data_set.repeat(self.epochs).batch(self.batch_size, drop_remainder=True)
            return data_set

    def next_batch(self):
        data_set = self._gen_dataset()
        iterator = data_set.make_one_shot_iterator()
        next_batch = iterator.get_next()

        return next_batch
