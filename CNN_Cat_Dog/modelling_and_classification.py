import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import pickle

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

''' 全局参数 '''
IMAGE_SIZE = 100
LEARNING_RATE = 1e-4
TRAIN_STEP = 500
TRAIN_SIZE = 12
TEST_STEP = 5
TEST_SIZE = 5

IS_TRAIN = True

SAVE_PATH = './model/'

data_dir = './batch_files'
pic_path = './data/train_company'
all_test_files_dir = './data/test_company'

def load_data(filename):
    """从batch文件中读取图片信息"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'], data['label'], data['filenames']

# 读取数据的类
class InputData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        all_names = []
        for file in filenames:
            data, labels, filename = load_data(file)

            all_data.append(data)
            all_labels.append(labels)
            all_names += filename

        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        print(str(self._data.shape))
        print(str(self._labels.shape))

        self._filenames = all_names

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._indicator:
            self._shuffle_data()

    def _shuffle_data(self):
        # 把数据再混排
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """返回每一批次的数据"""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        batch_filenames = self._filenames[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels, batch_filenames

# 定义一个类
class MyTensor:
    def __init__(self):
        # 载入训练集和测试集
        train_filenames = [os.path.join(data_dir, 'train_batch_%d' % i) for i in range(1, 13)]
        test_filenames = [os.path.join(data_dir, 'test_batch')]
        self.batch_train_data = InputData(train_filenames, True)
        self.batch_test_data = InputData(test_filenames, True)

        #以下参数只用于宣告
        self.x, self.y, self.keep_prob, self.loss, self.y_, self.predict, self.acc, self.train_op, self.saver = \
            None, None, None, None, None, None, None, None, None


    def flow(self):
        self.x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], 'input_data')
        self.y = tf.placeholder(tf.int64, [None], 'output_data')
        self.keep_prob = tf.placeholder(tf.float32)

        fc = self.conv_net(self.x)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=fc)
        self.y_ = tf.nn.softmax(fc)  # 计算每一类的概率
        self.predict = tf.argmax(fc, 1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.y), tf.float32))

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

        print('计算流图已经搭建.')
        # self.x = self.x / 255.0  需不需要这一步？
        # 图片输入网络中

    # 训练
    def myTrain(self):
        acc_list = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(TRAIN_STEP):
                train_data, train_label, _ = self.batch_train_data.next_batch(TRAIN_SIZE)

                eval_ops = [self.loss, self.acc, self.train_op]
                eval_ops_results = sess.run(eval_ops, feed_dict={
                    self.x: train_data,
                    self.y: train_label,
                    self.keep_prob: 0.7
                })
                loss_val, train_acc = eval_ops_results[0:2]

                acc_list.append(train_acc)
                if (i+1) % 1 == 0:
                    acc_mean = np.mean(acc_list)
                    print('step:{0},loss:{1:.5},acc:{2:.5},acc_mean:{3:.5}'.format(
                        i+1, loss_val, train_acc, acc_mean
                    ))
                if (i+1) % 1000 == 0:
                    test_acc_list = []
                    for j in range(TEST_STEP):
                        test_data, test_label, _ = self.batch_test_data.next_batch(TRAIN_SIZE)
                        acc_val = sess.run([self.acc], feed_dict={
                            self.x: test_data,
                            self.y: test_label,
                            self.keep_prob: 1.0
                        })
                        test_acc_list.append(acc_val)
                    print('[Test ] step:{0}, mean_acc:{1:.5}'.format(
                        i+1, np.mean(test_acc_list)
                    ))
            # 保存训练后的模型
            os.makedirs(SAVE_PATH, exist_ok=True)
            self.saver.save(sess, SAVE_PATH + 'my_model.ckpt')

    #用于测试训练集中的固定测试集
    def myTest(self):
        with tf.Session() as sess:
            # 此为加载模型
            model_file = tf.train.latest_checkpoint(SAVE_PATH)
            self.saver.restore(sess, save_path=model_file)

            test_acc_list = []
            #predict_list = []
            for j in range(TEST_STEP):
                test_data, test_label, test_name = self.batch_test_data.next_batch(TEST_SIZE)
                for each_data, each_label, each_name in zip(test_data, test_label, test_name):
                    acc_val, y__, pre, test_img_data = sess.run(
                        [self.acc, self.y_, self.predict, self.x],
                        feed_dict={
                            self.x: each_data.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3),
                            self.y: each_label.reshape(1),
                            self.keep_prob: 1.0
                        }
                    )

                    #predict_list.append(pre[0])

                    test_acc_list.append(acc_val)

                    # 把测试结果显示出来
                    self.compare_test(each_label, pre[0], y__[0], each_name)
            #print('[Test ] mean_acc:{0:.5}'.format(np.mean(test_acc_list)))

    def main(self):
        self.flow()
        if IS_TRAIN is True:
            self.myTrain()
        else:
            self.myTest()

    #多张图片
    def final_classify(self):

        all_test_filenames = os.listdir(all_test_files_dir)
        print(all_test_filenames)
        if IS_TRAIN is False:
            self.flow()
            with tf.Session() as sess:
                #此为加载模型
                model_file = tf.train.latest_checkpoint(SAVE_PATH)
                self.saver.restore(sess, save_path=model_file)

                predict_list = []
                for each_filename in all_test_filenames:
                    each_data = self.get_img_data(os.path.join(all_test_files_dir, each_filename))
                    #img_original = cv.imread(os.path.join(all_test_files_dir, each_filename))
                    y__, pre, test_img_data = sess.run(
                        [self.y_, self.predict, self.x],
                        feed_dict={
                            self.x: each_data.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)),
                            self.keep_prob: 1.0
                        }
                    )
                    predict_list.append(pre[0])
                    #print(self.result(pre[0], each_filename))
                    #self.classify(img_original, pre[0], y__[0], each_filename)
        else:
            print('now is training model...')

    # 当前图片（单张）
    def final_classify_single(self):
        tf.reset_default_graph()
        all_test_filenames = os.listdir(all_test_files_dir)
        if IS_TRAIN is False:
            self.flow()
            with tf.Session() as sess:
                # 此为加载模型
                model_file = tf.train.latest_checkpoint(SAVE_PATH)
                self.saver.restore(sess, save_path=model_file)
                each_data = self.get_img_data(os.path.join(all_test_files_dir, all_test_filenames[0]))
                # img_original = cv.imread(os.path.join(all_test_files_dir, each_filename))
                y__, pre, test_img_data = sess.run(
                    [self.y_, self.predict, self.x],
                    feed_dict={
                        self.x: each_data.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)),
                        self.keep_prob: 1.0
                    }
                )

                return self.result(pre[0])
                # self.classify(img_original, pre[0], y__[0], each_filename)
        else:
            print('now is training model...')

    #用于存储
    @staticmethod
    def classify(img, output, probability, img_name):

        classes = ['business_certificate', 'certificate_copy', 'doctor', 'id', 'sign']
        single_image = img  # * 255
        if output == 0:
            output_dir = 'business_certificate/'
        elif output == 1:
            output_dir = 'certificate_copy/'
        elif output == 2:
            output_dir = 'doctor/'
        elif output == 3:
            output_dir = 'id/'
        elif output == 4:
            output_dir = 'sign/'


        os.makedirs(os.path.join('./classiedResult', output_dir), exist_ok=True)
        cv.imwrite(os.path.join('./classiedResult', output_dir, img_name), single_image)
        print('输入的图片名称:{0}，预测得有{1:5}的概率是{2}:{3}'.format(
            img_name,
            probability[output],
            output,
            classes[output]
        ))

    # 用于显示结果
    @staticmethod
    def result(output):
        classes = ['business_certificate', 'certificate_copy', 'doctor', 'id', 'sign']
        result_str = "{0}".format(classes[output])
        return result_str

    @staticmethod
    def compare_test(input_label, output, probability, img_name):
        classes = ['business_certificate', 'certificate_copy', 'doctor', 'id', 'sign']
        if input_label == output:
            result = '正确'
        else:
            result = '错误'
        print('测试【{0}】, 预测得是->{1}<-的概率:{2:.5}, 输入的图片名称:{3}'.format(
            result, classes[output], probability[output], img_name
        ))

    @staticmethod
    def conv_net(x):
        conv1_1 = tf.layers.conv2d(x, 16, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1, 16, (3, 3), padding='same', activation=tf.nn.relu, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, (2, 2), (2, 2), name='pool1')

        conv2_1 = tf.layers.conv2d(pool1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1, 32, (3, 3), padding='same', activation=tf.nn.relu, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, (2, 2), (2, 2), name='pool2')

        conv3_1 = tf.layers.conv2d(pool2, 64, (3, 3), padding='same', activation=tf.nn.relu, name='conv3_1')
        conv3_2 = tf.layers.conv2d(conv3_1, 64, (3, 3), padding='same', activation=tf.nn.relu, name='conv3_2')
        pool3 = tf.layers.max_pooling2d(conv3_2, (2, 2), (2, 2), name='pool3')

        conv4_1 = tf.layers.conv2d(pool3, 128, (3, 3), padding='same', activation=tf.nn.relu, name='conv4_1')
        conv4_2 = tf.layers.conv2d(conv4_1, 128, (3, 3), padding='same', activation=tf.nn.relu, name='conv4_2')
        pool4 = tf.layers.max_pooling2d(conv4_2, (2, 2), (2, 2), name='pool4')

        flatten = tf.layers.flatten(pool4)  # 把网络展平，以输入到后面的全连接层

        fc1 = tf.layers.dense(flatten, 512, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 256, tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 5, None)  # 得到输出fc3

        return fc3


    @staticmethod
    def get_img_data(img_name):
        img = cv.imread(img_name)
        resized_img = cv.resize(img, (100, 100))
        img_data = np.array(resized_img)
        return img_data

    @staticmethod
    def get_original_img_data(img_name):
        img = cv.imread(img_name)
        img_data = np.array(img)
        return img_data


if __name__ == '__main__':

    my_tensor = MyTensor()
    my_tensor.main()  # 用于训练或测试

    #my_tensor.final_classify()  # 用于最后的分类
