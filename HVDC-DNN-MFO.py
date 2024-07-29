import numpy as np
import random as rd
import copy
from math import exp, cos, pi
# import Experiment_dir
import os
import pywt

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Concatenate, Flatten, MaxPooling1D, \
    BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, multiply, Reshape, ReLU, \
    Add, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax

from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model, to_categorical
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import pickle
import os

# 列出所有可用的物理GPU设备
physical_devices = tf.config.experimental.list_physical_devices('GPU')

# 确保至少有两块GPU设备可用
if len(physical_devices) > 1:
    # 设置第二块GPU设备
    tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
else:
    print("没有足够的GPU设备可用。")

X_train = np.load('/external_data/data_train.npy')
y_train = np.load('/external_data/label_train.npy')
x_val = np.load('/external_data/data_val.npy')
y_val = np.load('/external_data/label_val.npy')
X_test = np.load('/external_data/data_test.npy')
y_test = np.load('/external_data/label_test.npy')


# 均值归一化函数
def mean_normalization(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1e-8  # 避免标准差为零
    return (data - mean) / std

# 对每个样本进行均值归一化
X_train = mean_normalization(X_train)
x_val = mean_normalization(x_val)
X_test = mean_normalization(X_test)
class_num = len(np.unique(y_train))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

"""
  参数修改：
"""
wavelet = 'db3'

level = 3

# X_train 加一个维度
a0 = np.expand_dims(X_train, axis=-1)
y_train = to_categorical(y=y_train, num_classes=class_num)

a1 = pywt.dwt(a0, wavelet, axis=-2)[0]
a2 = pywt.dwt(a1, wavelet, axis=-2)[0]
a3 = pywt.dwt(a2, wavelet, axis=-2)[0]
# a4 = pywt.dwt(a3, wavelet, axis=-2)[0]
print(y_train.shape)

print(a0.shape)
print(a1.shape)
print(a2.shape)
print(a3.shape)

v1 = np.expand_dims(x_val, -1)
v2 = pywt.dwt(v1, wavelet, axis=-2)[0]
v3 = pywt.dwt(v2, wavelet, axis=-2)[0]
v4 = pywt.dwt(v3, wavelet, axis=-2)[0]
y_val = to_categorical(y=y_val, num_classes=class_num)

t1 = np.expand_dims(X_test, -1)
t2 = pywt.dwt(t1, wavelet, axis=-2)[0]
t3 = pywt.dwt(t2, wavelet, axis=-2)[0]
t4 = pywt.dwt(t3, wavelet, axis=-2)[0]
# t5 = pywt.dwt(t4, wavelet, axis=-2)[0]
y_test = to_categorical(y=y_test, num_classes=class_num)

print(t2.shape)

input_1 = Input(shape=a1.shape[1:])
input_2 = Input(shape=a2.shape[1:])
input_3 = Input(shape=a3.shape[1:])


def SE(se_ratio=16, input_channel=32, activation="relu", data_format='channels_last', ki="he_normal"):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''

    def f(input_x):
        channel_axis = -1
        input_channels = input_channel
        reduced_channels = input_channels // se_ratio

        # Squeeze operation
        x = GlobalAveragePooling1D()(input_x)
        x = Dense(reduced_channels, kernel_initializer=ki)(x)
        x = Activation(activation)(x)
        # Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = multiply([input_x, x])

        return x

    return f


# ### CBAM block

# In[17]:


def CBAM(se_ratio=8, input_channel=32):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''

    def f(input_x):
        channel_axis = -1
        input_channels = input_channel
        reduced_channels = input_channels // se_ratio

        # Channel Attention
        avgpool = GlobalAveragePooling1D()(input_x)
        maxpool = GlobalMaxPooling1D()(input_x)
        # shared MLP
        Dense_layer1 = Dense(reduced_channels, activation='relu', kernel_initializer='he_normal', use_bias=False)
        Dense_layer2 = Dense(input_channels, activation='relu', kernel_initializer='he_normal', use_bias=False)
        avg_out = Dense_layer2(Dense_layer1(avgpool))
        max_out = Dense_layer2(Dense_layer1(maxpool))

        channel = layers.add([avg_out, max_out])
        channel = Activation('sigmoid')(channel)

        channel = Reshape((1, input_channels))(channel)
        channel_out = tf.multiply(input_x, channel)

        #
        # Spatial Attention
        avgpool = tf.reduce_mean(channel_out, axis=2, keepdims=True)
        maxpool = tf.reduce_max(channel_out, axis=2, keepdims=True)
        spatial = Concatenate(axis=2)([avgpool, maxpool])
        print(spatial.shape)
        spatial = Conv1D(1, 7, strides=1, padding='same')(spatial)
        spatial_out = Activation('sigmoid')(spatial)
        print(spatial.shape)
        CBAM_out = tf.multiply(channel_out, spatial_out)

        return CBAM_out

    return f


# In[18]:
def create_branch(input_layer, kernel_size, gru, se_ratio):
    x = Conv1D(filters=32, kernel_size=kernel_size, padding='same')(input_layer)
    x = Conv1D(filters=32, kernel_size=kernel_size, padding='same')(x)
    x = MaxPooling1D(pool_size=kernel_size, strides=3)(x)
    x = CBAM(se_ratio=se_ratio, input_channel=32)(x)
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same')(x)
    x = Conv1D(filters=64, kernel_size=kernel_size, padding='same')(x)
    x = MaxPooling1D(pool_size=kernel_size, strides=3)(x)
    x = CBAM(se_ratio=se_ratio, input_channel=64)(x)
    x = Conv1D(filters=128, kernel_size=kernel_size, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=kernel_size, padding='same')(x)
    x = MaxPooling1D(pool_size=kernel_size, strides=3)(x)
    x = CBAM(se_ratio=se_ratio, input_channel=128)(x)
    x = GRU(units=gru, return_sequences=False)(x)
    flatten = Flatten()(x)
    return flatten


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = (rd.uniform(lb[j], ub[j]))

    return X, lb, ub


def fobj(x, Iter, i):
    iter = Iter
    num_pop = i + 1
    x1, x2, x3, x4 = x
    print(f'learn_rate：{x1},kernel_size:{x2}，gru_size:{x3},se_ratio:{x4}')
    x1_learning_rate = x1  # 学习率
    x2_kernel_size = int(x2)
    gru1 = int(x3)
    se_ratio = int(x4)

    flatten1 = create_branch(input_1, x2_kernel_size, gru1, se_ratio)
    flatten2 = create_branch(input_2, x2_kernel_size, gru1, se_ratio)
    flatten3 = create_branch(input_3, x2_kernel_size, gru1, se_ratio)
    concatenate = Concatenate()([flatten1, flatten2, flatten3])
    ##########################################################

    dense1 = Dense(50, activation='relu')(concatenate)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(20, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    output = Dense(class_num, activation='softmax')(dropout2)
    ###########################################################
    model = Model(inputs=[input_1, input_2, input_3], outputs=[output])

    # In[19]:

    def binary_focal_loss(gamma=2, alpha=0.25):
        """
        Binary form of focal loss.
        适用于二分类问题的focal loss

        focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

            p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
            return K.mean(focal_loss)

        return binary_focal_loss_fixed

    # In[21]:

    def categorical_focal_loss(alpha, gamma=2.):
        """
        Softmax version of focal loss.
        When there is a skew between different categories/labels in your data set, you can try to apply this function as a
        loss.
               m
          FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
              c=1
          where m = number of classes, c = class and o = observation
        Parameters:
          alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
          categories/labels, the size of the array needs to be consistent with the number of classes.
          gamma -- focusing parameter for modulating factor (1-p)
        Default value:
          gamma -- 2.0 as mentioned in the paper
          alpha -- 0.25 as mentioned in the paper
        References:
            Official paper: https://arxiv.org/pdf/1708.02002.pdf
            https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
        Usage:
         model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
        """

        alpha = np.array(alpha, dtype=np.float32)

        def categorical_focal_loss_fixed(y_true, y_pred):
            """
            :param y_true: A tensor of the same shape as `y_pred`
            :param y_pred: A tensor resulting from a softmax
            :return: Output tensor.
            """

            # Clip the prediction value to prevent NaN's and Inf's
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

            # Calculate Cross Entropy
            cross_entropy = -y_true * K.log(y_pred)

            # Calculate Focal Loss
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

            # Compute mean loss in mini_batch
            return K.mean(K.sum(loss, axis=-1))

        return categorical_focal_loss_fixed

    # In[22]:

    Reduce = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=6,
                               verbose=0,
                               mode='auto',
                               epsilon=0.0001,
                               cooldown=0,
                               min_lr=0)
    # 早停机制待确定后加入
    # callback = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    model_checkpoint = ModelCheckpoint(filepath='/external_data/hxx/SI_python_code/checkpoint/best_weights.h5',
                                 monitor='val_loss', save_best_only=True, verbose=0)

    # In[23]:

    model.compile(optimizer=Adam(x1_learning_rate),
                  loss=[binary_focal_loss(alpha=.25, gamma=2)],
                  #                 loss=[categorical_focal_loss(alpha=[[.1, .6, .4,]], gamma=2)],
                  #               loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit([a1, a2, a3], y_train, epochs=100,
                        verbose=2, validation_data=([v2, v3, v4], y_val), batch_size=64, callbacks=[Reduce, model_checkpoint]
                        # , callbacks=[Reduce, callback, model_checkpoint]
                        #                       ,callbacks=[Reduce]
                        )
    # Extract the training and validation loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # In[24]:

    # model.evaluate([t2, t3, t4], y_test)

    # In[25]:

    # from tensorflow.keras.models import save_model
    # save_model(model, filepath='/external_data/hxx/SI_python_code/results/mfo-llc/db3_3.h5')

    # In[26]:

    from tensorflow.keras.models import load_model

    def binary_focal_loss_fixed(y_true, y_pred):
        gamma = 2
        alpha = 0.25
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    model = load_model('/external_data/hxx/SI_python_code/checkpoint/best_weights.h5',
                       custom_objects={'binary_focal_loss_fixed': binary_focal_loss_fixed})

    # In[27]:

    y_pred = np.argmax(model.predict([t2, t3, t4]), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    fitness = 1-round(accuracy, 4)
    print('当前的iter:', iter)
    print('本次fitness:', fitness)
    print('accuracy:', round(accuracy, 4))
    print('precision:', round(precision, 4))
    print('recall:', round(recall, 4))
    print('f1-sore:', round(f1, 4))
    print(cm)
    eval = [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)]
    NEWCM = np.array([[2543, 69], [82, 2577]])
    print(type(cm))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    CF, ax = plot_confusion_matrix(NEWCM,
                                   show_normed=False, figure=fig

                                   , class_names=["高压直流输\n电干扰事件", "正常地磁时\n变观测样本"]
                                   )
    ax.set_xlabel('预测类别')

    ax.set_ylabel('真实类别')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xticks(rotation=0)
    plt.tight_layout()
    return fitness, train_loss, val_loss, eval, train_acc, val_acc


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fobj, t):
    pop = X.shape[0]
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    evals = []
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i], train_loss, val_loss, eval, train_acc, val_acc = fobj(X[i, :], t, i)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        evals.append(eval)
    return fitness, train_losses, val_losses, evals, train_accuracies, val_accuracies


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def MFO(pop, dim, lb, ub, MaxIter, fobj):
    a = 2;  # 参数
    t = 0;
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness, train_losses, val_losses, evals, train_accuracies, val_accuracies = CaculateFitness(X, fobj, t)  # 计算适应度值
    fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
    Xs = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitnessS[0])
    print('GbestScore:',GbestScore)
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(Xs[0, :])
    best_train_loss = train_losses[sortIndex[0][0]]
    best_val_loss = val_losses[sortIndex[0][0]]
    best_train_acc = train_accuracies[sortIndex[0][0]]
    best_val_acc = val_accuracies[sortIndex[0][0]]
    best_eval = evals[sortIndex[0][0]]
    Curve = np.zeros([MaxIter+1, 1])
    Curve[t] = GbestScore
    for t in range(MaxIter):

        Flame_no = round(pop - t * ((pop - 1) / MaxIter))
        a = -1 + t * (-1) / MaxIter  # a 线性从-1降到-2
        for i in range(pop):
            for j in range(dim):
                if i <= Flame_no:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * rd.random() + 1

                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * pi) + Xs[i, j]
                else:
                    distance_to_flame = np.abs(Xs[i, j] - X[i, j])
                    b = 1
                    r = (a - 1) * rd.random() + 1
                    X[i, j] = distance_to_flame * np.exp(b * r) * np.cos(r * 2 * pi) + Xs[Flame_no, j]

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness, train_losses, val_losses, evals, train_accuracies, val_accuracies = CaculateFitness(X, fobj, t + 1)  # 计算适应度值
        fitnessS, sortIndex = SortFitness(fitness)  # 对适应度值排序
        Xs = SortPosition(X, sortIndex)  # 种群排序
        if fitnessS[0] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitnessS[0])
            GbestPositon[0, :] = copy.copy(Xs[0, :])
            best_train_loss = train_losses[sortIndex[0][0]]
            best_val_loss = val_losses[sortIndex[0][0]]
            best_eval = evals[sortIndex[0][0]]
            Xs = np.concatenate((GbestPositon, Xs[:-1, :]), axis=0)
        X[-1, :] = copy.copy(GbestPositon)
        print('GbestScore:', GbestScore)
        Curve[t+1] = GbestScore

    return GbestScore, GbestPositon, Curve, best_train_loss, best_val_loss, best_eval, best_train_acc, best_val_acc


if __name__ == '__main__':
    # 导入所需的库和函数
    import numpy as np

    # 定义算法参数
    N = 5  # 种群大小
    Max_iteration = 5  # 最大迭代次数
    lb = [0.0001, 1, 8, 4]  # 变量下界
    ub = [0.001, 30, 20, 16]  # 变量上界
    dim = len(lb)  # 变量维度

    # 调用MFO函数求解优化问题
    best_fitness, best_solution, convergence_curve, best_train_loss, best_val_loss, best_eval, best_train_acc, best_val_acc = MFO(N, dim, lb, ub, Max_iteration,fobj)
    print('打印最优的优化结果——————————————————————————————————————————————')
    print('convergence_curve:', convergence_curve)
    # 打印结果
    print("最优解：", best_solution)
    print("最优适应度：", best_fitness)
    print('best_eval:',best_eval)
    print('accuracy:', best_eval[0])
    print('precision:', best_eval[1])
    print('recall:', best_eval[2])
    print('f1-sore:', best_eval[3])
    x1_lr = best_solution[0][0]
    x2_kernel_size = int(best_solution[0][1])
    x3_gru = int(best_solution[0][2])
    x4_se_ratio = int(best_solution[0][3])
    # 保存超参数到excel
    import pandas as pd

    # 创建一个字典，包含要写入Excel的数据
    data = {'x1_lr': [x1_lr],
            'x2_kernel_size': [x2_kernel_size],
            'x3_stride': [x3_gru],
            'x4_se_ratio': [x3_se_ratio]
            }

    # 将字典转换为DataFrame
    df = pd.DataFrame(data)

    # 指定要保存的Excel文件路径
    exp_id = 'mfo_cbam1'  # 修改本组实验代号
    SAVE_DIR = '/external_data/hxx/SI_python_code/checkpoint' + '/{}'.format(exp_id)
    os.makedirs(SAVE_DIR, exist_ok=True)  # 确保路径存在
    excel_path = os.path.join(SAVE_DIR, 'hyperparameters.xlsx')

    # # 将DataFrame写入Excel文件
    # df.to_excel(excel_path, index=False)
    # Plot the best training and validation loss
    plt.figure()
    plt.plot(best_train_loss, label='Train Loss')
    plt.plot(best_val_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}/Loss.png'.format(SAVE_DIR))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(best_train_acc, label='Train_acc')
    plt.plot(best_val_acc, label="Valid_acc")
    plt.xlabel('Epochs')
    plt.ylabel('Accuarcy')
    plt.grid(True)
    plt.legend()
    plt.savefig('{}/Training and validation Accuarcy.png'.format(SAVE_DIR))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(Max_iteration), convergence_curve,label='convergence_curve')
    plt.xlabel('iter')
    plt.ylabel('fitness')
    plt.legend()
    plt.savefig('{}/fitness_curve.png'.format(SAVE_DIR))


