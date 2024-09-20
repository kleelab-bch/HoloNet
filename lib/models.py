from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Activation, Input, Concatenate, AveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
from lib.Utilities import *

# The Holoblock structure ==============================================================================================
def Holo_Block(img_inputs, x, kernel_size, filter_num, HB_number):
    x = Conv2D(filter_num-64, (3, 3), padding='same', name='Conv_{}'.format(HB_number+1))(x)
    #
    HoloB_x = Conv2D(64, (kernel_size, kernel_size), padding='same', strides=(2**(HB_number-1), 2**(HB_number-1)), name='HoloConv_{}'.format(HB_number))(img_inputs)
    Cat_x = Concatenate(axis=3, name='Cat_{}'.format(HB_number))([x, HoloB_x])
    #
    x = AveragePooling2D((2, 2), strides=2, name='Avg_{}'.format(HB_number))(Cat_x)
    x = BatchNormalization(name='BN_{}'.format(HB_number+1))(x)
    x = Activation('relu', name='RU_{}'.format(HB_number+1))(x)

    return x

# The HoloNet structure ================================================================================================
def HoloNet_model(IMG_SHAPE):
    img_inputs = Input(shape=IMG_SHAPE, name='Input')
    conv_1 = Conv2D(64, (3, 3), padding='same', name='Conv1')(img_inputs)
    BN_1 = BatchNormalization(name='BN_1')(conv_1)
    RU_1 = Activation('relu', name='RU_1')(BN_1)

    RU_2 = Holo_Block(img_inputs, RU_1, 16, 128, 1)
    RU_3 = Holo_Block(img_inputs, RU_2, 24, 256, 2)
    RU_4 = Holo_Block(img_inputs, RU_3, 32, 512, 3)

    conv_5 = Conv2D(512, (3, 3), padding='same', name='Conv_5')(RU_4)
    Avg_4 = AveragePooling2D((8, 8), name='Avg_4')(conv_5)
    BN_5 = BatchNormalization(name='BN_5')(Avg_4)
    RU_5 = Activation('relu', name='RU_5')(BN_5)

    flatten = Flatten(name='Flat')(RU_5)
    dense_1 = Dense(1000, name='Dense_1')(flatten)
    BN_d_1 = BatchNormalization(name='BN_dense_1')(dense_1)
    RU_d_1 = Activation('relu', name='RU_dense_1')(BN_d_1)
    dense_2 = Dense(500, name='Dense_2')(RU_d_1)
    BN_d_2 = BatchNormalization(name='BN_dense_2')(dense_2)
    RU_d_2 = Activation('relu', name='RU_dense_2')(BN_d_2)
    output_c = Dense(4, activation='softmax', name='Classification')(RU_d_2)
    # output_r = Dense(2, activation='relu', name='Regression')(RU_d_2)

    model = Model(inputs=img_inputs, outputs=output_c)

    print(model.summary())
    # plot_model(model, "Dual_HoloNet_model.png")

    return model

# Multi-task HoloNet Model =========================================================================================
def MT_HoloNet_model(IMG_SHAPE):
    img_inputs = Input(shape=IMG_SHAPE)
    conv_1 = Conv2D(64, (3, 3), padding='same')(img_inputs)
    BN_1 = BatchNormalization()(conv_1)
    RU_1 = Activation('relu')(BN_1)

    RU_2 = Holo_Block(img_inputs, RU_1, 16, 128, 1)
    RU_3 = Holo_Block(img_inputs, RU_2, 24, 256, 2)
    RU_4 = Holo_Block(img_inputs, RU_3, 32, 512, 3)

    conv_5 = Conv2D(512, (3, 3), padding='same')(RU_4)
    Avg_5 = AvgPool2D(8)(conv_5)
    BN_5 = BatchNormalization()(Avg_5)
    RU_5 = Activation('relu')(BN_5)

    flatten = Flatten()(RU_5)
    dense_1 = Dense(1000)(flatten)
    BN_d_1 = BatchNormalization()(dense_1)
    RU_d_1 = Activation('relu')(BN_d_1)
    dense_2 = Dense(500)(RU_d_1)
    BN_d_2 = BatchNormalization()(dense_2)
    RU_d_2 = Activation('relu')(BN_d_2)
    output_c = Dense(4, activation='softmax', name='Classification')(RU_d_2)
    output_r = Dense(2, activation='relu', name='Regression')(RU_d_2)

    model = Model(inputs=img_inputs, outputs=[output_c, output_r])

    print(model.summary())

    return model

# MLP structure ========================================================================================================
def MLP(IMG_SHAPE):
    img_inputs = Input(shape=IMG_SHAPE, name='Input')
    dense_1 = Dense(500, name='Dense_1')(img_inputs)
    BN_d_1 = BatchNormalization(name='BN_dense_1')(dense_1)
    RU_d_1 = Activation('relu', name='RU_dense_1')(BN_d_1)
    drop_1 = Dropout(0.2)(RU_d_1)
    dense_2 = Dense(100, name='Dense_2')(drop_1)
    BN_d_2 = BatchNormalization(name='BN_dense_2')(dense_2)
    RU_d_2 = Activation('relu', name='RU_dense_2')(BN_d_2)
    drop_2 = Dropout(0.2)(RU_d_2)
    output_c = Dense(5, activation='softmax', name='Classification')(drop_2)

    model = Model(inputs=img_inputs, outputs=output_c)

    return model

# Training HoloNet model and predict the results if report_sign is True ================================================
def Holo_Model_Training(path, report_sign):
    trainX, y_train, testX, y_test, _, _, _ = data_collection(path)

    # Convert HWCN to NHWC and Normalize
    X_train = np.transpose(trainX, [3, 0, 1, 2])
    X_train = X_train / 65535

    X_test = np.transpose(testX, [3, 0, 1, 2])
    X_test = X_test / 65535

    # One-hot encode target column
    y_train = to_categorical(y_train)
    y_test = np.squeeze(y_test)

    # Callback Setting (if the user want to change the parameters, please check Utilities.py
    lrate = LearningRateScheduler(step_decay)

    Test_Acc = []

    # 5-fold vailation
    kF = KFold(n_splits=5, shuffle=True)

    for train_index, val_index in kF.split(X_train, y_train):

        # Create and train model
        FF_holoNet = HoloNet_model((64, 64, 2))
        # Compile model using accuracy to measure model performance
        FF_holoNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # split training and validation datasets
        X_train_split = X_train[train_index, :, :, :]
        X_val = X_train[val_index, :, :, :]
        y_train_split = y_train[train_index, :]
        y_val = y_train[val_index, :]

        # train the model
        History = FF_holoNet.fit(X_train_split,
                                 y_train_split,
                                 batch_size=128,
                                 epochs=100,
                                 callbacks=[lrate],
                                 validation_data=[X_val, y_val],
                                 verbose=1)

        # summarize history for accuracy
        y_pred_c = FF_holoNet.predict(X_test)
        y_pred = np.argmax(y_pred_c, axis=1)
        test_acc = accuracy_score(y_test, y_pred)
        Test_Acc.append(test_acc)

    if report_sign:
        FF_holoNet.save('Model_Save/HoloNet_Classification.h5')

        print(Test_Acc)
        print('mean :', np.mean(Test_Acc))
        print('std:', np.std(Test_Acc))

# Training Multi-Task HoloNet model and predict the results if report_sign is True =====================================
def MT_Holo_Model_Training(path, report_sign):
    X_train, Y_train, X_test, Y_test, y_train_Int, All_X_Data, cellLine_label = data_collection(path)

    # Convert HWCN to NHWC and Normalize
    X_train = np.transpose(X_train, [3, 0, 1, 2])
    X_train = X_train / 65535

    X_test = np.transpose(X_test, [3, 0, 1, 2])
    X_test = X_test / 65535

    # One-hot encode target column
    y_train = to_categorical(Y_train)
    y_test = np.squeeze(Y_test)

    # Callback Setting
    lrate = LearningRateScheduler(step_decay)

    # Create and train model
    MT_HoloNet_Model = MT_HoloNet_model((64, 64, 2))
    # Compile model using accuracy to measure model performance
    lossWeights = {'Classification': 5, 'Regression': 1}
    MT_HoloNet_Model.compile(optimizer='adam', loss={'Classification': brier_multi, 'Regression': 'mse'},
                       loss_weights=lossWeights, metrics={'Classification': ['accuracy'], 'Regression': ['mse']})

    # train the model
    History = MT_HoloNet_Model.fit(X_train,
                                   [y_train, y_train_Int],
                                   batch_size=128,
                                   epochs=100,
                                   callbacks=[lrate],
                                   validation_split=0.2,
                                   verbose=1)

    if report_sign:
        MT_HoloNet_Model.save('Model_Save/FF_HoloNet_Model.h5')

    return MT_HoloNet_Model, All_X_Data, cellLine_label

def Trans_Holo_Model(path, report_sign):

    MT_HoloNet_Model, X_data, cellLine_label = MT_Holo_Model_Training(path, report_sign)

    X_data = np.transpose(X_data, [3, 0, 1, 2])
    X_data = X_data / 65535

    Features_extract_model = Model(MT_HoloNet_Model.inputs, MT_HoloNet_Model.layers[-7].output)
    Features = Features_extract_model.predict(X_data)

    random_ind = np.arange(Features.shape[0])
    random.shuffle(random_ind)

    X_train_split_h = Features[random_ind[:round(0.8 * Features.shape[0])], :]
    y_train_split_h = cellLine_label[random_ind[:round(0.8 * Features.shape[0])]]
    y_train_split_h = to_categorical(y_train_split_h - 1)

    X_test_split_h = Features[random_ind[round(0.8 * Features.shape[0]):], :]
    y_test_split_h = cellLine_label[random_ind[round(0.8 * Features.shape[0]):]]
    y_test_split_h = y_test_split_h - 1

    Test_Acc = []

    kF = KFold(n_splits=5, shuffle=True)

    for train_index, val_index in kF.split(X_train_split_h, y_train_split_h):
        MLP_MT_holoNet = MLP((1000))
        MLP_MT_holoNet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        X_train_split = X_train_split_h[train_index, :]
        X_val = X_train_split_h[val_index, :]
        y_train_split = y_train_split_h[train_index, :]
        y_val = y_train_split_h[val_index, :]

        History = MLP_MT_holoNet.fit(X_train_split,
                                     y_train_split,
                                     batch_size=128,
                                     epochs=100,
                                     validation_data=[X_val, y_val],
                                     verbose=1)

        y_pred_c = MLP_MT_holoNet.predict(X_test_split_h)
        y_pred = np.argmax(y_pred_c, axis=1)
        test_acc = accuracy_score(y_test_split_h, y_pred)
        Test_Acc.append(test_acc)

    if report_sign:
        MLP_MT_holoNet.save('Model_Save/MLP_FF_HoloNet_Model.h5')

        print(Test_Acc)
        print('mean :', np.max(Test_Acc))
        print('std:', np.std(Test_Acc))

def main(path, **kwargs):
    """
    Input:
        path: the data path
        kwargs:
            Model_Type: String, select the 'HoloNet' model or 'Trans-HoloNet' model, default='HoloNet'
            report_sign: Integer, show the accuracy results and save the model, default=False

    Return: The accuracy results if report_sign is True

    """
    Model_Type = kwargs.get('Model_Type', 'HoloNet')
    report_sign = kwargs.get('report_sign', False)

    if Model_Type == 'HoloNet':
        Holo_Model_Training(path, report_sign)
    elif Model_Type == 'Trans_HoloNet':
        Trans_Holo_Model(path, report_sign)
