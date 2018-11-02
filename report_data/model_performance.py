   # t = MaxPooling3D(pool_size=(2,2,2))(im_input)
    t = Convolution3D(4, (6,6,6),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    #t = Dropout(0.25)(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)
    return model

    0.93 after 10 epochs
    15  0.502   
    30  0.688
    60  0.716

    --------------------------------------------------------

    weights_1_2
   # t = MaxPooling3D(pool_size=(2,2,2))(im_input)
    t = Convolution3D(32, (12,12,12),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.25)(t)

    t = Flatten()(t)
    t = Dense(32)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)
    return model


      model = Model(input=im_input, output=output)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 32)    110624    
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 32)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 32)    0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 24, 32)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 442368)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                14155808  
_________________________________________________________________
activation_2 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 66        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 14,266,498
Trainable params: 14,266,498
Non-trainable params: 0
_________________________________________________________________
None
~                 

500/500 [==============================] - 353s 707ms/step - loss: 0.3042 - acc: 0.8938 - val_loss: 0.2632 - val_acc: 0.9060
Epoch 2/10
50
500/500 [==============================] - 348s 697ms/step - loss: 0.2268 - acc: 0.9270 - val_loss: 0.2622 - val_acc: 0.8940
Epoch 3/10
50
500/500 [==============================] - 349s 697ms/step - loss: 0.2185 - acc: 0.9270 - val_loss: 0.2205 - val_acc: 0.9210
Epoch 4/10
50
500/500 [==============================] - 348s 696ms/step - loss: 0.1898 - acc: 0.9362 - val_loss: 0.2408 - val_acc: 0.9110
Epoch 5/10
50
500/500 [==============================] - 349s 698ms/step - loss: 0.1770 - acc: 0.9406 - val_loss: 0.2055 - val_acc: 0.9250
Epoch 6/10
50
500/500 [==============================] - 348s 697ms/step - loss: 0.1769 - acc: 0.9382 - val_loss: 0.2553 - val_acc: 0.9100
Epoch 7/10
50
500/500 [==============================] - 349s 697ms/step - loss: 0.1697 - acc: 0.9438 - val_loss: 0.2374 - val_acc: 0.9200
Epoch 8/10
50
500/500 [==============================] - 349s 697ms/step - loss: 0.1475 - acc: 0.9506 - val_loss: 0.2282 - val_acc: 0.9290
Epoch 9/10
50
500/500 [==============================] - 348s 696ms/step - loss: 0.1317 - acc: 0.9550 - val_loss: 0.1894 - val_acc: 0.9240
Epoch 10/10
50
500/500 [==============================] - 348s 696ms/step - loss: 0.1496 - acc: 0.9524 - val_loss: 0.2126 - val_acc: 0.9290
Epoch 6/10


500
500
accuracy:0.658    30


--------------------------------------------------

weights_2_1
   # t = MaxPooling3D(pool_size=(2,2,2))(im_input)
    t = Convolution3D(4, (12,12,12),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.25)(t)

    t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)

    t = Flatten()(t)
    
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)

    output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_2 (Conv3D)            (None, 49, 49, 49, 8)     3464      
_________________________________________________________________
activation_2 (Activation)    (None, 49, 49, 49, 8)     0         
_________________________________________________________________
max_pooling3d_2 (MaxPooling3 (None, 24, 24, 24, 8)     0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 110592)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 884744    
_________________________________________________________________
activation_3 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_4 (Activation)    (None, 2)                 0         
=================================================================
Total params: 888,226
Trainable params: 888,226
Non-trainable params: 0
_________________________________________________________________
None
0

500/500 [==============================] - 73s 146ms/step - loss: 0.3639 - acc: 0.8728 - val_loss: 0.3240 - val_acc: 0.8760
500/500 [==============================] - 71s 142ms/step - loss: 0.4239 - acc: 0.8662 - val_loss: 0.5278 - val_acc: 0.8500
500/500 [==============================] - 71s 141ms/step - loss: 0.7315 - acc: 0.8522 - val_loss: 0.4043 - val_acc: 0.8130
500/500 [==============================] - 71s 142ms/step - loss: 0.3815 - acc: 0.8470 - val_loss: 0.3869 - val_acc: 0.8250
500/500 [==============================] - 71s 142ms/step - loss: 0.5305 - acc: 0.8592 - val_loss: 0.8395 - val_acc: 0.8440
500/500 [==============================] - 71s 142ms/step - loss: 0.5218 - acc: 0.8518 - val_loss: 0.3854 - val_acc: 0.8270
500/500 [==============================] - 71s 142ms/step - loss: 0.5182 - acc: 0.8250 - val_loss: 0.4078 - val_acc: 0.8090
500/500 [==============================] - 71s 142ms/step - loss: 0.3761 - acc: 0.8328 - val_loss: 0.3977 - val_acc: 0.8170
500/500 [==============================] - 71s 142ms/step - loss: 0.3696 - acc: 0.8374 - val_loss: 0.3920 - val_acc: 0.8230
500/500 [==============================] - 71s 142ms/step - loss: 0.3941 - acc: 0.8446 - val_loss: 0.4484 - val_acc: 0.8460

500
500
accuracy:0.214


----------------------------------------------------
weights_1_3
with droput

    im_input = Input(shape=input_shape)
    t = Convolution3D(4, (6,6,6),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.25)(t)
    #t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)

Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 4)     1732      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 4)     0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 4)     0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 24, 4)     0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 55296)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 442376    
_________________________________________________________________
activation_2 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 444,126
Trainable params: 444,126
Non-trainable params: 0
_________________________________________________________________

500/500 [==============================] - 71s 143ms/step - loss: 0.3113 - acc: 0.8964 - val_loss: 0.2853 - val_acc: 0.9030
500/500 [==============================] - 70s 139ms/step - loss: 0.2256 - acc: 0.9194 - val_loss: 0.2258 - val_acc: 0.9270
500/500 [==============================] - 67s 135ms/step - loss: 0.2427 - acc: 0.9156 - val_loss: 0.3075 - val_acc: 0.8830
500/500 [==============================] - 69s 138ms/step - loss: 0.2240 - acc: 0.9246 - val_loss: 0.2253 - val_acc: 0.9250
500/500 [==============================] - 69s 137ms/step - loss: 0.2221 - acc: 0.9252 - val_loss: 0.2063 - val_acc: 0.9280
500/500 [==============================] - 69s 138ms/step - loss: 0.2071 - acc: 0.9310 - val_loss: 0.2775 - val_acc: 0.9140
500/500 [==============================] - 71s 141ms/step - loss: 0.2084 - acc: 0.9256 - val_loss: 0.2226 - val_acc: 0.9210
500/500 [==============================] - 69s 138ms/step - loss: 0.1933 - acc: 0.9330 - val_loss: 0.2493 - val_acc: 0.9100
500/500 [==============================] - 68s 136ms/step - loss: 0.1770 - acc: 0.9402 - val_loss: 0.2046 - val_acc: 0.9190
500/500 [==============================] - 69s 137ms/step - loss: 0.2053 - acc: 0.9282 - val_loss: 0.2626 - val_acc: 0.9120

500
500
accuracy:0.662   30

---------------------------------------------------------------------------
weights_1_4

    t = Convolution3D(16, (6,6,6),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    #t = Dropout(0.25)(t)
    #t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 16)    6928      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 16)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 16)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 221184)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1769480   
_________________________________________________________________
activation_2 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 1,776,426
Trainable params: 1,776,426
Non-trainable params: 0
______________________________________________________________
744 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
500/500 [==============================] - 78s 155ms/step - loss: 0.2777 - acc: 0.9086 - val_loss: 0.2474 - val_acc: 0.9200
Epoch 2/10
500/500 [==============================] - 75s 151ms/step - loss: 0.2082 - acc: 0.9294 - val_loss: 0.2160 - val_acc: 0.9360
Epoch 3/10
500/500 [==============================] - 75s 150ms/step - loss: 0.1969 - acc: 0.9398 - val_loss: 0.2043 - val_acc: 0.9290
Epoch 4/10
500/500 [==============================] - 75s 151ms/step - loss: 0.1634 - acc: 0.9480 - val_loss: 0.2179 - val_acc: 0.9210
Epoch 5/10
500/500 [==============================] - 76s 151ms/step - loss: 0.1550 - acc: 0.9506 - val_loss: 0.2027 - val_acc: 0.9320
Epoch 6/10
500/500 [==============================] - 75s 151ms/step - loss: 0.1483 - acc: 0.9514 - val_loss: 0.2057 - val_acc: 0.9220
Epoch 7/10
500/500 [==============================] - 75s 151ms/step - loss: 0.1342 - acc: 0.9570 - val_loss: 0.1938 - val_acc: 0.9270
Epoch 8/10
500/500 [==============================] - 75s 150ms/step - loss: 0.1048 - acc: 0.9682 - val_loss: 0.1939 - val_acc: 0.9320
Epoch 9/10
500/500 [==============================] - 75s 151ms/step - loss: 0.1006 - acc: 0.9698 - val_loss: 0.1961 - val_acc: 0.9240
Epoch 10/10
500/500 [==============================] - 75s 151ms/step - loss: 0.1077 - acc: 0.9668 - val_loss: 0.1996 - val_acc: 0.9410
Predicting -----------------------------

500
500
accuracy:0.720   30

---------------------------------------------------------------------------------------------

weights_1_5

    t = Convolution3D(16, (4,4,4),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    #t = Dropout(0.25)(t)
    #t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)

    model = Model(input=im_input, output=output)

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 16)    2064      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 16)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 16)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 221184)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 1769480   
_________________________________________________________________
activation_2 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 1,771,562
Trainable params: 1,771,562
Non-trainable params: 0
_________________________________________________________________
500/500 [==============================] - 66s 131ms/step - loss: 0.2800 - acc: 0.9026 - val_loss: 0.2918 - val_acc: 0.8780
Epoch 2/10
500/500 [==============================] - 65s 129ms/step - loss: 0.1918 - acc: 0.9356 - val_loss: 0.2154 - val_acc: 0.9260
Epoch 3/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1697 - acc: 0.9426 - val_loss: 0.1766 - val_acc: 0.9410
Epoch 4/10
500/500 [==============================] - 64s 128ms/step - loss: 0.1391 - acc: 0.9540 - val_loss: 0.1658 - val_acc: 0.9450
Epoch 5/10
500/500 [==============================] - 64s 127ms/step - loss: 0.1233 - acc: 0.9592 - val_loss: 0.1781 - val_acc: 0.9440
Epoch 6/10
500/500 [==============================] - 64s 127ms/step - loss: 0.1173 - acc: 0.9618 - val_loss: 0.1766 - val_acc: 0.9360
Epoch 7/10
500/500 [==============================] - 65s 130ms/step - loss: 0.1103 - acc: 0.9652 - val_loss: 0.1668 - val_acc: 0.9360
Epoch 8/10
500/500 [==============================] - 63s 125ms/step - loss: 0.0800 - acc: 0.9766 - val_loss: 0.1571 - val_acc: 0.9500
Epoch 9/10
500/500 [==============================] - 63s 125ms/step - loss: 0.0741 - acc: 0.9748 - val_loss: 0.1528 - val_acc: 0.9440
Epoch 10/10
500/500 [==============================] - 63s 125ms/step - loss: 0.0835 - acc: 0.9734 - val_loss: 0.1460 - val_acc: 0.9510

500
500
accuracy:0.800


-----------------------------------------------------------------------------
weights_1_6

    im_input = Input(shape=input_shape)
    t = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    #t = Dropout(0.25)(t)
    #t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(16)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

    Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 32)    4128      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 32)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 32)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 442368)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 16)                7077904   
_________________________________________________________________
activation_2 (Activation)    (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 34        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 7,082,066
Trainable params: 7,082,066
Non-trainable params: 0
_________________________________________________________________
744 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
500/500 [==============================] - 64s 129ms/step - loss: 0.2590 - acc: 0.9098 - val_loss: 0.2434 - val_acc: 0.9080
Epoch 2/10
500/500 [==============================] - 64s 129ms/step - loss: 0.1965 - acc: 0.9296 - val_loss: 0.2127 - val_acc: 0.9360
Epoch 3/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1797 - acc: 0.9392 - val_loss: 0.1785 - val_acc: 0.9360
Epoch 4/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1423 - acc: 0.9528 - val_loss: 0.1666 - val_acc: 0.9440
Epoch 5/10
500/500 [==============================] - 63s 125ms/step - loss: 0.1208 - acc: 0.9582 - val_loss: 0.1799 - val_acc: 0.9490
Epoch 6/10
500/500 [==============================] - 63s 125ms/step - loss: 0.1122 - acc: 0.9628 - val_loss: 0.1515 - val_acc: 0.9500
Epoch 7/10
500/500 [==============================] - 64s 128ms/step - loss: 0.1021 - acc: 0.9680 - val_loss: 0.1504 - val_acc: 0.9450
Epoch 8/10
500/500 [==============================] - 63s 125ms/step - loss: 0.0823 - acc: 0.9702 - val_loss: 0.1756 - val_acc: 0.9470
Epoch 9/10
500/500 [==============================] - 63s 127ms/step - loss: 0.0725 - acc: 0.9742 - val_loss: 0.1412 - val_acc: 0.9500
Epoch 10/10
500/500 [==============================] - 63s 127ms/step - loss: 0.0763 - acc: 0.9772 - val_loss: 0.1488 - val_acc: 0.9560
500
500
accuracy:0.814

---------------------------------------------------------------------

    im_input = Input(shape=input_shape)
    t = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    #t = Dropout(0.25)(t)
    #t = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

    500/500 [==============================] - 65s 130ms/step - loss: 0.3133 - acc: 0.8888 - val_loss: 0.3542 - val_acc: 0.8990
Epoch 2/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1931 - acc: 0.9354 - val_loss: 0.1906 - val_acc: 0.9400
Epoch 3/10
500/500 [==============================] - 62s 124ms/step - loss: 0.1793 - acc: 0.9386 - val_loss: 0.2284 - val_acc: 0.9180
Epoch 4/10
500/500 [==============================] - 62s 123ms/step - loss: 0.1399 - acc: 0.9518 - val_loss: 0.1759 - val_acc: 0.9430
Epoch 5/10
500/500 [==============================] - 62s 124ms/step - loss: 0.1164 - acc: 0.9622 - val_loss: 0.2005 - val_acc: 0.9420
Epoch 6/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1128 - acc: 0.9658 - val_loss: 0.1820 - val_acc: 0.9270
Epoch 7/10
500/500 [==============================] - 64s 128ms/step - loss: 0.1027 - acc: 0.9672 - val_loss: 0.1584 - val_acc: 0.9430
Epoch 8/10
500/500 [==============================] - 63s 126ms/step - loss: 0.0784 - acc: 0.9742 - val_loss: 0.1682 - val_acc: 0.9550
Epoch 9/10
500/500 [==============================] - 62s 124ms/step - loss: 0.0729 - acc: 0.9778 - val_loss: 0.1656 - val_acc: 0.9480
Epoch 10/10
500/500 [==============================] - 63s 126ms/step - loss: 0.0685 - acc: 0.9772 - val_loss: 0.1739 - val_acc: 0.9550

500
500
accuracy:0.820

----------------------------------------------------------
weight_2_2
    im_input = Input(shape=input_shape)
    t = Convolution3D(4, (6,6,6),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.5)(t)
    t = Convolution3D(8, (4,4,4),padding='same')(t) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

    500/500 [==============================] - 76s 152ms/step - loss: 0.3396 - acc: 0.8764 - val_loss: 0.4130 - val_acc: 0.8060
Epoch 2/10
500/500 [==============================] - 74s 148ms/step - loss: 0.2510 - acc: 0.9126 - val_loss: 0.2915 - val_acc: 0.8860
Epoch 3/10
500/500 [==============================] - 74s 148ms/step - loss: 0.2554 - acc: 0.9128 - val_loss: 0.9395 - val_acc: 0.5590
Epoch 4/10
500/500 [==============================] - 74s 147ms/step - loss: 0.2249 - acc: 0.9246 - val_loss: 0.6064 - val_acc: 0.7240
Epoch 5/10


----------------------------------------------------

weight_2_2

    im_input = Input(shape=input_shape)
    t = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.5)(t)
    t = Convolution3D(8, (4,4,4),padding='same')(t) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

500/500 [==============================] - 71s 143ms/step - loss: 0.3266 - acc: 0.8932 - val_loss: 0.3270 - val_acc: 0.8640
Epoch 2/10
500/500 [==============================] - 69s 139ms/step - loss: 0.2342 - acc: 0.9232 - val_loss: 0.2128 - val_acc: 0.9320
Epoch 3/10
500/500 [==============================] - 69s 138ms/step - loss: 0.2286 - acc: 0.9250 - val_loss: 0.3810 - val_acc: 0.7980
Epoch 4/10
500/500 [==============================] - 69s 138ms/step - loss: 0.2041 - acc: 0.9334 - val_loss: 0.2165 - val_acc: 0.9270
Epoch 5/10
500/500 [==============================] - 69s 138ms/step - loss: 0.1952 - acc: 0.9368 - val_loss: 0.2234 - val_acc: 0.9130
Epoch 6/10
500/500 [==============================] - 69s 139ms/step - loss: 0.1942 - acc: 0.9378 - val_loss: 0.2194 - val_acc: 0.9290
Epoch 7/10
500/500 [==============================] - 69s 139ms/step - loss: 0.1817 - acc: 0.9408 - val_loss: 0.2291 - val_acc: 0.9290
Epoch 8/10
500/500 [==============================] - 69s 138ms/step - loss: 0.1713 - acc: 0.9448 - val_loss: 0.2579 - val_acc: 0.9120
Epoch 9/10
500/500 [==============================] - 70s 139ms/step - loss: 0.1618 - acc: 0.9466 - val_loss: 0.2312 - val_acc: 0.9340
Epoch 10/10
500/500 [==============================] - 69s 138ms/step - loss: 0.1785 - acc: 0.9394 - val_loss: 0.2057 - val_acc: 0.9310
Predicting -----------------------------

500
500
accuracy:0.586

---------------------------------------------------------------
increased number of channels to 64

weigths_1_7

    im_input = Input(shape=input_shape)
    t = Convolution3D(64, (4,4,4),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.5)(t)
    #t = Convolution3D(8, (4,4,4),padding='same')(t) #(24,24,24)
    #t = Activation('relu')(t)
    #t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)
    return model
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 64)    8256      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 64)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 64)    0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 24, 64)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 884736)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 7077896   
_________________________________________________________________
activation_2 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 7,086,170
Trainable params: 7,086,170
Non-trainable params: 0
_________________________________________________________________

    500/500 [==============================] - 88s 176ms/step - loss: 0.2473 - acc: 0.9144 - val_loss: 0.2545 - val_acc: 0.9080
Epoch 2/10
500/500 [==============================] - 86s 171ms/step - loss: 0.2129 - acc: 0.9262 - val_loss: 0.3130 - val_acc: 0.9010
Epoch 3/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1988 - acc: 0.9322 - val_loss: 0.1966 - val_acc: 0.9410
Epoch 4/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1702 - acc: 0.9458 - val_loss: 0.1904 - val_acc: 0.9370
Epoch 5/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1480 - acc: 0.9496 - val_loss: 0.1935 - val_acc: 0.9360
Epoch 6/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1490 - acc: 0.9470 - val_loss: 0.1806 - val_acc: 0.9320
Epoch 7/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1394 - acc: 0.9548 - val_loss: 0.1666 - val_acc: 0.9360
Epoch 8/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1126 - acc: 0.9628 - val_loss: 0.1634 - val_acc: 0.9480
Epoch 9/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1070 - acc: 0.9642 - val_loss: 0.1546 - val_acc: 0.9370
Epoch 10/10
500/500 [==============================] - 86s 172ms/step - loss: 0.1175 - acc: 0.9622 - val_loss: 0.1663 - val_acc: 0.9470

500
500
accuracy:0.800

------------------------------------------------------------------------------
two conv layers parallel
weight_2p_1
    im_input = Input(shape=input_shape)
    t1 = Convolution3D(16, (4,4,4),padding='same')(im_input) #(24,24,24)
    t1 = Activation('relu')(t1)
    t1 = MaxPooling3D(pool_size=(2,2,2))(t1)
    t1 = Dropout(0.5)(t1)
    t2 = Convolution3D(8, (6,6,6),padding='same')(im_input) #(24,24,24)
    t2 = Activation('relu')(t2) 
    t2 = MaxPooling3D(pool_size=(2,2,2))(t2)
    t1 = Flatten()(t1)
    t2 = Flatten()(t2)
    t1 = Dense(8)(t1)
    t2 = Dense(8)(t2)
    t = concatenate([t1,t2])
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

500/500 [==============================] - 107s 214ms/step - loss: 0.2813 - acc: 0.9074 - val_loss: 0.2489 - val_acc: 0.9120
Epoch 2/10
500/500 [==============================] - 104s 209ms/step - loss: 0.2127 - acc: 0.9316 - val_loss: 0.2148 - val_acc: 0.9340
Epoch 3/10
500/500 [==============================] - 104s 209ms/step - loss: 0.2015 - acc: 0.9316 - val_loss: 0.2605 - val_acc: 0.8990
Epoch 4/10
500/500 [==============================] - 105s 209ms/step - loss: 0.1778 - acc: 0.9404 - val_loss: 0.2002 - val_acc: 0.9340
Epoch 5/10
500/500 [==============================] - 105s 209ms/step - loss: 0.1547 - acc: 0.9474 - val_loss: 0.1871 - val_acc: 0.9370
Epoch 6/10
500/500 [==============================] - 104s 208ms/step - loss: 0.1608 - acc: 0.9486 - val_loss: 0.1958 - val_acc: 0.9390
Epoch 7/10
500/500 [==============================] - 105s 209ms/step - loss: 0.1476 - acc: 0.9530 - val_loss: 0.2028 - val_acc: 0.9240
Epoch 8/10
500/500 [==============================] - 104s 209ms/step - loss: 0.1301 - acc: 0.9576 - val_loss: 0.1743 - val_acc: 0.9400
Epoch 9/10
500/500 [==============================] - 104s 209ms/step - loss: 0.1268 - acc: 0.9596 - val_loss: 0.1785 - val_acc: 0.9360
Epoch 10/10
500/500 [==============================] - 104s 209ms/step - loss: 0.1368 - acc: 0.9540 - val_loss: 0.1823 - val_acc: 0.9450


    500
500
accuracy:0.738

-------------------------------------------------------------------------
weights_1_8


    im_input = Input(shape=input_shape)
    t = Convolution3D(32, (2,2,2),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.5)(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)
500/500 [==============================] - 66s 131ms/step - loss: 0.2521 - acc: 0.9146 - val_loss: 0.2311 - val_acc: 0.9120
Epoch 2/10
500/500 [==============================] - 63s 127ms/step - loss: 0.1969 - acc: 0.9316 - val_loss: 0.1927 - val_acc: 0.9410
Epoch 3/10
500/500 [==============================] - 63s 125ms/step - loss: 0.1794 - acc: 0.9400 - val_loss: 0.1800 - val_acc: 0.9390
Epoch 4/10
500/500 [==============================] - 63s 126ms/step - loss: 0.1641 - acc: 0.9466 - val_loss: 0.1555 - val_acc: 0.9480
Epoch 5/10
500/500 [==============================] - 62s 124ms/step - loss: 0.1322 - acc: 0.9530 - val_loss: 0.1848 - val_acc: 0.9450
Epoch 6/10
500/500 [==============================] - 62s 124ms/step - loss: 0.1235 - acc: 0.9574 - val_loss: 0.1359 - val_acc: 0.9540
Epoch 7/10
500/500 [==============================] - 63s 127ms/step - loss: 0.1209 - acc: 0.9590 - val_loss: 0.1521 - val_acc: 0.9490
Epoch 8/10
500/500 [==============================] - 62s 125ms/step - loss: 0.1002 - acc: 0.9664 - val_loss: 0.1577 - val_acc: 0.9520
Epoch 9/10
500/500 [==============================] - 62s 124ms/step - loss: 0.0929 - acc: 0.9684 - val_loss: 0.1386 - val_acc: 0.9550
Epoch 10/10
500/500 [==============================] - 62s 125ms/step - loss: 0.1067 - acc: 0.9658 - val_loss: 0.1341 - val_acc: 0.9490

500
500
accuracy:0.828

--------------------------------
weights_1_9
    im_input = Input(shape=input_shape)
    t = Convolution3D(32, (3,3,3),padding='same')(im_input) #(24,24,24)
    t = Activation('relu')(t)
    t = MaxPooling3D(pool_size=(2,2,2))(t)
    t = Dropout(0.5)(t)
    t = Flatten()(t)
    t = Dense(8)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 49, 49, 49, 2)     0         
_________________________________________________________________
conv3d_1 (Conv3D)            (None, 49, 49, 49, 32)    1760      
_________________________________________________________________
activation_1 (Activation)    (None, 49, 49, 49, 32)    0         
_________________________________________________________________
max_pooling3d_1 (MaxPooling3 (None, 24, 24, 24, 32)    0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 24, 32)    0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 442368)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 3538952   
_________________________________________________________________
activation_2 (Activation)    (None, 8)                 0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 18        
_________________________________________________________________
activation_3 (Activation)    (None, 2)                 0         
=================================================================
Total params: 3,540,730
Trainable params: 3,540,730
Non-trainable params: 0
_________________________________________________________________

500/500 [==============================] - 64s 129ms/step - loss: 0.2907 - acc: 0.8982 - val_loss: 0.2294 - val_acc: 0.9250
Epoch 2/10
500/500 [==============================] - 63s 127ms/step - loss: 0.3471 - acc: 0.8772 - val_loss: 0.4360 - val_acc: 0.8040
Epoch 3/10
500/500 [==============================] - 63s 125ms/step - loss: 0.3343 - acc: 0.8772 - val_loss: 0.3220 - val_acc: 0.8790
Epoch 4/10
500/500 [==============================] - 63s 126ms/step - loss: 0.2707 - acc: 0.9012 - val_loss: 0.5086 - val_acc: 0.7500
Epoch 5/10
500/500 [==============================] - 62s 125ms/step - loss: 0.3735 - acc: 0.8672 - val_loss: 0.3187 - val_acc: 0.8730
Epoch 6/10
500/500 [==============================] - 63s 127ms/step - loss: 0.5836 - acc: 0.8528 - val_loss: 0.6404 - val_acc: 0.8490
Epoch 7/10
500/500 [==============================] - 65s 130ms/step - loss: 0.6012 - acc: 0.8530 - val_loss: 0.6189 - val_acc: 0.8460
Epoch 8/10
500/500 [==============================] - 63s 126ms/step - loss: 0.5656 - acc: 0.8502 - val_loss: 0.4417 - val_acc: 0.8030
Epoch 9/10
500/500 [==============================] - 63s 126ms/step - loss: 0.3960 - acc: 0.8472 - val_loss: 0.3954 - val_acc: 0.8540
Epoch 10/10
500/500 [==============================] - 63s 126ms/step - loss: 0.3782 - acc: 0.8526 - val_loss: 0.4745 - val_acc: 0.8540

500
500
accuracy:0.256

------------------------------------------------
weight_2p_2
    
    im_input = Input(shape=input_shape)
    t1 = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t1 = Activation('relu')(t1)
    t1 = MaxPooling3D(pool_size=(2,2,2))(t1)
    t1 = Dropout(0.5)(t1)
    t2 = Convolution3D(32, (2,2,2),padding='same')(im_input) #(24,24,24)
    t2 = Activation('relu')(t2)
    t2 = MaxPooling3D(pool_size=(2,2,2))(t2)
    t1 = Flatten()(t1)
    t2 = Flatten()(t2)
    t1 = Dense(8)(t1)
    t2 = Dense(8)(t2)
    t = concatenate([t1,t2])
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

  model = Model(input=im_input, output=output)
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 49, 49, 49, 2 0                                            
__________________________________________________________________________________________________
conv3d_1 (Conv3D)               (None, 49, 49, 49, 3 4128        input_1[0][0]                    
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 49, 49, 49, 3 0           conv3d_1[0][0]                   
__________________________________________________________________________________________________
conv3d_2 (Conv3D)               (None, 49, 49, 49, 3 544         input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling3d_1 (MaxPooling3D)  (None, 24, 24, 24, 3 0           activation_1[0][0]               
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 49, 49, 49, 3 0           conv3d_2[0][0]                   
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 24, 24, 24, 3 0           max_pooling3d_1[0][0]            
__________________________________________________________________________________________________
max_pooling3d_2 (MaxPooling3D)  (None, 24, 24, 24, 3 0           activation_2[0][0]               
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 442368)       0           dropout_1[0][0]                  
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 442368)       0           max_pooling3d_2[0][0]            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 8)            3538952     flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 8)            3538952     flatten_2[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 16)           0           dense_1[0][0]                    
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 16)           0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 2)            34          activation_3[0][0]               
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 2)            0           dense_3[0][0]                    


500/500 [==============================] - 80s 159ms/step - loss: 0.2505 - acc: 0.9156 - val_loss: 0.2294 - val_acc: 0.9120
Epoch 2/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1899 - acc: 0.9386 - val_loss: 0.2012 - val_acc: 0.9310
Epoch 3/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1743 - acc: 0.9432 - val_loss: 0.1962 - val_acc: 0.9350
Epoch 4/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1512 - acc: 0.9490 - val_loss: 0.1788 - val_acc: 0.9360
Epoch 5/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1150 - acc: 0.9626 - val_loss: 0.1621 - val_acc: 0.9440
Epoch 6/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1049 - acc: 0.9642 - val_loss: 0.1375 - val_acc: 0.9500
Epoch 7/10
500/500 [==============================] - 77s 155ms/step - loss: 0.0967 - acc: 0.9696 - val_loss: 0.1524 - val_acc: 0.9400
Epoch 8/10
500/500 [==============================] - 77s 155ms/step - loss: 0.0692 - acc: 0.9792 - val_loss: 0.1441 - val_acc: 0.9470
Epoch 9/10
500/500 [==============================] - 77s 155ms/step - loss: 0.0542 - acc: 0.9844 - val_loss: 0.1580 - val_acc: 0.9390
Epoch 10/10
500/500 [==============================] - 77s 155ms/step - loss: 0.0653 - acc: 0.9792 - val_loss: 0.1371 - val_acc: 0.9520

500
500
accuracy:0.838

--------------------------------------------------------------------------------------------
weight_2p_3
 same network as in weights_2p_2

 h = 1 p = 2 changed to h=1 p=10 

                 if atomtype_list_ == 'h':
                        atomtype_list.append(1) # ('h') # 'h' means hydrophobic TODO take this into account in a betterway
                else:
                        atomtype_list.append(10)

500/500 [==============================] - 79s 159ms/step - loss: 0.2696 - acc: 0.9032 - val_loss: 0.2241 - val_acc: 0.9140
Epoch 2/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1969 - acc: 0.9292 - val_loss: 0.2260 - val_acc: 0.9180
Epoch 3/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1922 - acc: 0.9330 - val_loss: 0.3011 - val_acc: 0.8970
Epoch 4/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1589 - acc: 0.9468 - val_loss: 0.2187 - val_acc: 0.9090
Epoch 5/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1562 - acc: 0.9474 - val_loss: 0.2220 - val_acc: 0.9140
Epoch 6/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1434 - acc: 0.9526 - val_loss: 0.2130 - val_acc: 0.9090
Epoch 7/10
500/500 [==============================] - 77s 155ms/step - loss: 0.1315 - acc: 0.9564 - val_loss: 0.2524 - val_acc: 0.9070
Epoch 8/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1030 - acc: 0.9688 - val_loss: 0.2382 - val_acc: 0.9090
Epoch 9/10
500/500 [==============================] - 77s 154ms/step - loss: 0.0960 - acc: 0.9692 - val_loss: 0.2920 - val_acc: 0.8940
Epoch 10/10
500/500 [==============================] - 77s 154ms/step - loss: 0.1085 - acc: 0.9638 - val_loss: 0.2816 - val_acc: 0.9110



---------------------

weight_2p_4


changed back to h = 1 p = 2

 im_input = Input(shape=input_shape)
    t1 = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t1 = Activation('relu')(t1)
    t1 = MaxPooling3D(pool_size=(2,2,2))(t1)
    t1 = Dropout(0.5)(t1)
    t2 = Convolution3D(32, (2,2,2),padding='same')(im_input) #(24,24,24)
    t2 = Activation('relu')(t2)
    t2 = MaxPooling3D(pool_size=(2,2,2))(t2)
    t2 = Dropout(0.5)(t2)
    t1 = Flatten()(t1)
    t2 = Flatten()(t2)
    t1 = Dense(8)(t1)
    t2 = Dense(8)(t2)
    t = concatenate([t1,t2])
    t = Dense(4)(t)
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)


744 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
500/500 [==============================] - 83s 165ms/step - loss: 0.2708 - acc: 0.9176 - val_loss: 0.2190 - val_acc: 0.9220
Epoch 2/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1806 - acc: 0.9416 - val_loss: 0.1929 - val_acc: 0.9300
Epoch 3/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1602 - acc: 0.9454 - val_loss: 0.1709 - val_acc: 0.9430
Epoch 4/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1365 - acc: 0.9550 - val_loss: 0.1852 - val_acc: 0.9370
Epoch 5/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1300 - acc: 0.9576 - val_loss: 0.1605 - val_acc: 0.9430
Epoch 6/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1120 - acc: 0.9620 - val_loss: 0.1503 - val_acc: 0.9370
Epoch 7/10
500/500 [==============================] - 80s 161ms/step - loss: 0.1093 - acc: 0.9632 - val_loss: 0.1515 - val_acc: 0.9460
Epoch 8/10
500/500 [==============================] - 80s 160ms/step - loss: 0.0878 - acc: 0.9722 - val_loss: 0.1490 - val_acc: 0.9480
Epoch 9/10
500/500 [==============================] - 80s 160ms/step - loss: 0.0795 - acc: 0.9770 - val_loss: 0.1619 - val_acc: 0.9430
Epoch 10/10
500/500 [==============================] - 80s 160ms/step - loss: 0.0914 - acc: 0.9694 - val_loss: 0.1686 - val_acc: 0.9510

500
500
accuracy:0.820

------------------------------------------------
going back to 2p_4 network and add dropout to 2nd parallel network as well
weight_2p_5
    
    im_input = Input(shape=input_shape)
    t1 = Convolution3D(32, (4,4,4),padding='same')(im_input) #(24,24,24)
    t1 = Activation('relu')(t1)
    t1 = MaxPooling3D(pool_size=(2,2,2))(t1)
    t1 = Dropout(0.5)(t1)

    t2 = Convolution3D(32, (2,2,2),padding='same')(im_input) #(24,24,24)
    t2 = Activation('relu')(t2)
    t2 = MaxPooling3D(pool_size=(2,2,2))(t2)
    t2 = Dropout(0.5)(t2)

    t1 = Flatten()(t1)
    t2 = Flatten()(t2)
    t1 = Dense(8)(t1)
    t2 = Dense(8)(t2)
    t = concatenate([t1,t2])
    t = Activation('relu')(t)
    t = Dense(class_num)(t)
    output = Activation('softmax')(t)
    model = Model(input=im_input, output=output)

500/500 [==============================] - 82s 163ms/step - loss: 0.2407 - acc: 0.9198 - val_loss: 0.2219 - val_acc: 0.9300
Epoch 2/10
500/500 [==============================] - 80s 159ms/step - loss: 0.1757 - acc: 0.9378 - val_loss: 0.2075 - val_acc: 0.9300
Epoch 3/10
500/500 [==============================] - 79s 159ms/step - loss: 0.1635 - acc: 0.9446 - val_loss: 0.1595 - val_acc: 0.9390
Epoch 4/10
500/500 [==============================] - 80s 159ms/step - loss: 0.1392 - acc: 0.9514 - val_loss: 0.1558 - val_acc: 0.9430
Epoch 5/10
500/500 [==============================] - 79s 159ms/step - loss: 0.1189 - acc: 0.9584 - val_loss: 0.1520 - val_acc: 0.9470
Epoch 6/10
500/500 [==============================] - 79s 159ms/step - loss: 0.1259 - acc: 0.9568 - val_loss: 0.1429 - val_acc: 0.9490
Epoch 7/10
500/500 [==============================] - 80s 160ms/step - loss: 0.1202 - acc: 0.9598 - val_loss: 0.1319 - val_acc: 0.9500
Epoch 8/10
500/500 [==============================] - 80s 159ms/step - loss: 0.1007 - acc: 0.9652 - val_loss: 0.1565 - val_acc: 0.9500
Epoch 9/10
500/500 [==============================] - 80s 159ms/step - loss: 0.0836 - acc: 0.9730 - val_loss: 0.1547 - val_acc: 0.9480
Epoch 10/10
500/500 [==============================] - 80s 159ms/step - loss: 0.0916 - acc: 0.9686 - val_loss: 0.1700 - val_acc: 0.9460


    500
500
accuracy:0.814

--------------------------------------------------

going back to 2p_4 network with adagrad optimizer

500/500 [==============================] - 81s 162ms/step - loss: 8.0324 - acc: 0.5004 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 2/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 3/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 4/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 5/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 6/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 7/10
500/500 [==============================] - 79s 158ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 8/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 9/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000
Epoch 10/10
500/500 [==============================] - 79s 157ms/step - loss: 8.0590 - acc: 0.5000 - val_loss: 8.0590 - val_acc: 0.5000


--------------------------------------------------------
weight_2p_6 same weight_2p_2

500/500 [==============================] - 81s 162ms/step - loss: 0.2541 - acc: 0.9120 - val_loss: 0.2431 - val_acc: 0.9160
Epoch 2/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1892 - acc: 0.9354 - val_loss: 0.1901 - val_acc: 0.9380
Epoch 3/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1831 - acc: 0.9374 - val_loss: 0.1847 - val_acc: 0.9340
Epoch 4/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1595 - acc: 0.9504 - val_loss: 0.1797 - val_acc: 0.9420
Epoch 5/10
500/500 [==============================] - 78s 157ms/step - loss: 0.1426 - acc: 0.9532 - val_loss: 0.1868 - val_acc: 0.9360
Epoch 6/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1470 - acc: 0.9508 - val_loss: 0.1666 - val_acc: 0.9440
Epoch 7/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1358 - acc: 0.9558 - val_loss: 0.1769 - val_acc: 0.9350
Epoch 8/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1175 - acc: 0.9624 - val_loss: 0.1771 - val_acc: 0.9440
Epoch 9/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1153 - acc: 0.9604 - val_loss: 0.1729 - val_acc: 0.9380
Epoch 10/10
500/500 [==============================] - 79s 157ms/step - loss: 0.1243 - acc: 0.9612 - val_loss: 0.1913 - val_acc: 0.9400

------------------

500/500 [==============================] - 81s 163ms/step - loss: 0.2372 - acc: 0.9216 - val_loss: 0.2026 - val_acc: 0.9310
Epoch 2/20
500/500 [==============================] - 79s 159ms/step - loss: 0.1567 - acc: 0.9466 - val_loss: 0.1676 - val_acc: 0.9410
Epoch 3/20
500/500 [==============================] - 79s 159ms/step - loss: 0.1228 - acc: 0.9568 - val_loss: 0.1359 - val_acc: 0.9510
Epoch 4/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0949 - acc: 0.9706 - val_loss: 0.1503 - val_acc: 0.9420
Epoch 5/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0785 - acc: 0.9752 - val_loss: 0.1602 - val_acc: 0.9530
Epoch 6/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0756 - acc: 0.9774 - val_loss: 0.1332 - val_acc: 0.9580
Epoch 7/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0746 - acc: 0.9758 - val_loss: 0.1525 - val_acc: 0.9450
Epoch 8/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0543 - acc: 0.9834 - val_loss: 0.1765 - val_acc: 0.9490
Epoch 9/20
500/500 [==============================] - 79s 157ms/step - loss: 0.0434 - acc: 0.9860 - val_loss: 0.2865 - val_acc: 0.9250
Epoch 10/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0564 - acc: 0.9842 - val_loss: 0.1445 - val_acc: 0.9560
Epoch 11/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0517 - acc: 0.9874 - val_loss: 0.1643 - val_acc: 0.9490
Epoch 12/20
500/500 [==============================] - 79s 157ms/step - loss: 0.0461 - acc: 0.9882 - val_loss: 0.2131 - val_acc: 0.9440
Epoch 13/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0378 - acc: 0.9904 - val_loss: 0.1898 - val_acc: 0.9490
Epoch 14/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0353 - acc: 0.9908 - val_loss: 0.2006 - val_acc: 0.9510
Epoch 15/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0401 - acc: 0.9912 - val_loss: 0.2636 - val_acc: 0.9340
Epoch 16/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0386 - acc: 0.9918 - val_loss: 0.2156 - val_acc: 0.9320
Epoch 17/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0471 - acc: 0.9886 - val_loss: 0.2447 - val_acc: 0.9390
Epoch 18/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0320 - acc: 0.9918 - val_loss: 0.3457 - val_acc: 0.9180
Epoch 19/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0357 - acc: 0.9910 - val_loss: 0.3146 - val_acc: 0.9230
Epoch 20/20

500
500
accuracy:0.828

-----------------------------------

20 epochs with prune parameter 60

Epoch 2/20
500/500 [==============================] - 79s 159ms/step - loss: 0.1749 - acc: 0.9380 - val_loss: 0.1874 - val_acc: 0.9350
Epoch 3/20
500/500 [==============================] - 80s 159ms/step - loss: 0.1414 - acc: 0.9538 - val_loss: 0.1418 - val_acc: 0.9480
Epoch 4/20
500/500 [==============================] - 79s 159ms/step - loss: 0.1073 - acc: 0.9670 - val_loss: 0.1566 - val_acc: 0.9480
Epoch 5/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0891 - acc: 0.9710 - val_loss: 0.1841 - val_acc: 0.9330
Epoch 6/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0796 - acc: 0.9758 - val_loss: 0.1453 - val_acc: 0.9460
Epoch 7/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0831 - acc: 0.9766 - val_loss: 0.1745 - val_acc: 0.9290
Epoch 8/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0517 - acc: 0.9854 - val_loss: 0.1588 - val_acc: 0.9400
Epoch 9/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0492 - acc: 0.9852 - val_loss: 0.1997 - val_acc: 0.9270
Epoch 10/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0568 - acc: 0.9836 - val_loss: 0.1476 - val_acc: 0.9560
Epoch 11/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0594 - acc: 0.9844 - val_loss: 0.1814 - val_acc: 0.9380
Epoch 12/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0520 - acc: 0.9866 - val_loss: 0.1744 - val_acc: 0.9380
Epoch 13/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0437 - acc: 0.9874 - val_loss: 0.2179 - val_acc: 0.9380
Epoch 14/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0379 - acc: 0.9912 - val_loss: 0.1555 - val_acc: 0.9510
Epoch 15/20
500/500 [==============================] - 79s 158ms/step - loss: 0.0429 - acc: 0.9896 - val_loss: 0.1732 - val_acc: 0.9440
Epoch 16/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0436 - acc: 0.9908 - val_loss: 0.2165 - val_acc: 0.9350
Epoch 17/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0453 - acc: 0.9898 - val_loss: 0.2154 - val_acc: 0.9370
Epoch 18/20
500/500 [==============================] - 80s 159ms/step - loss: 0.0350 - acc: 0.9910 - val_loss: 0.2499 - val_acc: 0.9290
Epoch 19/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0353 - acc: 0.9916 - val_loss: 0.2271 - val_acc: 0.9350
Epoch 20/20
500/500 [==============================] - 79s 159ms/step - loss: 0.0438 - acc: 0.9896 - val_loss: 0.3694 - val_acc: 0.9050

500
500
accuracy:0.872
