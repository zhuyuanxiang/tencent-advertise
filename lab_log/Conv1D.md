# Comment

1.  大窗口的最高精确度没有小窗口大，说明广告数据的细节比较重要
2.  深网络的最高精度没有浅网络大，说明网络过深后很容易过拟合
3.  小窗口的深网络不容易过拟合
4.  加了正则化，可以进一步抑制过拟合
5.  取消池化效果 `MaxPooling1D()` 效果好，说明数据的细节会被池化给屏蔽了
6.  RMSProp() 的 lr 慢一点，可以避免模型过拟合

## 小窗口，浅网络

```python
data_sequence_times_with_interval ( )
Conv1D ( 32,2, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
Conv1D ( 32,2, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
GlobalMaxPooling1D ( )
optimizers.RMSprop(lr = 3e-04)
```

```txt
Epoch 1/30
 - 4s - loss: 0.7384 - binary_accuracy: 0.6571 - val_loss: 0.7238 - val_binary_accuracy: 0.6531
Epoch 2/30
 - 4s - loss: 0.7123 - binary_accuracy: 0.6574 - val_loss: 0.7071 - val_binary_accuracy: 0.6531
Epoch 3/30
 - 4s - loss: 0.6978 - binary_accuracy: 0.6574 - val_loss: 0.6984 - val_binary_accuracy: 0.6531
Epoch 4/30
 - 4s - loss: 0.6861 - binary_accuracy: 0.6574 - val_loss: 0.6887 - val_binary_accuracy: 0.6531
Epoch 5/30
 - 4s - loss: 0.6723 - binary_accuracy: 0.6574 - val_loss: 0.6770 - val_binary_accuracy: 0.6531
Epoch 6/30
 - 4s - loss: 0.6553 - binary_accuracy: 0.6574 - val_loss: 0.6623 - val_binary_accuracy: 0.6531
Epoch 7/30
 - 4s - loss: 0.6328 - binary_accuracy: 0.6574 - val_loss: 0.6432 - val_binary_accuracy: 0.6531
Epoch 8/30
 - 4s - loss: 0.6033 - binary_accuracy: 0.6577 - val_loss: 0.6196 - val_binary_accuracy: 0.6535
Epoch 9/30
 - 4s - loss: 0.5673 - binary_accuracy: 0.6720 - val_loss: 0.5933 - val_binary_accuracy: 0.6736
Epoch 10/30
 - 4s - loss: 0.5271 - binary_accuracy: 0.7512 - val_loss: 0.5697 - val_binary_accuracy: 0.7037
Epoch 11/30
 - 4s - loss: 0.4858 - binary_accuracy: 0.8458 - val_loss: 0.5437 - val_binary_accuracy: 0.7720
Epoch 12/30
 - 4s - loss: 0.4464 - binary_accuracy: 0.9127 - val_loss: 0.5250 - val_binary_accuracy: 0.7857
Epoch 13/30
 - 4s - loss: 0.4087 - binary_accuracy: 0.9280 - val_loss: 0.5100 - val_binary_accuracy: 0.7928
Epoch 14/30
 - 4s - loss: 0.3727 - binary_accuracy: 0.9350 - val_loss: 0.4961 - val_binary_accuracy: 0.8033
Epoch 15/30
 - 4s - loss: 0.3409 - binary_accuracy: 0.9385 - val_loss: 0.4862 - val_binary_accuracy: 0.8055
Epoch 16/30
 - 4s - loss: 0.3128 - binary_accuracy: 0.9412 - val_loss: 0.4811 - val_binary_accuracy: 0.8072
Epoch 17/30
 - 4s - loss: 0.2886 - binary_accuracy: 0.9443 - val_loss: 0.4790 - val_binary_accuracy: 0.8088
Epoch 18/30
 - 4s - loss: 0.2675 - binary_accuracy: 0.9470 - val_loss: 0.4767 - val_binary_accuracy: 0.8111
Epoch 19/30
 - 4s - loss: 0.2498 - binary_accuracy: 0.9496 - val_loss: 0.4761 - val_binary_accuracy: 0.8133
Epoch 20/30
 - 4s - loss: 0.2345 - binary_accuracy: 0.9523 - val_loss: 0.4813 - val_binary_accuracy: 0.8151
Epoch 21/30
 - 4s - loss: 0.2207 - binary_accuracy: 0.9550 - val_loss: 0.4811 - val_binary_accuracy: 0.8149
Epoch 22/30
 - 4s - loss: 0.2093 - binary_accuracy: 0.9573 - val_loss: 0.4908 - val_binary_accuracy: 0.8177
Epoch 23/30
 - 4s - loss: 0.1983 - binary_accuracy: 0.9608 - val_loss: 0.4923 - val_binary_accuracy: 0.8185
Epoch 24/30
 - 4s - loss: 0.1877 - binary_accuracy: 0.9627 - val_loss: 0.4958 - val_binary_accuracy: 0.8193
Epoch 25/30
 - 4s - loss: 0.1778 - binary_accuracy: 0.9654 - val_loss: 0.5133 - val_binary_accuracy: 0.8185
Epoch 26/30
 - 4s - loss: 0.1687 - binary_accuracy: 0.9676 - val_loss: 0.5068 - val_binary_accuracy: 0.8187
Epoch 27/30
 - 4s - loss: 0.1604 - binary_accuracy: 0.9693 - val_loss: 0.5231 - val_binary_accuracy: 0.8192
Epoch 28/30
 - 4s - loss: 0.1529 - binary_accuracy: 0.9717 - val_loss: 0.5283 - val_binary_accuracy: 0.8180
Epoch 29/30
 - 4s - loss: 0.1459 - binary_accuracy: 0.9732 - val_loss: 0.5288 - val_binary_accuracy: 0.8175
Epoch 30/30
 - 4s - loss: 0.1391 - binary_accuracy: 0.9750 - val_loss: 0.5432 - val_binary_accuracy: 0.8168
模型预测-->损失值 = 0.5286855862617492，精确度 = 0.8256800174713135
sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  50.75704635453063 %
前10个真实的目标数据 = [0. 1. 0. 1. 1. 0. 0. 0. 0. 1.]
前10个预测的目标数据 = [0 1 0 1 1 0 1 0 0 0]
sum(predictions>0.5) = 3700
sum(y_test_scaled) = 4293.0
sum(abs(predictions-y_test_scaled))= 2179.0
```

## 大窗口，浅网络

```python
data_sequence_times_with_interval ( )
Conv1D ( 32,5, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
Conv1D ( 32,5, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
# 再加一层，效果不好
# 窗口扩大到 7 ，效果不好
GlobalMaxPooling1D ( )
optimizers.RMSprop(lr = 2e-04)
```

```txt
Epoch 1/30
 - 5s - loss: 0.7523 - binary_accuracy: 0.6208 - val_loss: 0.7371 - val_binary_accuracy: 0.6531
Epoch 2/30
 - 5s - loss: 0.7241 - binary_accuracy: 0.6574 - val_loss: 0.7143 - val_binary_accuracy: 0.6531
Epoch 3/30
 - 5s - loss: 0.7072 - binary_accuracy: 0.6574 - val_loss: 0.7071 - val_binary_accuracy: 0.6531
Epoch 4/30
 - 5s - loss: 0.6995 - binary_accuracy: 0.6574 - val_loss: 0.7015 - val_binary_accuracy: 0.6531
Epoch 5/30
 - 5s - loss: 0.6924 - binary_accuracy: 0.6574 - val_loss: 0.6965 - val_binary_accuracy: 0.6531
Epoch 6/30
 - 5s - loss: 0.6853 - binary_accuracy: 0.6574 - val_loss: 0.6900 - val_binary_accuracy: 0.6531
Epoch 7/30
 - 5s - loss: 0.6765 - binary_accuracy: 0.6574 - val_loss: 0.6819 - val_binary_accuracy: 0.6531
Epoch 8/30
 - 5s - loss: 0.6643 - binary_accuracy: 0.6574 - val_loss: 0.6713 - val_binary_accuracy: 0.6531
Epoch 9/30
 - 4s - loss: 0.6485 - binary_accuracy: 0.6574 - val_loss: 0.6576 - val_binary_accuracy: 0.6531
Epoch 10/30
 - 5s - loss: 0.6273 - binary_accuracy: 0.6622 - val_loss: 0.6415 - val_binary_accuracy: 0.6591
Epoch 11/30
 - 5s - loss: 0.6012 - binary_accuracy: 0.6860 - val_loss: 0.6252 - val_binary_accuracy: 0.7297
Epoch 12/30
 - 5s - loss: 0.5737 - binary_accuracy: 0.7410 - val_loss: 0.6018 - val_binary_accuracy: 0.7399
Epoch 13/30
 - 5s - loss: 0.5425 - binary_accuracy: 0.7758 - val_loss: 0.5796 - val_binary_accuracy: 0.7623
Epoch 14/30
 - 5s - loss: 0.5078 - binary_accuracy: 0.8098 - val_loss: 0.5574 - val_binary_accuracy: 0.7839
Epoch 15/30
 - 5s - loss: 0.4725 - binary_accuracy: 0.8404 - val_loss: 0.5374 - val_binary_accuracy: 0.7976
Epoch 16/30
 - 5s - loss: 0.4387 - binary_accuracy: 0.8662 - val_loss: 0.5233 - val_binary_accuracy: 0.8117
Epoch 17/30
 - 5s - loss: 0.4078 - binary_accuracy: 0.8859 - val_loss: 0.5027 - val_binary_accuracy: 0.8165
Epoch 18/30
 - 5s - loss: 0.3777 - binary_accuracy: 0.9002 - val_loss: 0.4810 - val_binary_accuracy: 0.8088
Epoch 19/30
 - 5s - loss: 0.3488 - binary_accuracy: 0.9098 - val_loss: 0.4720 - val_binary_accuracy: 0.8229
Epoch 20/30
 - 5s - loss: 0.3227 - binary_accuracy: 0.9203 - val_loss: 0.4643 - val_binary_accuracy: 0.8268
Epoch 21/30
 - 5s - loss: 0.3005 - binary_accuracy: 0.9273 - val_loss: 0.4571 - val_binary_accuracy: 0.8276
Epoch 22/30
 - 4s - loss: 0.2819 - binary_accuracy: 0.9346 - val_loss: 0.4511 - val_binary_accuracy: 0.8291
Epoch 23/30
 - 4s - loss: 0.2636 - binary_accuracy: 0.9398 - val_loss: 0.4475 - val_binary_accuracy: 0.8307
Epoch 24/30
 - 4s - loss: 0.2475 - binary_accuracy: 0.9461 - val_loss: 0.4539 - val_binary_accuracy: 0.8301
Epoch 25/30
 - 5s - loss: 0.2342 - binary_accuracy: 0.9506 - val_loss: 0.4592 - val_binary_accuracy: 0.8279
Epoch 26/30
 - 5s - loss: 0.2224 - binary_accuracy: 0.9539 - val_loss: 0.4595 - val_binary_accuracy: 0.8271
Epoch 27/30
 - 4s - loss: 0.2119 - binary_accuracy: 0.9585 - val_loss: 0.4545 - val_binary_accuracy: 0.8359
Epoch 28/30
 - 5s - loss: 0.2020 - binary_accuracy: 0.9608 - val_loss: 0.4754 - val_binary_accuracy: 0.8224
Epoch 29/30
 - 5s - loss: 0.1945 - binary_accuracy: 0.9633 - val_loss: 0.4632 - val_binary_accuracy: 0.8303
Epoch 30/30
 - 5s - loss: 0.1851 - binary_accuracy: 0.9656 - val_loss: 0.4839 - val_binary_accuracy: 0.8268
模型预测-->损失值 = 0.47246567377090454，精确度 = 0.8336799740791321
sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  48.42767295597484 %
前10个真实的目标数据 = [0. 1. 0. 1. 1. 0. 0. 0. 0. 1.]
前10个预测的目标数据 = [0 1 0 1 1 0 0 0 0 1]
sum(predictions>0.5) = 3336
sum(y_test_scaled) = 4293.0
sum(abs(predictions-y_test_scaled))= 2079.0
```

## 小窗口，深网络

```python
data_sequence_times_with_interval ( )
Conv1D ( 32,2, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
Conv1D ( 32,2, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
Conv1D ( 32,2, activation = relu, kernel_regularizer = keras.regularizers.l2(0.001) )
GlobalMaxPooling1D ( )
optimizers.RMSprop(lr = 3e-04)
```

```txt
Epoch 1/30
 - 5s - loss: 0.7768 - binary_accuracy: 0.6525 - val_loss: 0.7601 - val_binary_accuracy: 0.6531
Epoch 2/30
 - 4s - loss: 0.7466 - binary_accuracy: 0.6574 - val_loss: 0.7381 - val_binary_accuracy: 0.6531
Epoch 3/30
 - 4s - loss: 0.7296 - binary_accuracy: 0.6574 - val_loss: 0.7296 - val_binary_accuracy: 0.6531
Epoch 4/30
 - 4s - loss: 0.7194 - binary_accuracy: 0.6574 - val_loss: 0.7208 - val_binary_accuracy: 0.6531
Epoch 5/30
 - 4s - loss: 0.7077 - binary_accuracy: 0.6574 - val_loss: 0.7100 - val_binary_accuracy: 0.6531
Epoch 6/30
 - 4s - loss: 0.6927 - binary_accuracy: 0.6574 - val_loss: 0.6958 - val_binary_accuracy: 0.6531
Epoch 7/30
 - 4s - loss: 0.6708 - binary_accuracy: 0.6574 - val_loss: 0.6756 - val_binary_accuracy: 0.6531
Epoch 8/30
 - 4s - loss: 0.6382 - binary_accuracy: 0.6574 - val_loss: 0.6497 - val_binary_accuracy: 0.6531
Epoch 9/30
 - 4s - loss: 0.5971 - binary_accuracy: 0.6575 - val_loss: 0.6185 - val_binary_accuracy: 0.6551
Epoch 10/30
 - 4s - loss: 0.5492 - binary_accuracy: 0.7350 - val_loss: 0.5974 - val_binary_accuracy: 0.6773
Epoch 11/30
 - 4s - loss: 0.5007 - binary_accuracy: 0.8540 - val_loss: 0.5637 - val_binary_accuracy: 0.7743
Epoch 12/30
 - 4s - loss: 0.4531 - binary_accuracy: 0.9108 - val_loss: 0.5443 - val_binary_accuracy: 0.7789
Epoch 13/30
 - 4s - loss: 0.4080 - binary_accuracy: 0.9218 - val_loss: 0.5323 - val_binary_accuracy: 0.7819
Epoch 14/30
 - 4s - loss: 0.3663 - binary_accuracy: 0.9276 - val_loss: 0.5241 - val_binary_accuracy: 0.7861
Epoch 15/30
 - 4s - loss: 0.3324 - binary_accuracy: 0.9323 - val_loss: 0.5186 - val_binary_accuracy: 0.7915
Epoch 16/30
 - 4s - loss: 0.3024 - binary_accuracy: 0.9361 - val_loss: 0.5163 - val_binary_accuracy: 0.7972
Epoch 17/30
 - 4s - loss: 0.2778 - binary_accuracy: 0.9407 - val_loss: 0.5158 - val_binary_accuracy: 0.7997
Epoch 18/30
 - 4s - loss: 0.2582 - binary_accuracy: 0.9448 - val_loss: 0.5130 - val_binary_accuracy: 0.8031
Epoch 19/30
 - 4s - loss: 0.2421 - binary_accuracy: 0.9483 - val_loss: 0.5122 - val_binary_accuracy: 0.8061
Epoch 20/30
 - 4s - loss: 0.2270 - binary_accuracy: 0.9528 - val_loss: 0.5186 - val_binary_accuracy: 0.8088
Epoch 21/30
 - 4s - loss: 0.2134 - binary_accuracy: 0.9562 - val_loss: 0.5196 - val_binary_accuracy: 0.8117
Epoch 22/30
 - 4s - loss: 0.2018 - binary_accuracy: 0.9602 - val_loss: 0.5475 - val_binary_accuracy: 0.8075
Epoch 23/30
 - 4s - loss: 0.1911 - binary_accuracy: 0.9634 - val_loss: 0.5424 - val_binary_accuracy: 0.8101
Epoch 24/30
 - 4s - loss: 0.1809 - binary_accuracy: 0.9667 - val_loss: 0.5436 - val_binary_accuracy: 0.8129
Epoch 25/30
 - 4s - loss: 0.1710 - binary_accuracy: 0.9698 - val_loss: 0.5911 - val_binary_accuracy: 0.8095
Epoch 26/30
 - 4s - loss: 0.1628 - binary_accuracy: 0.9724 - val_loss: 0.5545 - val_binary_accuracy: 0.8149
Epoch 27/30
 - 4s - loss: 0.1545 - binary_accuracy: 0.9752 - val_loss: 0.5730 - val_binary_accuracy: 0.8144
Epoch 28/30
 - 4s - loss: 0.1471 - binary_accuracy: 0.9775 - val_loss: 0.5806 - val_binary_accuracy: 0.8107
Epoch 29/30
 - 4s - loss: 0.1407 - binary_accuracy: 0.9800 - val_loss: 0.5934 - val_binary_accuracy: 0.8123
Epoch 30/30
 - 4s - loss: 0.1339 - binary_accuracy: 0.9821 - val_loss: 0.6227 - val_binary_accuracy: 0.8107
模型预测-->损失值 = 0.608517030992508，精确度 = 0.8187199831008911
sum(abs(predictions>0.5-y_test_scaled))/sum(y_test_scaled) =  52.78360121127417 %
前10个真实的目标数据 = [0. 1. 0. 1. 1. 0. 0. 0. 0. 1.]
前10个预测的目标数据 = [0 1 0 1 1 0 1 1 0 0]
sum(predictions>0.5) = 3599
sum(y_test_scaled) = 4293.0
sum(abs(predictions-y_test_scaled))= 2266.0
```
