# VGG-16

![vgg-16](..\img\vgg configu.jpg)

VGG-16은 D에 해당하며, 16개의 층으로 구성된 모델이다.
13개의 Convolution Layers와 3개의 Fully-connected Layers로 구성되어 있다.
zero-padding은 1이며 convoultion Layers의 stride는 1로 설정되어있다. 
또한, Polling Layers의 크기는 2*2의 maxpooling을 적용하였고, stride는 2로 설정되어있다.

![vgg configu](..\img\vgg-16.jpg)



|                         |     shape      | filter_shape |            parameter            |               sum                |         product         |
| :---------------------: | :------------: | :----------: | :-----------------------------: | :------------------------------: | :---------------------: |
|       input_data        | (224\*224\*3)  |              |                                 |                                  |                         |
|   convolution_layer_1   |                |  (3\*3\*3)   |      (3\*3\*3)\*64 = 1,728      |  (3\*3\*3-1)\*(224\*224\*64-1)   |  3\*3\*3\*224\*224*64   |
|   convolution_layer_2   |                |  (3\*3\*64)  |     (3\*3\*64)\*64 = 36,864     |  (3\*3\*64-1)\*(224\*224\*64-1)  |  3\*3\*64\*224\*224*64  |
|     pooling_layer_1     | (112\*112\*64) |    (2\*2)    |                                 |                                  |                         |
|   convolution_layer_3   |                |  (3*3\*64)   |     (3*3\*64)\*128 = 73,728     | (3\*3\*64-1)\*(112\*112\*128-1)  | 3\*3\*64\*112\*112*128  |
|   convolution_layer_4   |                | (3\*3\*128)  |   (3\*3\*128)\*128 = 147,456    | (3\*3\*128-1)\*(112\*112\*128-1) | 3\*3\*128\*112\*112*128 |
|     pooling_layer_2     | (56\*56\*128)  |    (2\*2)    |                                 |                                  |                         |
|   convolution_layer_5   |                | (3\*3\*128)  |    (3\*3\*128)*256 = 294,912    |  (3\*3\*128-1)\*(56\*56\*256-1)  |  3\*3\*128\*56\*56*256  |
|   convolution_layer_6   |                | (3\*3\*256)  |   (3\*3\*256)\*256 = 589,824    |  (3\*3\*256-1)\*(56\*56\*256-1)  |  3\*3\*256\*56\*56*256  |
|   convolution_layer_7   |                | (3\*3\*256)  |   (3\*3\*256)\*256 = 589,824    |  (3\*3\*256-1)\*(56\*56\*256-1)  |  3\*3\*256\*56\*56*256  |
|     pooling_layer_3     | (28\*28\*256)  |    (2\*2)    |                                 |                                  |                         |
|   convolution_layer_8   |                | (3\*3\*256)  |  (3\*3\*256)\*512 = 1,179,648   |  (3\*3\*256-1)\*(28\*28\*512-1)  |  3\*3\*256\*28\*28*512  |
|   convolution_layer_9   |                | (3\*3\*512)  |  (3\*3\*512)\*512 = 2,359,296   |  (3\*3\*512-1)\*(28\*28\*512-1)  |  3\*3\*512\*28\*28*512  |
|  convolution_layer_10   |                | (3\*3\*512)  |  (3\*3\*512)\*512 = 2,359,296   |  (3\*3\*512-1)\*(28\*28\*512-1)  |  3\*3\*512\*28\*28*512  |
|     pooling_layer_4     | (14\*14\*512)  |    (2\*2)    |                                 |                                  |                         |
|  convolution_layer_11   |                | (3\*3\*512)  |  (3\*3\*512)\*512 = 2,359,296   |  (3\*3\*512-1)\*(14\*14\*512-1)  |  3\*3\*512\*14\*14*512  |
|  convolution_layer_12   |                | (3\*3\*512)  |  (3\*3\*512)\*512 = 2,359,296   |  (3\*3\*512-1)\*(14\*14\*512-1)  |  3\*3\*512\*14\*14*512  |
|  convolution_layer_13   |                | (3\*3\*512)  |  (3\*3\*512)\*512 = 2,359,296   |  (3\*3\*512-1)\*(14\*14\*512-1)  |  3\*3\*512\*14\*14*512  |
|     pooling_layer_5     |  (7\*7\*512)   |    (2\*2)    |                                 |                                  |                         |
| Fully-Connected-Layer_1 |  (1\*1\*4096)  |              | (7\*7\*512)\*4096 = 102,760,448 |                0                 |          4096           |
| Fully-Connected-Layer_2 |  (1\*1\*4096)  |              |     4096*4096 = 16,777,216      |                0                 |          4096           |
| Fully-Connected-Layer_3 |  (1\*1\*1000)  |              |      4096*1000 = 4,096,000      |                0                 |          1000           |
|          TOTAL          |                |              |           138,344,128           |          13,484,968,433          |     12,572,098,560      |

3\*3 필터를 3번 혹은 2번 반복하여 사용하였다. 
2번 반복하여 사용하였을 경우 이것은 5\*5 필터를 한번 사용한것과 같은 효과를 나타낸다.
또한 3번 반복해 사용했을때는 7\*7 필터를 한번 사용한것과 같은 효과를 보여주는데, 이들은 더 많은 비선형 함수(ReLU)를 더 많이 사용할 수 있다.
또한 파라미터의 수가 감소한다. 7\*7필터를 한번 사용하게 되면 파라미터의 수는 49개이다. 하지만 3\*3필터를 3번 반복하여 사용하게 되면 파라미터의 수는 27개로 파라미터의 수가 크게 감소한다. 위의 표를 보면 약 1억3천개로 많은 파라미터인데 이것보다 더 많은 파라미터가 생기는 것이다!!!

입력 이미지는 모두 224\*224로 고정시켰지만, 학습 이미지는 256\*256 \~ 512\*512 내에서 임의의 크기로 변환하고 이를 crop하여 사용하였다.
이렇게 함으로써 얻게되는 효과가 있다.

1. 한정적인 데이터의 수를 늘릴 수 있다.
2. 하나의 객체에 대해 다양하게 학습시킬 수 있다. 
   이미지가 작을 수록 전체에 대해 학습을 시키며, 이미지가 클수록 특정 부분에 대해서만 학습을 시킬 수 있다.

이 효과는 overfitting을 방지하는데 도움이 된다.