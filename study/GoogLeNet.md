# GoogLeNet

네트워크의 성능을 향상시키는 간단한 방법은 depth(레이어수)와 width(노드수)를 늘리는 것이다.
하지만 이 두 방법에는 파라미터의 수가 많아지며, overfitting이 일어나기도 쉽다. 또한 고품질의 학습데이터의 경우 병목현상이 발생할 수 있다.

구글은 inception 모델을 위와 같은 문제의 결과로 나타내었다.

## 구조

![GoogLeNet](..\img\GoogLeNet.jpg)

GoogLeNet은 22개의 층으로 구성되어있다.

### 특징

이 모델은1\*1 사이즈의 필터로 convolution을 해준다.

![GoogLeNet_1filter](..\img\GoogLeNet_1filter.jpg)

빨간 사각형 부분이 1\*1 convolution이 있는 곳이다.
이 모델에서 1\*1 Convolution을 넣어 채널을 줄였다가 다시 확장시켜 연산량을 감소시키게 된다.

어떻게 감소될까?
특성맵이 (14\*14\*512)일때 (5\*5\*512)인 64장의 필터로 convolution을 진행한다면 특성맵 (14\*14\*64)가 생성된다.(zero padding=2, stride=1)
이때 연산횟수는 (5\*5\*512)\*(14\*14\*64) = 160,563,200이 된다.

이걸 1\*1인 16장의 필터로 Convolution을 한다음 진행하게된다면 이때 연산 횟수는
(1\*1\*512)\*(14\*14\*16) + (5\*5\*16)\*(14\*14\*64) = 6,623,232이 된다.

다른 모델에서는 FC가 쓰였지만 이 모델에서는 FC모델 대신 global average pooling 방식을 사용하였다.
global average pooling은 이전 층의 특성맵을 평균내어 1차원으로 이어주는 것이다.
기존의 FC방식은 특성맵들을 필터 convolution을 진행하여 각각 1개의 가중치가 필요한데, global average pooling은 각 특성맵의 평균치를 연결한 것이기 때문에 가중치가 필요하지 않다.

즉, 이 모델은 연산량은 크게 늘리지 않으며 네트워크기 크기를 키울 수 있다는 것이다.

![GoogLeNet_table](..\img\GoogLeNet_table.jpg)