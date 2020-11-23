# 7장 CNN 코드



```python
class CnnModelBasic(AdamModel):
    def __init__(self, name, dataset, mode, hconfigs, show_maps=False, use_adam=True):
        if isinstance(hconfigs, list) and not isinstance(hconfigs[0], (list, int)):
            hconfigs = [hconfigs]
        self.show_maps = show_maps
        self.need_maps = False
        self.kernels = []
        super(CnnModelBasic, self).__init__(name, dataset, mode, hconfigs, use_adam)
```

먼저 cnnmodel의 basic를 만들었습니다. 여기서는 커널과 특징맵을 시각화 대상으로 포함할지 여부를 알려주는 show_maps 매개변숫값을 객체 변수에 저장해둡니다.  하지만 시각화 단계가 아닐 때는 show_maps 값과 관계 없이 특징맵을 수집할 필요가 없으므로 특징맵 수집이 필요한지 나타내는 별도의 프래그 need_map을 함께 두고 값을 false로 초기화합니다. 또한 모델 초기화 과정 중에 생성도는 커널을 모아 저장할 객체 변수인 kernels를 공백 리스트로 초기화합니다.

그리고 AdamModel 클래스에서 adam의 사용여부를 결정할 use_adam을 True로 설정하여 아담알고리즘이 자동으로 적용되도록했습니다. 

```python
class CnnModel(Fully,Convolution,Max_Pooling,Avg_Pooling):
    def __init__(self, name, dataset, mode, hconfigs, show_maps=False, use_adam=True):
        super(CnnModel, self).__init__(name, dataset, mode, hconfigs, show_maps, use_adam)
```

실직적인 cnnmodel을 구현한 곳이자 최 하위 클래스입니다. 이는 다음에 나올 cnn layer들의 클래스를 상속 받습니다.



```python
def alloc_layer_param(self, input_shape, hconfig):
    # print("cnn alloc_layer_param")
    layer_type = cm.get_layer_type(hconfig)

    m_name = 'alloc_{}_layer'.format(layer_type)
    method = getattr(self, m_name)
    pm, output_shape = method(input_shape, hconfig)

    return pm, output_shape
```

전체 신경망에 대한 파라미터 생성과 초기화를 담당하는 메서드입니다.

hconfig에 담긴 계층이름을 알아낸 후 이를 alloc_A_layer 형식의 메서드 이름을 만듭니다.  다음으로 이 이름의 메서드를 변숫값으로 구해내 호출하게 됩니다.

이때 생성된 메서드들은 input_shape과 hconfig의 은닉 계층 구성정보를  매개변수로 전달 받으며 필요한 처리를 마친 후 파라미터 정보와 출력 형태를 반환합니다.



```python
    def forward_layer(self, x, hconfig, pm):
        layer_type = cm.get_layer_type(hconfig)
        m_name = 'forward_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        y, aux = method(x, hconfig, pm)

        return y, aux

    def backprop_layer(self, G_y, hconfig, pm, aux):
        # print("cnn backprop_layer")
        layer_type = cm.get_layer_type(hconfig)

        m_name = 'backprop_{}_layer'.format(layer_type)
        method = getattr(self, m_name)
        G_input = method(G_y, hconfig, pm, aux)

        return G_input
```

이 메서드들 또한 전달받은 은닉층 정보를 가지고 순전파처리와 역전파 처리 메서드들을 생성하게 되는 메서드들입니다.



```python
class Fully(CnnModelBasic):
    def alloc_full_layer(self, input_shape, hconfig):
        input_cnt = np.prod(input_shape)
        output_cnt = cm.get_conf_param(hconfig, 'width', hconfig)

        weight = np.random.normal(0, self.rand_std, [input_cnt, output_cnt])
        bias = np.zeros([output_cnt])

        return {'w': weight, 'b': bias}, [output_cnt]
```

이는 완전 연결 계층(fully connected layer) 클래스로 이전에 다루웠던 부분과 크게 달라지지 않았습니다. 

이 메서드에서는 weight와 bias를 초기화하고 output_cnt를 반환 해주는 메서드입니다.

다만 이때 반환 해주는 output_cnt는 계층 구성정보를 받은 내용이 (['full', {'width':10}]) 처럼  들어 왔을 때 output_cnt는 10을 반환하게됩니다.



```python
class Convolution(CnnModelBasic):
    def alloc_conv_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        kh, kw = cm.get_conf_param_2d(hconfig, 'ksize')
        ychn = cm.get_conf_param(hconfig, 'chn')

        kernel = np.random.normal(0, self.rand_std, [kh, kw, xchn, ychn])
        bias = np.zeros([ychn])

        if self.show_maps: self.kernels.append(kernel)

        return {'k': kernel, 'b': bias}, [xh, xw, ychn]
```

합성곱 계층을 처리하는 메서드입니다. 여기서는 먼저 input_shape이 3차원인지 부터 확인을 하게 됩니다. 

그후 커널 크기와 채널 수를 은닉 계층 설정 정보로부터 키값['conv', {'ksize':3, 'chn':12}]으로 얻어낸 후 4차원의 커널과 출력 채널별로 적용될 편향 벡터를 파라미터로 생성한다.()()%#^#^%$%#%$#^



```python
class Max_Pooling(CnnModelBasic):
    def alloc_max_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        print(xh,xw,sh,sw)
        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]
```

```python
class Avg_Pooling(CnnModelBasic):
    def alloc_avg_layer(self, input_shape, hconfig):
        assert len(input_shape) == 3
        xh, xw, xchn = input_shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')

        assert xh % sh == 0
        assert xw % sw == 0

        return {}, [xh // sh, xw // sw, xchn]
```

다음으로 풀링의 파라미터를 생성하는 메서드입니다. 최대 풀링과 평균 풀링 계층 작업은 같은 각은 내용으로 진행 됩니다. 이 함수에서는 우선 합성곱 계층에서처럼 입력 형태가 3차원인지 검사합니다. 은닉 계층 설정 정보로부터 stride정보를 읽어낸 후 이번 절의 제약 조건인 이미지 크기가 건너뛰기 보폭의 배수가 되는지를 검사합니다. 커널 크기와 건너뛰기 보폭를 같은 값으로 설정하여 따로 읽어드리지 않습니다. (둘을 다르게 설정하여 진행하는 과정은 9장에서 볼 예정).

풀링 계층에서의 처리는 가변적인 요소가 없어서 파라미터가 필요 없기 때문에 파라미터 정보로 빈 딕셔너리 구조를 반환합니다. 풀링 계층에서는 건너뛰기 보폭당 한 픽셀씩으로 줄어든 해상도의 출력이 생성되며 채널 수는 변하지 않기 때문에 출력 형태는 [xh // sh, xw // sw, xchn]와 같이 반환됩니다.

다음으로 계층 생성에서 사용된 보조 함수입니다.

```python
def get_layer_type(hconfig):
    if not isinstance(hconfig, list): return 'full'
    return hconfig[0]
```

은닉 계층 구성 정보가 리스트 형식인 경우 첫 원소 값을 은닉 계층의 유형으로 보고 합니다. 

``` python
def get_conf_param(hconfig, key, defval=None):
    if not isinstance(hconfig, list): return defval
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    return hconfig[1][key]
```

은닉 계층 구성 정보에서 특정 키에 해당하는 정보를 찾아 반환하는 함수로 딕셔너리 형식을 갖는 리스트 두번째 원소에서 키에 대한 값을 찾아 반환합니다. 여러가지 예외 상황들을 검사하여 이 값을 찾을 수 없을 때는 디폴트값으로 주어진 defval값을 대신 반환합니다.

따라서 계층 구성 정보가 리스트 형식이 아닌 숫자로 주워 졌을때는 예외처리로 full을 반환 합니다. 또한 예외 처리에 의해 defval을 반환 하는데 이 인수값을 alloc_full_layer함수에서 계층 구성 정보 숫자로 주어진 상태입니다. 따라서 숫자로 지정되는 계층은 output_cnt값이 숫자인 완전 연결 계층으로 구성됩니다.

``` python
def get_conf_param_2d(hconfig, key, defval=None):
    if len(hconfig) <= 1: return defval
    if not key in hconfig[1]: return defval
    val = hconfig[1][key]
    if isinstance(val, list): return val
    return [val, val]
```

이는 get_conf_param와 비슷하게 동작합니다. 단 찾아낸 val값이 리스트가 아닐 경우 [val, val] 형태의 리스트로 반환하여 정방형 모양의 커널이나 보폭들을 더 간단하게 지정할 수 있게 하였습니다.

다음으로는 완전연결계층에대한 순전파와 역전파메서드입니다.

```python
def forward_full_layer(self, x, hconfig, pm):
    if pm is None: return x, None

    x_org_shape = x.shape

    if len(x.shape) != 2:
        mb_size = x.shape[0]
        x = x.reshape([mb_size, -1])

    affine = np.matmul(x, pm['w']) + pm['b']
    y = self.activate(affine, hconfig)

    return y, [x, y, x_org_shape]
```

여기서는 먼저 pm이 None으로 들어 왔을때는 pm값으로 x를 역전파보조를 None으로 반환하였습니다.(15장에 출력 계층의 동장윽 무력화해야하는 경우를 지원하기위한 것) 그후 2차원 보다 많은 차원이 들어 올시 미니 배치 크기의 차원을 제외한 차원을 하나의 차원으로 축소하는 처리를 추가하게 됬습니다. 그런데 역천파에서 입력의 손실기울기는 입력의 형태를 맞추어 만들어져야 하므로 원래의 역전파용 보조 정보에 기존 x의 shape 또한 추가하여 반환 하였습니니다.

또한 relu함수를 바로 사용하여 계산하는 대신 activate 함수를 호출해 은닉 계층 구성 정보에 따라 알맞은 비선형 활성화 함수를 골라 호출하도록 하였습니다.



```python
def backprop_full_layer(self, G_y, hconfig, pm, aux):
    if pm is None: return G_y

    x, y, x_org_shape = aux

    G_affine = self.activate_derv(G_y, y, hconfig)

    g_affine_weight = x.transpose()
    g_affine_input = pm['w'].transpose()

    G_weight = np.matmul(g_affine_weight, G_affine)
    G_bias = np.sum(G_affine, axis=0)
    G_input = np.matmul(G_affine, g_affine_input)

    self.update_param(pm, 'w', G_weight)
    self.update_param(pm, 'b', G_bias)

    return G_input.reshape(x_org_shape)
```

기존과 달라진 점은 순전파 처리에서 비선형 활성화 함수를 골라 이용할 수 있게 확장했던 데로 맞추어 해당 함수의 역전파 처리를 수행하는 메서드를 호출하고 있습니다. 또한 반환하는 형식을  x 형태로 변환하여 반환 됩니다.

```python
def activate(self, affine, hconfig):
    if hconfig is None: return affine

    func = cm.get_conf_param(hconfig, 'actfunc', 'relu')

    if func == 'none':        return affine
    elif func == 'relu':      return mu.relu(affine)
    elif func == 'sigmoid':   return mu.sigmoid(affine)
    elif func == 'tanh':      return mu.tanh(affine)
    else:
        assert 0

def activate_derv(self, G_y, y, hconfig):
    if hconfig is None: return G_y

    func = cm.get_conf_param(hconfig, 'actfunc', 'relu')

    if func == 'none':        return G_y
    elif func == 'relu':      return mu.relu_derv(y) * G_y
    elif func == 'sigmoid':   return mu.sigmoid_derv(y) * G_y
    elif func == 'tanh':      return mu.tanh_derv(y) * G_y
    else:
        assert 0
```

은닉 계층 구성 정보로부터 사용할 비선형 활성화 함수 이름을 얻습니다.  먼저 지원하는 함수는 relu, sigmoid, tanh 함수를 사용하게 되고 비선형 활성화 함수의 동작을 생략하는 none 또한 허용됩니다. 이때 디폴트값으로 relu를 사용하게 됩니다.

또한 각 첫줄에서 hconfig가 none일 경우 비선형 함수를 적용하지 않고 바로 반환 됩니다.



다음으로 합성곱의 순전파와 역전파의 설명입니다. 

합성곱은 채널과 미니배치 데이터 개념을 모두 반영한 상태에서의 4차원 합성곱 연산 과정을 수식으로 표현하게 될때, 입력 형태를 [mb_size, xh, xw, xchn], 커널 형태를 [kh,kw,xch,ychn], 출력 형태를 [mb_size, xh, xw, ychn]하게 되고, alloc same 패딩 방식을 사용하고 건너뛰기 처리가 없다고 가정하면 yh=xh, yw=xw가 되면서 출력은 [mb,xh,xw,ychn]의 형태가 되며 출력 원소 각각은 다음식으로 계산됩니다.
$$
y_{n,r,c,m}=\sum^{kh}_{i=1}\sum^{kw}_{j=1}\sum^{xchn}_{k=1}k_{i,j,k,m}x_{n,r+i-bh,c+j-bw,k}
$$
 이 식에서 커널 원소에 곱해지는 x좌표는 (r + i - bh, c + j - bw)이다. 이는 출력 좌표(r,c)에 대응하는 입력 좌표를 구한 것인데 우선(i,j)를 더해 커널 좌표만큼 평행 이동시킨 후 여기에 다시(-bh, -bw)를 더해 출력 위치와 같은(r,c) 위치의 입력 픽셀이 커널 중심에 맞추어지도록 좌표를 보정하기 위한 것이다. 이때(bh,bw)는 (kh/2,kw/2)가적당하지만 소수점 성분을 갖는 실수 좌푯값이 발생을 막기 위하여 정수형 나눗셈 연산자 //를 이용하여 구한것이다. 물론 이렇게 구해지는 좌표가 정상적인 입력 범위를 벗어나는 경우 해당 좌표의 x값은 0으로 간주한다.



```python
def forward_conv_layer_adhoc(self, x, hconfig, pm):
    # print("cnn forward_conv_layer_adhoc")
    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    conv = np.zeros((mb_size, xh, xw, ychn))

    for n in range(mb_size):
        for r in range(xh):
            for c in range(xw):
                for ym in range(ychn):
                    for i in range(kh):
                        for j in range(kw):
                            rx = r + i - (kh - 1) // 2
                            cx = c + j - (kw - 1) // 2
                            if rx < 0 or rx >= xh: continue
                            if cx < 0 or cx >= xw: continue
                            for xm in range(xchn):
                                kval = pm['k'][i][j][xm][ym]
                                ival = x[n][rx][cx][xm]
                                conv[n][r][c][ym] += kval * ival

    y = self.activate(conv + pm['b'], hconfig)

    return y, [x, y]
```



여기서는 7중 반복문을 사용하여 합성곱 연산의 순전파 처리를 하게 됩니다. 이때 처음 4중 반복문은 4차원 출력 텐서 각각의 픽셀 위치를 지정하게 되고, 안쪽 3중 반복문은 커널의 두차원과 입력 채널 등 세차원에 걸쳐 대응하는 입력 픽셀값과 커널 가중치를 찾아 그 곱을 누적시키는 방법으로 출력 픽셀값을 구하게 됩니다. 편향은 커널 가중치 처리 결과 얻어진 4차원 텐서에 편향 벡터를 일괄적으로 더하는 방법으로 반영합니다. 한편 두번의 if문은 범위를 벗어난는 입력 픽셀을 누적 계산에서 배제시키기 위한 것으로 이 처리 덕분에 범위 바깥의 입력값은 0으로 간주하는 효과(SAME 패딩)를 갖게 됩니다.

세가지 합성곱연산 방법에서 첫번째에 해당하는 이방법은 합성곱 계층이 어떤 과정을 거쳐 계산되는지 가장 분명하게 보여주지만 실행 효율이 너무 나빠 실제로는 이용하지 않으며 따라서 메서드로 등록하지 않았습니다.



```python
def forward_conv_layer_better(self, x, hconfig, pm):
    # print("cnn forward_conv_layer_better")
    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    conv = np.zeros((mb_size, xh, xw, ychn))

    bh, bw = (kh - 1) // 2, (kw - 1) // 2
    eh, ew = xh + kh - 1, xw + kw - 1

    x_ext = np.zeros((mb_size, eh, ew, xchn))
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x

    k_flat = pm['k'].transpose([3, 0, 1, 2]).reshape([ychn, -1])

    for n in range(mb_size):
        for r in range(xh):
            for c in range(xw):
                for ym in range(ychn):
                    xe_flat = x_ext[n, r:r + kh, c:c + kw, :].flatten()
                    conv[n, r, c, ym] = (xe_flat * k_flat[ym]).sum()

    y = self.activate(conv + pm['b'], hconfig)

    return y, [x, y]
```

이 함수  안쪽에 있던 3중 반복을 없애고 대신 계산에 이용될 입력과 커널의 3차원 성분들을 각각 하나의 벡터로 차원을 축소 시킨 후 단 한번의 내적 계산으로 출력 픽셀값을 구하게 됩니다. 이때 내적 처리 과정에서 출력 픽셀 위치에 대응하는 입력 영역을 따질 때 범위를 벗어나는 입력값의 처리가 문제가 됩니다. 이를 해결하기 위해 반복 처리 전에 커널 크기를 고려해 확장된 버퍼를 준비해 0으로 채운 후 버퍼의 중앙 부분에 입력을 복사하게 됩니다.

또한 출력 채널에 따라 나머지 세 차원의 커널 원소들이 내적 계산에 반복 이용되고 있습니다. 따라서 반복 처리 실행 전에 미리 출력 채널별로 커널의 나머지 세차원을 한 차원 벡터로 축소시켜 저장해두었습니다.

반복문 안에서는 확장된 x_ext 버퍼에서 출력 픽셀 위치에 대응하는 커널 면적의 사각 영역을 지정해 차원을 축소하여 내적 계산에 이용할 벡터을 구한다. 확장 버퍼를 이용하는 덕분에 범위를 벗어나는 입력 문제를 별도의 조전 처리 없이 해결할 수 있다. 입력 성분과 커널 성붕을 적절히 벡터화 했기 때문에 이제 xe_flat 벡터와 kflat[ym]벡터의 내적을 계산하는 방법으로 간단히 출력 픽셀 하나의 값을 구할 수 있다. 

```python
def forward_conv_layer(self, x, hconfig, pm):
    # print("cnn forward_conv_layer")

    mb_size, xh, xw, xchn = x.shape
    kh, kw, _, ychn = pm['k'].shape

    x_flat = cm.get_ext_regions_for_conv(x, kh, kw)

    k_flat = pm['k'].reshape([kh * kw * xchn, ychn])
    conv_flat = np.matmul(x_flat, k_flat)
    conv = conv_flat.reshape([mb_size, xh, xw, ychn])

    y = self.activate(conv + pm['b'], hconfig)

    if self.need_maps: self.maps.append(y)

    return y, [x_flat, k_flat, x, y]
```



이는 위에 함수들에서 사용한 반복문들을 전부 없에고 numpy 패키지를 활요하여 구현하게 됩니다. 이방법에서는 입력x와 커널pm['k']의 차원을 축소시켜 행렬 x_flat과 k_flat을 구한 후 행렬 곱셈 한 번으로 계산을 끝나게 됩니다.

먼저 x_flat은 get_ext_regions_for_conv 함수를 통해 [mb_size * xh * xw, kh * kw * xchn]형태의 차원 축소 되어 저장되어야 합니다. 

커널은[kh ,kw , xchn, ychn] 4차원 형태를 [kh * kw * xchn, ychn] 형태의 행렬로 바꾸어야 합니다. 이렇게 하면 행 크기가 앞에서 살펴본 확장 입력 행렬의 열 크기와 같아져 행렬 곱셈이 가능해집니다.

이제 두행렬을 곱하여 [mb_size * xh * xw, ychn] 형태의 행렬 conv_flat을 얻게 됩니다. 행렬 곱셈에서는 확장 입력 행렬의 각 행과 변환된 커널 행렬의 각 열 사이의 모든 내적 값들이 계산됩니다. 그리고 이 각각의 내적 계산은  7중 반복문함수의 4중 반복문 안에서 3중반복을 이용해 계산하던 내용과 정학하게 일치하게 됩니다. 게다가 이 conv_flat행렬에는 출력 픽셀값들이 순서대로 잘 담겨 있기 때문에 reshape 명령을 이용해 [mb_size, xh, xw, ychn] 형태의 4차원 형태로 재해석하는 것만으로 합성곱 연산이 끝나게 됩니다.

이렇게 계산된 합성곱연산 결과에 출력채널마다 하나씩 값을 갖는 편향 벡터값을 더하고 비선형 활성화 함수 적용까지 마치면 합성곱 계층의 순천파 처리가 끝나게 됩니다. 

이후 나오는 need_maps는 시각화 처리를 할때 사용하게 됩니다. 시각화 처리를 할때 만 ture를 반영하여 maps에 y을 저장하게 됩니다.

합성곱 연산의 핵심이 행렬 곱셈인 만큼 역전파 과정에서 곱셈에 이용된 두 행렬의 내용이 필요합니다. 또한 역전파용 보조 정보로는 형태 파악에 필요한 x와 활성화 함수 역전파에 이용할 y까지 포함시켜 [x_flat, k_flat, x, y]를 반환합니다.



```python
def get_ext_regions_for_conv(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape
    regs = get_ext_regions(x, kh, kw)
    regs = regs.transpose([2, 0, 1, 3, 4, 5])

    return regs.reshape([mb_size * xh * xw, kh * kw * xchn])


def get_ext_regions(x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    x_ext = np.zeros((mb_size, eh, ew, xchn), dtype='float32') 
    x_ext[:, bh:bh + xh, bw:bw + xw, :] = x
    regs = np.zeros((xh, xw, mb_size * kh * kw * xchn), dtype='float32')

    for r in range(xh):
        for c in range(xw):
            regs[r, c, :] = x_ext[:, r:r + kh, c:c + kw, :].flatten()

    return regs.reshape([xh, xw, mb_size, kh, kw, xchn])
```

x_flat을 계산할때 사용 된 메서드입니다. 입력의 확장과 차원 축소 작업을 처리한 후  그 결과를 합성곱 연산에 알맞게 손질하여 반환한다. 또한 get_ext_regions 함수에서는 2중 반복문을 이용하여 이미지의 픽셀 좌표별로 커널 크기의 인근 영역에 대해 미니배치 데이터 전체와 입력 채널 전체 등 4차원 공간에서 해당되는 원소들을 찾아 일괄적으로 차원을 축소시켜 하나의 벡터로 만든다. 

이중 반복 처리를 이용해 차원 축소 작업을 수행한결과[xh, xw, mb_size * kh * kw * xchn] 형태의 3차원 텐서가 얻어진다. 이후 이 텐서의 데이터 순서를 유지한 채 [xh, xw, mb_size, kh, kw, xchn]의 6차원 텐서로 형태만 재해석해 보고한다. 또한 이 보고를 받은 get_ext_regions_for_conv 메서드는 미니배치 축을 제일 앞으로 옮기는방식으로 데이터 순서를 변경한 후 세 축씩을 묶어 합성곱 연산에 필요한[mb_size * xh * xw, kh * kw * xchn] 형태의 2차원 행렬로 재해석해 처리결과로 보고한다.



```python
    def backprop_conv_layer(self, G_y, hconfig, pm, aux):
        # print("cnn backprop_conv_layer")
        x_flat, k_flat, x, y = aux

        kh, kw, xchn, ychn = pm['k'].shape
        mb_size, xh, xw, _ = G_y.shape

        G_conv = self.activate_derv(G_y, y, hconfig)

        G_conv_flat = G_conv.reshape(mb_size * xh * xw, ychn)

        g_conv_k_flat = x_flat.transpose()
        g_conv_x_flat = k_flat.transpose()

        G_k_flat = np.matmul(g_conv_k_flat, G_conv_flat)
        G_x_flat = np.matmul(G_conv_flat, g_conv_x_flat)
        G_bias = np.sum(G_conv_flat, axis=0)

        G_kernel = G_k_flat.reshape([kh, kw, xchn, ychn])
        G_input = cm.undo_ext_regions_for_conv(G_x_flat, x, kh, kw)

        self.update_param(pm, 'k', G_kernel)
        self.update_param(pm, 'b', G_bias)

        return G_input
```

비선형 활성화 함수에 대한 역전파 처리를 수 행한다. 이어서 2차원 행렬을 4차원 텐서로 재해석했던 순전파 과정에서 처리에 대한 역처리로서 4차원 행렬을 2차원 행렬로 재해석한다 그후 과정은 기존과 같은 과정입니다.

2차원의 g_k_flat을 커널의 원래 형태은 4차원 텐서로 재해석하여 G_kernerl 을 얻는다.

순전파에서 수행한 get_ext_regions_forconv() 훔수에 대한 역처리를 수행하는 undo_ext_regions_for_conv()함수를 호출해 G_x_flat으로부터 G_input을 얻는다.

이제 G_kernel과 G_bias를 update_param() 메서드에 전달하여 커널 가중치 및 편향 파라미터를 수정한다. 이때 가중치 텐서의 키값으로 커널을 의미하는 'k'를 전달한다. G_input 텐서를 반환하면 합성곱 계층의 역전파 처리가 끝난다.



```python
def undo_ext_regions_for_conv(regs, x, kh, kw):
    mb_size, xh, xw, xchn = x.shape

    regs = regs.reshape([mb_size, xh, xw, kh, kw, xchn])
    regs = regs.transpose([1, 2, 0, 3, 4, 5])

    return undo_ext_regions(regs, kh, kw)


def undo_ext_regions(regs, kh, kw):
    xh, xw, mb_size, kh, kw, xchn = regs.shape

    eh, ew = xh + kh - 1, xw + kw - 1
    bh, bw = (kh - 1) // 2, (kw - 1) // 2

    gx_ext = np.zeros([mb_size, eh, ew, xchn], dtype='float32')

    for r in range(xh):
        for c in range(xw):
            gx_ext[:, r:r + kh, c:c + kw, :] += regs[r, c]

    return gx_ext[:, bh:bh + xh, bw:bw + xw, :]
```

순전파와 반대 순서로 반대 작업을 하기 위해 regs 텐서의 형태를 재해석하고 맨 앞의 미니배치 축을 세번째 축으로 옮긴다. 그런후에 get_ext_regions 함수에 대한 역처리를 수행하는 undo_ext_regions함수를 호출한다. 

한편 undo_ext_regions 함수에서는 커널을 이용한 합성곱 연산을 여러 입력  픽셀의 정보를 종합해 출력 픽셀을 만드는 과정이다. 따라서 한 입력 픽셀이 여러 출력 픽셀의 계산에 이용된다. 따라서 get_ext_retgions함수에서는 커널 크기가 반영된 넓은 공산에 더 적은 수의 입력값들을 복사하는 과정에서 각 입력 픽셀은 자기가 값에 영향을 미치는 출력 픽셀 수만큼 반복해서 복사된다.

그런데 순전파 과정에서 여기저기 중복하여 이용되는 성분의 손실기울기는 영향을 미친 각각의 성분에 관해 구한 손실 기울기들의 합으로 계산하여야 한다. 따라서 각 입력 픽셀의 손실 기울기는 undo_ext_regions 함수의 2중 반복문 처리에서는 get_ext_regions 함수에서와 반대로 좁은 공간에 더 많은 수의 값들을 누적시키는 방식으로 복사한다.



```python
    def forward_avg_layer(self, x, hconfig, pm):
        # print("cnn forward_avg_layer")
        mb_size, xh, xw, chn = x.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        yh, yw = xh // sh, xw // sw

        x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
        x2 = x1.transpose(0, 1, 3, 5, 2, 4)
        x3 = x2.reshape([-1, sh * sw])

        y_flat = np.average(x3, 1)
        y = y_flat.reshape([mb_size, yh, yw, chn])

        if self.need_maps: self.maps.append(y)

        return y, None

    def backprop_avg_layer(self, G_y, hconfig, pm, aux):
        # print("cnn backprop_avg_layer")
        mb_size, yh, yw, chn = G_y.shape
        sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
        xh, xw = yh * sh, yw * sw

        gy_flat = G_y.flatten() / (sh * sw)

        gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
        for i in range(sh * sw):
            gx1[:, i] = gy_flat
        gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
        gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

        G_input = gx3.reshape([mb_size, xh, xw, chn])

        return G_input
```

커널 크기가 건너뛰기 보폭과 같고 입력 이미지 크기가 이 보폭의 배수가 되도록 풀링 연산에 제한을 주었었다. 따라서 보폭 크기에 따라 이미지를 분할한 후 분할 된 각 구역을 일차원 벡터로 차원 축소시켜 풀링처리를 가능하게 하였습니다. 

이를 위하여 [mb_size,xh, xw, chn] 형태의 입력 x의 가로 방향 축과 세로 방향축을 보폭 크기만큼 분할하여

[mb_size, yh, sh, yw, sw, chn] 형태의 6차원 텐서로 재해석한 후 축 순서를 변경하여 [mb_size, yh, yw, chn,sh, sw] 형태로 재해석 하였습니다. 그이후 앞의 네 축을 한데 묶고 뒤 두축을 한데 묶어 2차원 테서로 재해석하게 됩니다.  이제 두 번째 축 원소들에 대한 평균을 구하면 [mb_size * yh * yw * chn]형태의 평균값 벡터를 얻게 됩니다. 이 벡터를 [mb_size, yh, yw, chn] 형태의 4차원 텐서로 재해석 하게 되면 반환할 y 값이 완성됩니다.

역전파처리는 4차원 형태의 손실 기울기를 먼저 1차원 벡터 형태로 축소시키는데 이때 (sh * sw)값으로 나눗셈이 함께 계산됩니다. 이는 평균 연산의 경우 손실 기울기가 평균 계산에 참여한 모든 원소에 같은 비율로 분배되기 때문에 이를 미리 반영한 것입니다.

이후 [mb_size * yh * yw * chn, sh * sw]형태의 버퍼를 마련 후 gy_flat를 보든 행에 복사를 하게 도비니다. 이후 6차원 텐서로 재해석한 후 다시 축 순서를 바꾸게 됩니다. 마지막으로 분할된 두축을 다시모으게 되면 G_input이 구해집니다.



```python
def forward_max_layer(self, x, hconfig, pm):
    # print("cnn forward_max_layer")
    mb_size, xh, xw, chn = x.shape
    sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
    yh, yw = xh // sh, xw // sw

    x1 = x.reshape([mb_size, yh, sh, yw, sw, chn])
    x2 = x1.transpose(0, 1, 3, 5, 2, 4)
    x3 = x2.reshape([-1, sh * sw])

    idxs = np.argmax(x3, axis=1)
    y_flat = x3[np.arange(mb_size * yh * yw * chn), idxs]
    y = y_flat.reshape([mb_size, yh, yw, chn])

    if self.need_maps: self.maps.append(y)

    return y, idxs

def backprop_max_layer(self, G_y, hconfig, pm, aux):
    # print("cnn backprop_max_layer")
    idxs = aux

    mb_size, yh, yw, chn = G_y.shape
    sh, sw = cm.get_conf_param_2d(hconfig, 'stride')
    xh, xw = yh * sh, yw * sw

    gy_flat = G_y.flatten()

    gx1 = np.zeros([mb_size * yh * yw * chn, sh * sw], dtype='float32')
    gx1[np.arange(mb_size * yh * yw * chn), idxs] = gy_flat[:]
    gx2 = gx1.reshape([mb_size, yh, yw, chn, sh, sw])
    gx3 = gx2.transpose([0, 1, 4, 2, 5, 3])

    G_input = gx3.reshape([mb_size, xh, xw, chn])

    return G_input
```

최대치 풀링계층에 대한 처리 과정은평균치 풀링 계층에 대한 처리과정과 비슷합니다. 

차이점으로는 최대치의 위치를 idxs 벡터에 모으고 이를 인덱스 삼아 y_flat을 구하는 방식으로 처리합니다.

역전파 처리에서는 최대치 풀링에 맞추어 출력에 반영된 최대치 성분의 부분 기울기는 1이고 반영이 안된 성분의 부분기울기는 0으로 처리된다. 이때 동점자가 있더라도 그중 하나의 위치만 손실 기울기를 몰아준다.

```python
def load_visualize(self, num):
    print("cnn visualize")
    print('Model {} Visualization'.format(self.name))

    self.need_maps = self.show_maps
    self.maps = []

    deX, deY = self.dataset.dataset_get_validate_data(num)
    est = self.get_estimate(deX)
    print(self.show_maps)
    if self.show_maps:
        for kernel in self.kernels:
            kh, kw, xchn, ychn = kernel.shape
            grids = kernel.reshape([kh, kw, -1]).transpose(2, 0, 1)
            mu.draw_images_horz(grids[0:5, :, :])

        for pmap in self.maps:
            mu.draw_images_horz(pmap[:, :, :, 0])

    self.dataset.visualize(deX, est, deY)

    self.need_maps = False
    self.maps = None
```

먼저 need_maps를 show_maps로 설정해 특징 맵 수집 여부를 지정하여 특징맵 수집 버퍼인 maps도 공백 리스트로 초기화합니다.

데이터 처리 후에는 커널 출력을 수행하는 각 합성곱 계층의 커널에 대해 이미지 해상도와 관련없는 두 축을 앞으로 옮긴 후 합병하고 나서 앞의 다섯개를 고는 방법으로 이미지 해상도와 관련된 두 축을 갖는 2차원 형태의 커널 슬라이스를 추출해 출력한다.

특징맵의 경우 합성곱 계층이나 풀링 계층의 출력 모두들 수집한 maps 리스트 내용을 차례로 출력 하되 출력 대상을 첫번째 채널로 제한해 지나치게 많은 출력을 막는다. 다음으로 는 기존과 같은 시각화 내용을 출력하도록 하며 마지막으로 need_maps와 maps를 초기화해 주어 향후 잘못된 영향이나 메모리 부담을 미치지 않도록 한다.



```
Model flower train started:
    Epoch 2: cost=1.132, accuracy=0.555/0.490 (158/158 secs)
    Epoch 4: cost=0.894, accuracy=0.668/0.500 (158/316 secs)
    Epoch 6: cost=0.730, accuracy=0.727/0.550 (158/474 secs)
    Epoch 8: cost=0.616, accuracy=0.765/0.610 (160/634 secs)
    Epoch 10: cost=0.532, accuracy=0.803/0.530 (159/793 secs)
Model flower train ended in 793 secs:
Model flower test report: accuracy = 0.500, (0 secs)

cnn visualize
Model flower Visualization
True
추정확률분포 [ 0, 0, 0, 0, 0,100] => 추정 tulip : 정답 tulip => O
추정확률분포 [ 1, 1, 0, 1,83,14] => 추정 sunflower : 정답 sunflower => O
추정확률분포 [ 0, 0, 0, 0, 0,100] => 추정 tulip : 정답 tulip => O
```

![image-20201123173216362](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123173216362.png)



![image-20201123171443604](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171443604.png)



![image-20201123171501206](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171501206.png)

![image-20201123171510851](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171510851.png)

![image-20201123171518098](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171518098.png)

![image-20201123171526682](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171526682.png)

![image-20201123171535976](C:\Users\jm\AppData\Roaming\Typora\typora-user-images\image-20201123171535976.png)





``` python
    od = officedata.Office31Dataset([96, 96], [96, 96, 3])
    mode = modelmode.Office_Select(od.cnts)

    om1 = CnnModel('office31_model_1', od, mode,
                   [['conv', {'ksize': 3, 'chn': 6}],
                    ['max', {'stride': 2}],
                    ['conv', {'ksize': 3, 'chn': 12}],
                    ['max', {'stride': 2}],
                    ['conv', {'ksize': 3, 'chn': 24}],
                    ['avg', {'stride': 3}]])
    om1.exec_all(epoch_count=40, report=10, show_cnt=0)
```

```
Model office31_model_1 train started:
    Epoch 10: cost=30.748, accuracy=0.625/0.780 (750/750 secs)
    Epoch 20: cost=30.039, accuracy=0.647/0.690 (686/1436 secs)
    Epoch 30: cost=29.751, accuracy=0.654/0.760 (658/2094 secs)
    Epoch 40: cost=29.611, accuracy=0.656/0.710 (655/2749 secs)
Model office31_model_1 train ended in 2749 secs:
Model office31_model_1 test report: accuracy = 0.650, (1 secs)
```



