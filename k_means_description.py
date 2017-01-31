#K-평균 알고리즘 : 데이터를 다른 묶음과 구분되도록 유사한 것 끼리 자동으로 그룹화하는 알고리즘

import numpy as np

num_points = 2000
vectors_set = []

#랜덤한 임의의 점들을 2000개 생성
for i in xrange(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0,0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

import tensorflow as tf

#데이터를 텐서로 옮김, 무작위로 생성한 데이터를 가지고 상수 텐서를 만듦
vectors = tf.constant(vectors_set)

#K-평균 알고리즘의 초기단계(0단계) : 초기 중심 지정 - 입력 데이터에서 무작위로 K개의 데이터를 선택
#텐서플로가 입력 데이터를 무작위로 섞어서 K개의 중심을 선택
#Variable 매서드 : 텐서플로 내부의 그래프 자료구조에 만들어질 하나의 변수를 정의
#k개의 데이터 포인트는 2D텐서로 저장
k=4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

#유클리드 제곱 거리를 이용해 가장 가까운 중심을 계산
#유클리드 제곱거리 : d^2(vector, centroid) = (vector_x - centroid_x)^2 + (vector_y - centroid_y)^2
#이 계산을 하려는 두 텐서가 모두 2차원이긴 하지만 1차원의 크기가 다르다 : 2000과 4
#해결을 위해 두 텐서에 차원을 추가
expanded_vecters = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
#결과 : expanded_vectors : TensorShape([Dimension(1), [Dimension(2000), [Dimension(2)])
#결과 : expanded_centroides : TensorShape([Dimension(4), [Dimension(1), [Dimension (2)])
#차원 브로드 캐스팅 : expanded_vectors의 D0차원을 4로 늘려서 연산, D0의 각 원소별로 뺄셈이 이루어 진다

#할당 단계(1단계)
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vecters, expanded_centroides)), 2), 0)
#동일한 코드 4줄로 나타낸다면,,
#diff = tf.sub(expanded_vecters, expanded_centroides) : expanded_vectors와 expanded_centroides의 뺄셈값
#sqr = tf.square(diff) : diff의 제곱값
#distances = tf.reduce_sum(sqr, 2) : 지정한 차원에 따라 원소를 더한다 : x^2 + y^2
#assignments = tf.argmin(distances, 0) : 지정한 차원에서 가장 작은 값을 리턴 --> TensorShape([Dimension(2000)]) 리턴(각 벡터가 속한 군집)

#수정단계(2단계) : 새로운 중심 계산하기
#K개의 군집에 속하는 점들의 평균을 가진 K개의 탠서를 합쳐서 mean텐서를만든다
#1. equal함수 : 한 군집과 매칭되는 assignments 텐서의 각 원소 위치를 True로 표시하는 불리언 텐서(Dimension(2000)) 만듧
#2. where함수 : 매개 변수로 받은 불리안 텐서에서 True로 표시된 위치 값으로 가지는 텐서(Dimension(2000)*Dimension(1))을 만륾
#3. reshape함수 : c군집에 속한 vectors 텐서 포인트들의 인덱스로 구성된 텐서 (Dimension(1)*Dimension(2000))을 만듦
#4. gather함수: c군집을 이루는 점들의 좌표를 모은 텐서(Dimension(1)*Dimension(2000)*Dimension(2))를 만듦
#5. reduce_mean함수: c군집에 속한 모든 점의 평균 값을 가진 텐서 (Dimension(1)*Dimension(2))를 만듦
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in xrange(k)])

#새로 계산된 중심값을 업데이트
update_centroides = tf.assign(centroides, means)

#변수 초기화
init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

#코드를 통해 매 반복마다 중심은 업데이트되고 각점은 새롭게 군집에 할당
for step in xrange(100):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
