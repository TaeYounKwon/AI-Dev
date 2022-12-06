# 주성분을 사용해 특성 줄이기
# 사이킷런 손글씨 데이터를 활용
# 특성 행렬을 표준화 처리 및 주성분 특성 줄이기
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits() # 8x8 크기의 손글씨 숫자 데이터 로드

# 특성 행렬을 표준화 처리
features = StandardScaler().fit_transform(digits.data)

# 99% 분산을 유지하도록 PCA 클래스 객체 생성
pca = PCA(n_components=0.99, whiten=True)
features_pca = pca.fit_transform(features)
print("원본 특성 개수 : ",features.shape[1])
print("줄어든 특성 개수 : ", features_pca.shape[1])

# # 주성분에 투영된 처음 두 개의 특성을 사용해 산점도 출력
# plt.scatter(features_pca[:, 0], features_pca[:, 1])
# plt.show()

# 화이트닝
pca_nowhiten = PCA(n_components=0.99)
features_nowhiten = pca_nowhiten.fit_transform(features)
plt.scatter(features_nowhiten[:, 0], features_nowhiten[:, 1])
plt.show()

# 특성 행렬을 주성분에 투영하려면 components_ 배열 전치하여 행렬곱을 수행합니다.
# 넘파이 allclose()를 사용하여 features_nowhiten 배열과 동일한지 확인
np.allclose(features_nowhiten, np.dot(features, pca_nowhiten.components_.T))

pca = PCA(whiten=True).fit(features)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()


# 넘파이 cumsum()를 사용하여 분산을 누적하여 그래프 출력
# 대략 30개의 주성분으로도 80이상의 분산을 유지
# 표준화하지 않은 원본 데이터 사용
pca.fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()


'''
* n_components의 입력 매개변수값이 1보다 크면 n_components개수만큼의 특성이 반환
* n_componentsfmf 0과 1 사이로 지정하면 pca는 해당 비율의 분산을 유지할 수 있는 최소한의 특성 개수를 반환
* 원본 특성의 95%와 99%의 분산을 유지하는 0.95어ㅣ 0.99가 사용됨
* whiten=True로 지정하면 각 주성분의 값을 평균이 0이고 분산이 1이 되도록 변환
* solver='randomized'는 아주 짧은 시간 안에 첫 번째 주성분을 찾아주는 확률적 알고리즘을 사용
* 화이트닝은 주성분에 투영된 특성의 스케일을 맞추는 역할
'''