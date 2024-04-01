import torch

# 위에서 정의한 batch_cosine_similarity_penalty 함수
def batch_cosine_similarity_penalty(X):
    batch_size, time_point, num_features = X.shape
    penalties = []
    for batch in range(batch_size):
        current_batch = X[batch]
        norm = torch.norm(current_batch, p=2, dim=1, keepdim=True)
        current_batch_norm = current_batch / (norm + 1e-6)
        print(current_batch_norm)
        similarity_matrix = torch.matmul(current_batch_norm, current_batch_norm.T)
        print(similarity_matrix)
        num_elements = time_point
        penalty = (similarity_matrix.sum() - num_elements) / (num_elements * (num_elements - 1))
        penalties.append(penalty)
    average_penalty = sum(penalties) / batch_size
    return average_penalty

# 예제 입력 데이터 생성
batch_size = 1  # 배치 크기
time_point = 3  # 시간 단계의 수
num_features = 2  # 특징 벡터의 차원 수
X = torch.sigmoid(torch.randn(batch_size, time_point, num_features))

# 함수 실행 및 결과 출력
penalty = batch_cosine_similarity_penalty(X)
print(X)
print(f'Penalty: {penalty.item()}')
