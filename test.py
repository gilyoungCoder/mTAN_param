import torch

def efficient_spread_regularization_loss(hidden_states):
    batch_size, num_elements, _ = hidden_states.size()
    loss = 0.0

    for i in range(num_elements):
        diff = hidden_states - hidden_states[:, i:i+1, :]
        distance = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-9)
        inv_distance = 1.0 / distance
        inv_distance[:, i] = 0  # 대각선 요소를 0으로 설정
        loss += inv_distance.sum()

    loss /= (batch_size * num_elements * (num_elements - 1))
    return loss

def spread_regularization_loss(hidden_states):
    batch_size, num_elements, _ = hidden_states.size()
    
    # Expand the hidden states to calculate differences
    expanded_states = hidden_states.unsqueeze(1) - hidden_states.unsqueeze(2)
    
    # Calculate the Euclidean distance for each pair of points
    distance_matrix = torch.sqrt(torch.sum(expanded_states ** 2, dim=-1) + 1e-9)
    
    # Calculate the inverse of distances
    inv_distances = 1.0 / distance_matrix
    
    # Create a mask to zero out the diagonals
    mask = torch.eye(num_elements).to(hidden_states.device)
    mask = mask.unsqueeze(0).expand(batch_size, num_elements, num_elements)
    inv_distances = inv_distances * (1 - mask)
    
    # Calculate the mean of the inverse distances
    loss = inv_distances.sum() / (batch_size * num_elements * (num_elements - 1))
    return loss

# 예제 입력 데이터 생성
batch_size = 10  # 배치 크기
time_point = 30  # 시간 단계의 수
num_features = 20  # 특징 벡터의 차원 수
X = torch.sigmoid(torch.randn(batch_size, time_point, num_features))

# 함수 실행 및 결과 출력
penalty = spread_regularization_loss(X)
p2 = efficient_spread_regularization_loss(X)
print(f'Penalty: {penalty.item()}')
print(f'Penalty: {p2.item()}')

