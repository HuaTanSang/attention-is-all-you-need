import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Tính toán Scaled Dot-Product Attention.
        
        Arguments:
            Q: Tensor chứa các query, kích thước (batch_size, num_heads, seq_len, d_k)
            K: Tensor chứa các key, kích thước (batch_size, num_heads, seq_len, d_k)
            V: Tensor chứa các value, kích thước (batch_size, num_heads, seq_len, d_k)
            mask: (Tùy chọn) Tensor mask để bỏ qua các giá trị không cần thiết, kích thước phù hợp với scores
            
        Returns:
            output: Kết quả sau khi nhân attention với V, kích thước (batch_size, num_heads, seq_len, d_k)
            attn: Ma trận attention sau khi softmax, kích thước (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = K.size(-1)
        # Tính dot product giữa Q và K^T, sau đó chia cho căn bậc hai của d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
           
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        print(scores)
        # Áp dụng softmax để chuẩn hóa các scores thành phân phối xác suất
        attn = F.softmax(scores, dim=-1)
        # Nhân các trọng số attention với V
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Khởi tạo lớp Multi-Head Attention.
        
        Arguments:
            d_model: Kích thước của vector embedding (ví dụ: 512)
            num_heads: Số lượng đầu attention. Chú ý: d_model phải chia hết cho num_heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho số heads."
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Các lớp linear để biến đổi Q, K, V từ kích thước d_model về d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        print(self.W_Q)
        
        # Lớp linear để kết hợp kết quả từ các head
        self.fc = nn.Linear(d_model, d_model)
        
        # Khởi tạo lớp Scaled Dot-Product Attention đã cài đặt ở trên
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        """
        Thực hiện Multi-Head Attention.
        
        Arguments:
            Q: Tensor chứa các query, kích thước (batch_size, seq_len, d_model)
            K: Tensor chứa các key, kích thước (batch_size, seq_len, d_model)
            V: Tensor chứa các value, kích thước (batch_size, seq_len, d_model)
            mask: (Tùy chọn) Tensor mask, kích thước phù hợp với scores
        
        Returns:
            output: Tensor sau khi tính toán Multi-Head Attention, kích thước (batch_size, seq_len, d_model)
            attn: Ma trận attention của từng head, kích thước (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.size(0)
        
        # Biến đổi Q, K, V từ d_model về d_model, sau đó chia thành các head

        Q = self.W_Q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_K(K)  # (batch_size, seq_len, d_model)
        V = self.W_V(V)  # (batch_size, seq_len, d_model)
        
        
        # Reshape và chuyển vị sao cho các tensor có kích thước: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Nếu có mask, điều chỉnh kích thước của mask cho phù hợp
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        
        # Tính toán attention cho từng head
        output, attn = self.attention(Q, K, V, mask=mask)
        
        # Ghép các head lại với nhau
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Áp dụng lớp linear cuối cùng để thu gọn kết quả về kích thước d_model
        output = self.fc(output)
        return output, attn


# Ví dụ kiểm tra
if __name__ == "__main__":
    # Giả sử kích thước model là 512, số head là 8, và seq_len = 10
    d_model = 8
    num_heads = 2
    seq_len = 3
    batch_size = 2

    # Tạo các tensor giả định cho Q, K, V
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    
    # print(Q)
    
    # Khởi tạo lớp MultiHeadAttention
    mha = MultiHeadAttention(d_model, num_heads)
    output, attn = mha(Q, K, V)
    
    # sdpd = ScaledDotProductAttention()
    # output, attn = sdpd(Q, K, V)

    print("Output shape:", output.shape)  # Dự kiến: (batch_size, seq_len, d_model)
    print("Attention shape:", attn.shape) # Dự kiến: (batch_size, num_heads, seq_len, seq_len)
    print("Attention matrix:", attn)
    print("Attention output:", output)

