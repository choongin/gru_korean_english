import torch
import torch.nn as nn
import torch.optim as optim

# 1. 한국어-영어 데이터셋 pair 생성
data = [
    ("안녕하세요", "_Hello"),
    ("감사합니다", "_Thank you"),
    ("좋은 하루 되세요", "_Have a nice day"),
    ("반갑습니다", "_Nice to meet you"),
    ("행운을 빌어요", "_Good luck")
]


# 2. GRU 네트워크(encoder, decoder) 구조 생성
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# 모델 초기화 및 하이퍼파라미터 설정



# 문자 집합 생성
kor_chars = set("".join([pair[0] for pair in data]))
eng_chars = set("".join([pair[1] for pair in data]) + "#")  # 종료 문자 '#' 추가

kor_char2index = {char: index for index, char in enumerate(kor_chars)}
eng_char2index = {char: index for index, char in enumerate(eng_chars)}
print("kor_char2index", kor_char2index)
print("eng_char2index", eng_char2index)

hidden_size = 128
encoder = Encoder(len(kor_chars), hidden_size)
decoder = Decoder(hidden_size, len(eng_chars))

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# 나머지 코드는 동일합니다.
# 3. 한국어-영어 데이터셋 pair 학습
n_epochs = 5000
for epoch in range(n_epochs):
    for kor, eng in data:
        loss = 0
        encoder_hidden = torch.zeros(1, 1, hidden_size)

        # 인코더 학습
        for char in kor:
            input_tensor = torch.tensor([[kor_char2index[char]]])
            encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # 디코더 학습
        decoder_input = torch.tensor([[eng_char2index[eng[0]]]])  # 첫 번째 문자를 시작 토큰으로 사용
        decoder_hidden = encoder_hidden
        for char in eng[1:] + "#":  # 끝 문자 다음에 '#'를 추가하여 종료를 나타냄
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            target = torch.tensor([eng_char2index[char]])
            loss += criterion(decoder_output, target)
            decoder_input = target

        # 역전파 및 최적화
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item() / len(eng):.4f}")

# 4. 학습된 결과 확인
def translate(input_sentence):
    with torch.no_grad():
        encoder_hidden = torch.zeros(1, 1, hidden_size)
        for char in input_sentence:
            input_tensor = torch.tensor([[kor_char2index[char]]])
            _, encoder_hidden = encoder(input_tensor, encoder_hidden)

        translated = "_"  # '_'를 시작 토큰으로 가정하고 결과 문자열에 추가
        decoder_input = torch.tensor([[eng_char2index["H"]]])
        decoder_hidden = encoder_hidden

        for _ in range(20):  # 최대 길이 20 설정
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, top_index = decoder_output.topk(1)
            translated_char = list(eng_char2index.keys())[list(eng_char2index.values()).index(top_index.item())]
            if translated_char == "#":  # 종료 문자('#')를 만나면 종료
                break
            translated += translated_char
            decoder_input = top_index

        return translated


for kor, eng in data:
    print(f"{kor} -> {translate(kor)}")
