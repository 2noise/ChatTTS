import os
import torch
from tqdm import tqdm
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')
import torch.jit as jit
import torch.onnx as onnx
import ChatTTS
# from transformers.generation import TopKLogitsWarper, TopPLogitsWarper
# from ChatTTS.utils.infer_utils import CustomRepetitionPenaltyLogitsProcessorRepeat
import torch.nn.functional as F


chat = ChatTTS.Chat()
chat.load_models(source='local', custom_path='./model_files', compile=False)

gpt_model = chat.gpt.gpt.eval()

for param in gpt_model.parameters():
    param.requires_grad = False

config = gpt_model.config
layers = gpt_model.layers
model_norm = gpt_model.norm

NUM_OF_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS # 64
TEXT_VOCAB_SIZE = 21178
AUDIO_VOCAB_SIZE = 626 # config.vocab_size
SEQ_LENGTH = 512

folder = f"./tmp/onnx"
os.makedirs(folder, exist_ok=True)

for param in gpt_model.emb_text.parameters():
    param.requires_grad = False

for param in gpt_model.emb_code.parameters():
    param.requires_grad = False

for param in gpt_model.head_code.parameters():
    param.requires_grad = False

for param in gpt_model.head_text.parameters():
    param.requires_grad = False

class EmbeddingText(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        return gpt_model.emb_text(input_ids)

def convert_embedding_text():
    model = EmbeddingText()
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_text.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class EmbeddingCode(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        input_ids = input_ids.unsqueeze(2).expand(-1, -1, gpt_model.num_vq) # for forward_first_code
        code_emb = [gpt_model.emb_code[i](input_ids[:,:,i]) for i in range(gpt_model.num_vq)]
        return torch.stack(code_emb, 2).sum(2)

def convert_embedding_code():
    model = EmbeddingCode()
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_code.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class EmbeddingCodeCache(torch.nn.Module):  # for forward_next_code
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, input_ids):
        code_emb = [gpt_model.emb_code[i](input_ids[:,:,i]) for i in range(gpt_model.num_vq)]
        return torch.stack(code_emb, 2).sum(2)

def convert_embedding_code_cache():
    model = EmbeddingCodeCache()
    input_ids = torch.tensor([[[416, 290, 166, 212]]]) # torch.tensor([[range(gpt_model.num_vq)]])
    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding_code_cache.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)

class Block(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id] # LlamaDecoderLayer
        self.norm = model_norm
    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states=hidden_states,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        if(self.layer_id == NUM_OF_LAYERS - 1):
            hidden_states = self.norm(hidden_states)
        return hidden_states, present_k, present_v

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1)
    model(hidden_states, position_ids, attention_mask)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.norm = model_norm

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        if(self.layer_id == NUM_OF_LAYERS - 1):
            hidden_states = self.norm(hidden_states)
        return hidden_states, present_k, present_v

def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long) ############## shape???
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH+1), dtype=torch.float32).triu(diagonal=1)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)

class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token

def convert_greedy_head_text():   
    model = GreedyHead()
    m_logits = torch.randn(1, TEXT_VOCAB_SIZE)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head_text.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)
    
def convert_greedy_head_code():   
    model = GreedyHead()
    m_logits = torch.randn(1, AUDIO_VOCAB_SIZE, gpt_model.num_vq)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head_code.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)

# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHeadText(torch.nn.Module):
    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k, dim=1)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token
    
def convert_penalty_sample_head_text(VOCAB_SIZE):   
    model = PenaltySampleHeadText(top_k=20, min_tokens_to_keep=3)
    m_logits = torch.randn(1, VOCAB_SIZE) ### for text generation: VOCAB_SIZE
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.7])
    temperature = torch.tensor([0.7])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head_text.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)


class LmHead_infer_text(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        m_logits = gpt_model.head_text(hidden_states)
        return m_logits


class LmHead_infer_code(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states):
        m_logits = torch.stack([gpt_model.head_code[i](hidden_states) for i in range(gpt_model.num_vq)], 2)
        return m_logits

def convert_lm_head_text():
    model = LmHead_infer_text()
    input = torch.randn(1, HIDDEN_SIZE)

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head_text.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)
    
def convert_lm_head_code():
    model = LmHead_infer_code()
    input = torch.randn(1, HIDDEN_SIZE)
    print(input.shape)
    torch.onnx.export(model, (input),
                      f'{folder}/lm_head_code.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_OF_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print(f'Convert embedding')
convert_embedding_text()
convert_embedding_code()
convert_embedding_code_cache()

print(f'Convert lm_head')
convert_lm_head_code()
convert_lm_head_text()

print(f'Convert greedy_head')
convert_greedy_head_text()
convert_greedy_head_code()

print(f'Convert penalty_sample_head')
convert_penalty_sample_head_text(TEXT_VOCAB_SIZE)
print("Done")