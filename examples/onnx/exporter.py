import os, sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

now_dir = os.getcwd()
sys.path.append(now_dir)

from dataclasses import asdict
import argparse
import torch
from tqdm import tqdm
from ChatTTS.model.dvae import DVAE
from ChatTTS.config import Config
from vocos import Vocos
from vocos.pretrained import instantiate_class
import torch.jit as jit

from gpt import GPT

# disable cuda
torch.cuda.is_available = lambda: False

# add args to control which modules to export
parser = argparse.ArgumentParser()
parser.add_argument("--gpt", action="store_true", help="trace gpt")
parser.add_argument("--decoder", action="store_true", help="trace decoder")
parser.add_argument("--vocos", action="store_true", help="trace vocos")
parser.add_argument(
    "--pth_dir", default="./assets", type=str, help="path to the pth model directory"
)
parser.add_argument(
    "--out_dir", default="./tmp", type=str, help="path to output directory"
)

args = parser.parse_args()
chattts_config = Config()


def export_gpt():
    gpt_model = GPT(gpt_config=asdict(chattts_config.gpt), use_flash_attn=False).eval()
    gpt_model.from_pretrained(asdict(chattts_config.path)["gpt_ckpt_path"])
    gpt_model = gpt_model.eval()
    for param in gpt_model.parameters():
        param.requires_grad = False

    config = gpt_model.gpt.config
    layers = gpt_model.gpt.layers
    model_norm = gpt_model.gpt.norm

    NUM_OF_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 64
    TEXT_VOCAB_SIZE = gpt_model.emb_text.weight.shape[0]
    AUDIO_VOCAB_SIZE = gpt_model.emb_code[0].weight.shape[0]
    SEQ_LENGTH = 512

    folder = os.path.join(args.out_dir, "gpt")
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

        torch.onnx.export(
            model,
            (input_ids),
            f"{folder}/embedding_text.onnx",
            verbose=False,
            input_names=["input_ids"],
            output_names=["input_embed"],
            do_constant_folding=True,
            opset_version=15,
        )

    class EmbeddingCode(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def forward(self, input_ids):
            input_ids = input_ids.unsqueeze(2).expand(
                -1, -1, gpt_model.num_vq
            )  # for forward_first_code
            code_emb = [
                gpt_model.emb_code[i](input_ids[:, :, i])
                for i in range(gpt_model.num_vq)
            ]
            return torch.stack(code_emb, 2).sum(2)

    def convert_embedding_code():
        model = EmbeddingCode()
        input_ids = torch.tensor([range(SEQ_LENGTH)])

        torch.onnx.export(
            model,
            (input_ids),
            f"{folder}/embedding_code.onnx",
            verbose=False,
            input_names=["input_ids"],
            output_names=["input_embed"],
            do_constant_folding=True,
            opset_version=15,
        )

    class EmbeddingCodeCache(torch.nn.Module):  # for forward_next_code
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def forward(self, input_ids):
            code_emb = [
                gpt_model.emb_code[i](input_ids[:, :, i])
                for i in range(gpt_model.num_vq)
            ]
            return torch.stack(code_emb, 2).sum(2)

    def convert_embedding_code_cache():
        model = EmbeddingCodeCache()
        input_ids = torch.tensor(
            [[[416, 290, 166, 212]]]
        )  # torch.tensor([[range(gpt_model.num_vq)]])
        torch.onnx.export(
            model,
            (input_ids),
            f"{folder}/embedding_code_cache.onnx",
            verbose=False,
            input_names=["input_ids"],
            output_names=["input_embed"],
            do_constant_folding=True,
            opset_version=15,
        )

    class Block(torch.nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.layer = layers[layer_id]  # LlamaDecoderLayer
            self.norm = model_norm

        def forward(self, hidden_states, position_ids, attention_mask):
            hidden_states, past_kv = self.layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            present_k, present_v = past_kv
            if self.layer_id == NUM_OF_LAYERS - 1:
                hidden_states = self.norm(hidden_states)
            return hidden_states, present_k, present_v

    def convert_block(layer_id):
        model = Block(layer_id)
        hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
        position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
        attention_mask = -1000 * torch.ones(
            (1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32
        ).triu(diagonal=1)
        model(hidden_states, position_ids, attention_mask)
        torch.onnx.export(
            model,
            (hidden_states, position_ids, attention_mask),
            f"{folder}/block_{layer_id}.onnx",
            verbose=False,
            input_names=["input_states", "position_ids", "attention_mask"],
            output_names=["hidden_states", "past_k", "past_v"],
            do_constant_folding=True,
            opset_version=15,
        )

    class BlockCache(torch.nn.Module):

        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.layer = layers[layer_id]
            self.norm = model_norm

        def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
            hidden_states, past_kv = self.layer(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                past_key_value=(past_k, past_v),
                use_cache=True,
            )
            present_k, present_v = past_kv
            if self.layer_id == NUM_OF_LAYERS - 1:
                hidden_states = self.norm(hidden_states)
            return hidden_states, present_k, present_v

    def convert_block_cache(layer_id):
        model = BlockCache(layer_id)
        hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
        position_ids = torch.tensor([range(1)], dtype=torch.long)
        attention_mask = -1000 * torch.ones(
            (1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32
        ).triu(diagonal=1)
        past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))
        past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM))

        torch.onnx.export(
            model,
            (hidden_states, position_ids, attention_mask, past_k, past_v),
            f"{folder}/block_cache_{layer_id}.onnx",
            verbose=False,
            input_names=[
                "input_states",
                "position_ids",
                "attention_mask",
                "history_k",
                "history_v",
            ],
            output_names=["hidden_states", "past_k", "past_v"],
            do_constant_folding=True,
            opset_version=15,
        )

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
            model,
            (m_logits),
            f"{folder}/greedy_head_text.onnx",
            verbose=False,
            input_names=["m_logits"],
            output_names=["token"],
            do_constant_folding=True,
            opset_version=15,
        )

    def convert_greedy_head_code():
        model = GreedyHead()
        m_logits = torch.randn(1, AUDIO_VOCAB_SIZE, gpt_model.num_vq)

        torch.onnx.export(
            model,
            (m_logits),
            f"{folder}/greedy_head_code.onnx",
            verbose=False,
            input_names=["m_logits"],
            output_names=["token"],
            do_constant_folding=True,
            opset_version=15,
        )

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
            m_logits = torch.stack(
                [
                    gpt_model.head_code[i](hidden_states)
                    for i in range(gpt_model.num_vq)
                ],
                2,
            )
            return m_logits

    def convert_lm_head_text():
        model = LmHead_infer_text()
        input = torch.randn(1, HIDDEN_SIZE)

        torch.onnx.export(
            model,
            (input),
            f"{folder}/lm_head_text.onnx",
            verbose=False,
            input_names=["hidden_states"],
            output_names=["m_logits"],
            do_constant_folding=True,
            opset_version=15,
        )

    def convert_lm_head_code():
        model = LmHead_infer_code()
        input = torch.randn(1, HIDDEN_SIZE)
        torch.onnx.export(
            model,
            (input),
            f"{folder}/lm_head_code.onnx",
            verbose=False,
            input_names=["hidden_states"],
            output_names=["m_logits"],
            do_constant_folding=True,
            opset_version=15,
        )

    # export models
    print(f"Convert block & block_cache")
    for i in tqdm(range(NUM_OF_LAYERS)):
        convert_block(i)
        convert_block_cache(i)

    print(f"Convert embedding")
    convert_embedding_text()
    convert_embedding_code()
    convert_embedding_code_cache()

    print(f"Convert lm_head")
    convert_lm_head_code()
    convert_lm_head_text()

    print(f"Convert greedy_head")
    convert_greedy_head_text()
    convert_greedy_head_code()


def export_decoder():
    decoder = DVAE(
        decoder_config=asdict(chattts_config.decoder),
        dim=chattts_config.decoder.idim,
    ).eval()
    decoder.load_state_dict(
        torch.load(
            asdict(chattts_config.path)["decoder_ckpt_path"],
            weights_only=True,
            mmap=True,
        )
    )

    for param in decoder.parameters():
        param.requires_grad = False
    rand_input = torch.rand([1, 768, 1024], requires_grad=False)

    def mydec(_inp):
        return decoder(_inp, mode="decode")

    jitmodel = jit.trace(mydec, [rand_input])
    jit.save(jitmodel, f"{args.out_dir}/decoder_jit.pt")


def export_vocos():
    feature_extractor = instantiate_class(
        args=(), init=asdict(chattts_config.vocos.feature_extractor)
    )
    backbone = instantiate_class(args=(), init=asdict(chattts_config.vocos.backbone))
    head = instantiate_class(args=(), init=asdict(chattts_config.vocos.head))
    vocos = Vocos(
        feature_extractor=feature_extractor, backbone=backbone, head=head
    ).eval()
    vocos.load_state_dict(
        torch.load(
            asdict(chattts_config.path)["vocos_ckpt_path"], weights_only=True, mmap=True
        )
    )

    for param in vocos.parameters():
        param.requires_grad = False
    rand_input = torch.rand([1, 100, 2048], requires_grad=False)

    def myvocos(_inp):
        # return chat.vocos.decode(_inp) # TPU cannot support the istft OP, thus it has to be moved to postprocessing
        # reference: https://github.com/gemelo-ai/vocos.git
        x = vocos.backbone(_inp)
        x = vocos.head.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2
        )  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        return mag, x, y

    jitmodel = jit.trace(myvocos, [rand_input])
    torch.onnx.export(
        jitmodel,
        [rand_input],
        f"{args.out_dir}/vocos_1-100-2048.onnx",
        opset_version=12,
        do_constant_folding=True,
    )


if args.gpt:
    export_gpt()

if args.decoder:
    export_decoder()

if args.vocos:
    export_vocos()

print("Done. Please check the files in", args.out_dir)
