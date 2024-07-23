import torch
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')
import torch.jit as jit
import torch.onnx as onnx
import ChatTTS

chat = ChatTTS.Chat()
chat.load_models(source='local', custom_path='./model_files', compile=False)

def trace_decoder():
    chat.decoder = chat.decoder.eval()
    for param in chat.decoder.parameters():
        param.requires_grad = False
    rand_input = torch.rand([1, 768, 1024], requires_grad=False)
    def mydec(_inp):
        return chat.decoder(_inp)
    
    jitmodel = jit.trace(mydec, [rand_input])
    torch.onnx.export(jitmodel, [rand_input], 'model_files/traced/dec_1-768-1024.onnx', opset_version=12)
    # jit.save(jitmodel, 'model_files/traced/dec_1-768-1024.pt')


def trace_vocos():
    chat.vocos = chat.vocos.eval()
    for param in chat.vocos.parameters():
        param.requires_grad = False
    rand_input = torch.rand([1, 100, 2048], requires_grad=False)
    
    def myvocos(_inp):
        # return chat.vocos.decode(_inp) # TPU cannot support the istft OP, thus it has to be moved to postprocessing
        # reference: https://github.com/gemelo-ai/vocos.git
        x = chat.vocos.backbone(_inp)
        x = chat.vocos.head.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        return mag, x, y
    
    jitmodel = jit.trace(myvocos, [rand_input]) 
    torch.onnx.export(jitmodel, [rand_input], 'model_files/traced/vocos_1-100-2048.onnx', opset_version=12, do_constant_folding=True)
    # jit.save(jitmodel, 'model_files/traced/vocos_1-100-2048.pt')

trace_vocos()
trace_decoder()
