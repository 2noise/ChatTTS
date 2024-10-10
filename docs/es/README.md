<div align="center">

<a href="https://trendshift.io/repositories/10489" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10489" alt="2noise%2FChatTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# ChatTTS
Un modelo de generación de voz para la conversación diaria.

[![Licence](https://img.shields.io/github/license/2noise/ChatTTS?style=for-the-badge)](https://github.com/2noise/ChatTTS/blob/main/LICENSE)

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/2noise/ChatTTS/blob/main/examples/ipynb/colab.ipynb)

[**English**](../../README.md) | [**简体中文**](../cn/README.md) | [**日本語**](../jp/README.md) | [**Русский**](../ru/README.md) | **Español**
 | [**Français**](../fr/README.md) | [**한국어**](../kr/README.md)
</div>

> [!NOTE]
> Atención, es posible que esta versión no sea la última. Por favor, consulte la versión en inglés para conocer todo el contenido.

## Introducción

ChatTTS es un modelo de texto a voz diseñado específicamente para escenarios conversacionales como LLM assistant.

### Idiomas Soportados

- [x] Inglés
- [x] Chino
- [ ] Manténganse al tanto...

### Aspectos Destacados

> Puede consultar **[este video en Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)** para obtener una descripción detallada.

1. **TTS Conversacional**: ChatTTS está optimizado para tareas conversacionales, logrando una síntesis de voz natural y expresiva. Soporta múltiples hablantes, lo que facilita la generación de diálogos interactivos.
2. **Control Finas**: Este modelo puede predecir y controlar características detalladas de la prosodia, incluyendo risas, pausas e interjecciones.
3. **Mejor Prosodia**: ChatTTS supera a la mayoría de los modelos TTS de código abierto en cuanto a prosodia. Ofrecemos modelos preentrenados para apoyar estudios y desarrollos adicionales.

### Conjunto de Datos & Modelo

- El modelo principal se entrena con más de 100.000 horas de datos de audio en chino e inglés.
- La versión de código abierto en **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** es un modelo preentrenado con 40.000 horas, sin SFT.

### Hoja de Ruta

- [x] Publicar el modelo base de 40k horas y el archivo spk_stats como código abierto
- [ ] Publicar los códigos de codificador VQ y entrenamiento de Lora como código abierto
- [ ] Generación de audio en streaming sin refinar el texto
- [ ] Publicar la versión de 40k horas con control de múltiples emociones como código abierto
- [ ] ¿ChatTTS.cpp? (Se aceptan PR o un nuevo repositorio)

### Descargo de Responsabilidad

> [!Important]
> Este repositorio es sólo para fines académicos.

Este proyecto está destinado a fines educativos y estudios, y no es adecuado para ningún propósito comercial o legal. El autor no garantiza la exactitud, integridad o fiabilidad de la información. La información y los datos utilizados en este repositorio son únicamente para fines académicos y de investigación. Los datos provienen de fuentes públicas, y el autor no reclama ningún derecho de propiedad o copyright sobre ellos.

ChatTTS es un potente sistema de conversión de texto a voz. Sin embargo, es crucial utilizar esta tecnología de manera responsable y ética. Para limitar el uso de ChatTTS, hemos añadido una pequeña cantidad de ruido de alta frecuencia durante el proceso de entrenamiento del modelo de 40.000 horas y hemos comprimido la calidad del audio en formato MP3 tanto como sea posible para evitar que actores malintencionados lo usen con fines delictivos. Además, hemos entrenado internamente un modelo de detección y planeamos hacerlo de código abierto en el futuro.

### Contacto

> No dudes en enviar issues/PRs de GitHub.

#### Consultas Formales

Si desea discutir la cooperación sobre modelos y hojas de ruta, envíe un correo electrónico a **open-source@2noise.com**.

#### Chat en Línea

##### 1. Grupo QQ (Aplicación Social China)

- **Grupo 1**, 808364215 (Lleno)
- **Grupo 2**, 230696694 (Lleno)
- **Grupo 3**, 933639842

## Instalación (En Proceso)

> Se cargará en pypi pronto según https://github.com/2noise/ChatTTS/issues/269.

```bash
pip install git+https://github.com/2noise/ChatTTS
```

## Inicio
### Clonar el repositorio
```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

### Requerimientos de instalación
#### 1. Instalar directamente
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Instalar desde conda
```bash
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
```

### Inicio Rápido
#### 1. Iniciar la interfaz de usuario web (WebUI)
```bash
python examples/web/webui.py
```

#### 2. Inferir por línea de comando
> Guardará el audio en `./output_audio_xxx.wav`

```bash
python examples/cmd/run.py "Please input your text."
```

### Básico

```python
import ChatTTS
from IPython.display import Audio
import torchaudio
import torch

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["PUT YOUR TEXT HERE",]

wavs = chat.infer(texts)

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

### Avanzado

```python
###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

###################################
# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>Ejemplo: auto presentación</h4></summary>

```python
inputs_en = """
chat T T S is a text to speech model designed for dialogue applications. 
[uv_break]it supports mixed language input [uv_break]and offers multi speaker 
capabilities with precise control over prosodic elements [laugh]like like 
[uv_break]laughter[laugh], [uv_break]pauses, [uv_break]and intonation. 
[uv_break]it delivers natural and expressive speech,[uv_break]so please
[uv_break] use the project responsibly at your own risk.[uv_break]
""".replace('\n', '') # English is still experimental.

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
)

audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
torchaudio.save("output3.wav", torch.from_numpy(audio_array_en[0]), 24000)
```

<table>
<tr>
<td align="center">

**altavoz masculino**

</td>
<td align="center">

**altavoz femenino**

</td>
</tr>
<tr>
<td align="center">

[male speaker](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

</td>
<td align="center">

[female speaker](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)

</td>
</tr>
</table>


</details>

## Preguntas y Respuestas

#### 1. ¿Cuánta memoria gráfica de acceso aleatorio necesito? ¿Qué tal inferir la velocidad?
Para un clip de audio de 30 segundos, se requieren al menos 4 GB de memoria de GPU. Para la GPU 4090, puede generar audio correspondiente a aproximadamente 7 tokens semánticos por segundo. El Factor en Tiempo Real (RTF) es aproximadamente 0,3.

#### 2. La estabilidad del modelo no es lo suficientemente buena y existen problemas como varios altavoces o mala calidad del sonido.

Este es un problema común en los modelos autorregresivos (para bark y valle). Generalmente es difícil de evitar. Puede probar varias muestras para encontrar resultados adecuados.

#### 3. ¿Podemos controlar algo más que la risa? ¿Podemos controlar otras emociones?

En el modelo lanzado actualmente, las únicas unidades de control a nivel de token son `[risa]`, `[uv_break]` y `[lbreak]`. En una versión futura, es posible que abramos el código fuente del modelo con capacidades adicionales de control de emociones.

## Agradecimientos
- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS) y [valle](https://arxiv.org/abs/2301.02111) demuestran un resultado TTS notable mediante un sistema de estilo autorregresivo.
- [fish-speech](https://github.com/fishaudio/fish-speech) revela las capacidades de GVQ como tokenizador de audio para el modelado LLM.
- [vocos](https://github.com/gemelo-ai/vocos) se utiliza como codificador de voz previamente entrenado.

## Agradecimiento Especial
- [wlu-audio lab](https://audio.westlake.edu.cn/) para experimentos iniciales del algoritmo.

## Recursos Relacionados
- [Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS)

## Gracias a todos los contribuyentes por sus esfuerzos.
[![contributors](https://contrib.rocks/image?repo=2noise/ChatTTS)](https://github.com/2noise/ChatTTS/graphs/contributors)

<div align="center">

  ![counter](https://counter.seku.su/cmoe?name=chattts&theme=mbs)

</div>
