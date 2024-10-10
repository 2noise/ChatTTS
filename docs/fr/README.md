<div align="center">

<a href="https://trendshift.io/repositories/10489" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10489" alt="2noise%2FChatTTS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# ChatTTS
Un modèle de parole génératif pour le dialogue quotidien.

[![Licence](https://img.shields.io/github/license/2noise/ChatTTS?style=for-the-badge)](https://github.com/2noise/ChatTTS/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ChatTTS.svg?style=for-the-badge&color=green)](https://pypi.org/project/ChatTTS)

[![Huggingface](https://img.shields.io/badge/🤗%20-Models-yellow.svg?style=for-the-badge)](https://huggingface.co/2Noise/ChatTTS)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/2noise/ChatTTS/blob/main/examples/ipynb/colab.ipynb)
[![Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Ud5Jxgx5yD)

[**English**](../../README.md) | [**简体中文**](../cn/README.md) | [**日本語**](../jp/README.md) | [**Русский**](../ru/README.md) | [**Español**](../es/README.md)| **Français**  | [**한국어**](../kr/README.md)

</div>

## Introduction
> [!Note]
> Ce dépôt contient l'infrastructure de l'algorithme et quelques exemples simples.

> [!Tip]
> Pour les produits finaux étendus pour les utilisateurs, veuillez consulter le dépôt index [Awesome-ChatTTS](https://github.com/libukai/Awesome-ChatTTS/tree/en) maintenu par la communauté.

ChatTTS est un modèle de synthèse vocale conçu spécifiquement pour les scénarios de dialogue tels que les assistants LLM.

### Langues prises en charge
- [x] Anglais
- [x] Chinois
- [ ] À venir...

### Points forts
> Vous pouvez vous référer à **[cette vidéo sur Bilibili](https://www.bilibili.com/video/BV1zn4y1o7iV)** pour une description détaillée.

1. **Synthèse vocale conversationnelle**: ChatTTS est optimisé pour les tâches basées sur le dialogue, permettant une synthèse vocale naturelle et expressive. Il prend en charge plusieurs locuteurs, facilitant les conversations interactives.
2. **Contrôle granulaire**: Le modèle peut prédire et contrôler des caractéristiques prosodiques fines, y compris le rire, les pauses et les interjections.
3. **Meilleure prosodie**: ChatTTS dépasse la plupart des modèles TTS open-source en termes de prosodie. Nous fournissons des modèles pré-entraînés pour soutenir la recherche et le développement.

### Dataset & Modèle
- Le modèle principal est entraîné avec des données audio en chinois et en anglais de plus de 100 000 heures.
- La version open-source sur **[HuggingFace](https://huggingface.co/2Noise/ChatTTS)** est un modèle pré-entraîné de 40 000 heures sans SFT.

### Roadmap
- [x] Open-source du modèle de base de 40k heures et du fichier spk_stats.
- [x] Génération audio en streaming.
- [ ] Open-source de la version 40k heures avec contrôle multi-émotions.
- [ ] ChatTTS.cpp (nouveau dépôt dans l'organisation `2noise` est bienvenu)

### Avertissement
> [!Important]
> Ce dépôt est à des fins académiques uniquement.

Il est destiné à un usage éducatif et de recherche, et ne doit pas être utilisé à des fins commerciales ou légales. Les auteurs ne garantissent pas l'exactitude, l'exhaustivité ou la fiabilité des informations. Les informations et les données utilisées dans ce dépôt sont à des fins académiques et de recherche uniquement. Les données obtenues à partir de sources accessibles au public, et les auteurs ne revendiquent aucun droit de propriété ou de copyright sur les données.

ChatTTS est un système de synthèse vocale puissant. Cependant, il est très important d'utiliser cette technologie de manière responsable et éthique. Pour limiter l'utilisation de ChatTTS, nous avons ajouté une petite quantité de bruit haute fréquence pendant l'entraînement du modèle de 40 000 heures et compressé la qualité audio autant que possible en utilisant le format MP3, pour empêcher les acteurs malveillants de l'utiliser potentiellement à des fins criminelles. En même temps, nous avons entraîné en interne un modèle de détection et prévoyons de l'open-source à l'avenir.

### Contact
> Les issues/PRs sur GitHub sont toujours les bienvenus.

#### Demandes formelles
Pour les demandes formelles concernant le modèle et la feuille de route, veuillez nous contacter à **open-source@2noise.com**.

#### Discussion en ligne
##### 1. Groupe QQ (application sociale chinoise)
- **Groupe 1**, 808364215 (Complet)
- **Groupe 2**, 230696694 (Complet)
- **Groupe 3**, 933639842 (Complet)
- **Groupe 4**, 608667975

##### 2. Serveur Discord
Rejoignez en cliquant [ici](https://discord.gg/Ud5Jxgx5yD).

## Pour commencer
### Cloner le dépôt
```bash
git clone https://github.com/2noise/ChatTTS
cd ChatTTS
```

### Installer les dépendances
#### 1. Installation directe
```bash
pip install --upgrade -r requirements.txt
```

#### 2. Installer depuis conda
```bash
conda create -n chattts
conda activate chattts
pip install -r requirements.txt
```

#### Optionnel : Installer TransformerEngine si vous utilisez un GPU NVIDIA (Linux uniquement)
> [!Note]
> Le processus d'installation est très lent.

> [!Warning]
> L'adaptation de TransformerEngine est actuellement en cours de développement et NE PEUT PAS fonctionner correctement pour le moment.
> Installez-le uniquement à des fins de développement.

```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

#### Optionnel : Installer FlashAttention-2 (principalement GPU NVIDIA)
> [!Note]
> Voir les appareils pris en charge dans la [documentation Hugging Face](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2).

> [!Warning]
> Actuellement, FlashAttention-2 ralentira la vitesse de génération selon [ce problème](https://github.com/huggingface/transformers/issues/26990). 
> Installez-le uniquement à des fins de développement.

```bash
pip install flash-attn --no-build-isolation
```

### Démarrage rapide
> Assurez-vous que vous êtes dans le répertoire racine du projet lorsque vous exécutez ces commandes ci-dessous.

#### 1. Lancer WebUI
```bash
python examples/web/webui.py
```

#### 2. Inférence par ligne de commande
> Cela enregistrera l'audio sous ‘./output_audio_n.mp3’

```bash
python examples/cmd/run.py "Votre premier texte." "Votre deuxième texte."
```

## Installation

1. Installer la version stable depuis PyPI
```bash
pip install ChatTTS
```

2. Installer la dernière version depuis GitHub
```bash
pip install git+https://github.com/2noise/ChatTTS
```

3. Installer depuis le répertoire local en mode développement
```bash
pip install -e .
```

### Utilisation de base

```python
import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False) # Définissez sur True pour de meilleures performances

texts = ["METTEZ VOTRE PREMIER TEXTE ICI", "METTEZ VOTRE DEUXIÈME TEXTE ICI"]

wavs = chat.infer(texts)

torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
```

### Utilisation avancée

```python
###################################
# Échantillonner un locuteur à partir d'une distribution gaussienne.

rand_spk = chat.sample_random_speaker()
print(rand_spk) # sauvegardez-le pour une récupération ultérieure du timbre

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # ajouter le locuteur échantillonné 
    temperature = .3,   # en utilisant une température personnalisée
    top_P = 0.7,        # top P décode
    top_K = 20,         # top K décode
)

###################################
# Pour le contrôle manuel au niveau des phrases.

# utilisez oral_(0-9), laugh_(0-2), break_(0-7) 
# pour générer un token spécial dans le texte à synthétiser.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

###################################
# Pour le contrôle manuel au niveau des mots.

text = 'Quel est [uv_break]votre plat anglais préféré?[laugh][lbreak]'
wavs = chat.infer(text, skip_refine_text=True, params_refine_text=params_refine_text,  params_infer_code=params_infer_code)
torchaudio.save("output2.wav", torch.from_numpy(wavs[0]), 24000)
```

<details open>
  <summary><h4>Exemple : auto-présentation</h4></summary>

```python
inputs_en = """
chat T T S est un modèle de synthèse vocale conçu pour les applications de dialogue.
[uv_break]il prend en charge les entrées en langues mixtes [uv_break]et offre des capacités multi-locuteurs
avec un contrôle précis des éléments prosodiques comme 
[uv_break]le rire[uv_break][laugh], [uv_break]les pauses, [uv_break]et l'intonation.
[uv_break]il délivre une parole naturelle et expressive,[uv_break]donc veuillez
[uv_break]utiliser le projet de manière responsable à vos risques et périls.[uv_break]
""".replace('\n', '') # L'anglais est encore expérimental.

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
)

audio_array_en = chat.infer(inputs_en, params_refine_text=params_refine_text)
torchaudio.save("output3.wav", torch.from_numpy(audio_array_en[0]), 24000)
```

<table>
<tr>
<td align="center">

**locuteur masculin**

</td>
<td align="center">

**locutrice féminine**

</td>
</tr>
<tr>
<td align="center">

[locuteur masculin](https://github.com/2noise/ChatTTS/assets/130631963/e0f51251-db7f-4d39-a0e9-3e095bb65de1)

</td>
<td align="center">

[locutrice féminine](https://github.com/2noise/ChatTTS/assets/130631963/f5dcdd01-1091-47c5-8241-c4f6aaaa8bbd)

</td>
</tr>
</table>


</details>

## FAQ

#### 1. De combien de VRAM ai-je besoin ? Quelle est la vitesse d'inférence ?
Pour un clip audio de 30 secondes, au moins 4 Go de mémoire GPU sont nécessaires. Pour le GPU 4090, il peut générer de l'audio correspondant à environ 7 tokens sémantiques par seconde. Le Facteur Temps Réel (RTF) est d'environ 0.3.

#### 2. La stabilité du modèle n'est pas suffisante, avec des problèmes tels que des locuteurs multiples ou une mauvaise qualité audio.
C'est un problème qui se produit généralement avec les modèles autoregressifs (pour bark et valle). Il est généralement difficile à éviter. On peut essayer plusieurs échantillons pour trouver un résultat approprié.

#### 3. En plus du rire, pouvons-nous contrôler autre chose ? Pouvons-nous contrôler d'autres émotions ?
Dans le modèle actuellement publié, les seules unités de contrôle au niveau des tokens sont `[laugh]`, `[uv_break]`, et `[lbreak]`. Dans les futures versions, nous pourrions open-source des modèles avec des capacités de contrôle émotionnel supplémentaires.

## Remerciements
- [bark](https://github.com/suno-ai/bark), [XTTSv2](https://github.com/coqui-ai/TTS) et [valle](https://arxiv.org/abs/2301.02111) démontrent un résultat TTS remarquable par un système de style autoregressif.
- [fish-speech](https://github.com/fishaudio/fish-speech) révèle la capacité de GVQ en tant que tokenizer audio pour la modélisation LLM.
- [vocos](https://github.com/gemelo-ai/vocos) qui est utilisé comme vocodeur pré-entraîné.

## Appréciation spéciale
- [wlu-audio lab](https://audio.westlake.edu.cn/) pour les expériences d'algorithme précoce.

## Merci à tous les contributeurs pour leurs efforts
[![contributors](https://contrib.rocks/image?repo=2noise/ChatTTS)](https://github.com/2noise/ChatTTS/graphs/contributors)

<div align="center">

  ![counter](https://counter.seku.su/cmoe?name=chattts&theme=mbs)

</div>
