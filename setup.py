from setuptools import setup, find_packages
setup(name='chattts',
      version='0.0.1',
      author='2noise',
      url='https://github.com/2noise/ChatTTS',
      install_requires=['omegaconf>=2.3.0',
                'torch>=2.1.0',
                'tqdm',
                'einops',
                'vector_quantize_pytorch',
                'transformers>=4.41.1',
                'vocos',
                'IPython',
                ],  # 定义依赖哪些模块
      packages=find_packages(),  # 系统自动从当前目录开始找包
      )
