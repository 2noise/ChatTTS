from setuptools import setup, find_packages
setup(name='chattts',
      version='0.1.0',
      author='2noise',
      url='https://github.com/2noise/ChatTTS',
      package_data={
        'ChatTTS.res': ['homophones_map.json'],  # 指定路径和文件
      },
      install_requires=['omegaconf>=2.3.0',
                        'numpy<2.0.0',
                        'numba',
                        'pybase16384',
                'torch>=2.1.0',
                'tqdm',
                'vector_quantize_pytorch',
                'transformers>=4.41.1',
                'vocos',
                ],  # 定义依赖哪些模块
      packages=find_packages(),  # 系统自动从当前目录开始找包
      )
