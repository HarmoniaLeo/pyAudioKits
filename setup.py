from setuptools import setup
 
setup(name='pyAudioKits',
      version="1.0.6",
      description='Powerful Python audio workflow support based on librosa and other libraries',
      author='HarmoniaLeo',
      author_email='harmonialeo@gmail.com',
      maintainer='HarmoniaLeo',
      maintainer_email='harmonialeo@gmail.com',
      packages=['pyAudioKits','pyAudioKits.audio','pyAudioKits.record','pyAudioKits.algorithm','pyAudioKits.filters','pyAudioKits.datastructures','pyAudioKits.analyse'],
      license="Public domain",
      platforms=["any"],
      url="https://github.com/HarmoniaLeo/pyAudioKits",
      install_requires=[
            'scipy',
            'playsound',
            'librosa',
            'numpy',
            'SoundFile',
            #'PyAudio',
            #'pygobject',
            'Wave',
            'matplotlib',
            'seaborn',
            'pandas',
            'python-speech-features',
            'hmmlearn',
            ]
     )