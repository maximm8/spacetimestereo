from setuptools import setup

setup(
   name='spacetimestereo',
   version='0.1',
   description='space time stereo algorithm',
   author='maximm8',
   author_email='foomail@foo.example',
   packages=['spacetimestereo'],  #same as name
   install_requires=['numpy', 'matplotlib', 'open3d', 'opencv-python', 'screeninfo', 'tqdm'], #external packages as dependencies
)