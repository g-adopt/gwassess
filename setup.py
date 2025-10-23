from distutils.core import setup
import sys

script_args = sys.argv[1:]

setup(name='gwassess',
      version='1.0',
      author='Sia Ghelichkhan, Liam Morrow',
      author_email='siavash.ghelichkhani@anu.edu.au',
      description='gwassess is a python package that implements analytical and '
      'reference solutions to the Richards equation in 2D and 3D domains.',
      long_description=open('README.rst').read(),
      url='https://github.com/g-adopt/gwassess',
      packages=['gwassess'],
      install_requires=['numpy', 'scipy'],
      keywords=['Richards equation', 'analytical solutions',
                'groundwater flow', 'unsaturated flow', 'benchmarks'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: GNU Lesser General Public License v3 (LGPLv3)',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering'],
      script_args=script_args,
      ext_package='gwassess',
      )
