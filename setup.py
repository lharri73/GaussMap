import setuptools
import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    description = f.read()

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build extension")
        
        cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
        if cmake_version < '3.17.0':
            raise RuntimeError("CMake >=3.17 required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # cmake_args += ['-DCMAKE_BUILD_TYPE='+cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS',''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        print("CMake Compilation Complete...")

setup(
    name='gaussMap',
    version='0.2',
    author='Landon Harris',
    author_email='lharri73@vols.utk.edu',
    description='A Cuda implementation of gaussian heat maps',
    long_description=description,
    ext_modules=[CMakeExtension('gaussMap')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=requirements,
    dependency_links=['git+https://github.com/lharri73/nuscenes_dataset_private.git#egg=nuscenes_dataset'],
    python_requires=">=3.7",
)
