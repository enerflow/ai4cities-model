from setuptools import setup, find_packages

setup(
    name='rebase_ai4cities_models',
    url='https://github.com/rebaseenergy/ai4cities-project',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=['numpy', 'pyomo==5.7.3', 'numpy-financial==1.0.0'],
    include_package_data=True,
    version='0.0.3',
    license='',
    description='',
    long_description=open('README.md').read(),
)
