from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
LIBRARY_NAME = 'vwsd'
REPOSITORY_NAME = 'vwsd_experiment'
VERSION = '0.0.0'
setup(
    name=LIBRARY_NAME,
    packages=find_packages(exclude=['tests']),
    version=VERSION,
    # license='TBA',
    description="TBA",
    url=f'https://github.com/asahi417/{REPOSITORY_NAME}',
    keywords=['machine learning', 'computer vision', 'natural language processing'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        # 'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
      ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "pandas",
        "torch",
        "transformers",
        "Pillow",
        "numpy",
        "torchvision"
        "matplotlib",
        "ranx"
        # "ftfy",
        # "regex",
        # "tqdm",
        # "omegaconf>=2.0.0",
        # "pytorch-lightning>=1.0.8",
        # "torch-fidelity",
        # "einops",
        # "glide_text2im @ git+https://github.com/openai/glide-text2im@main",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            # 'harrogate-image-generation = harrogate_cl.image_generation:main',
            # 'harrogate-image-sr = harrogate_cl.image_super_resolution:main',
            # 'harrogate-canvas = harrogate_cl.canvas:main'
        ]
    }
)

