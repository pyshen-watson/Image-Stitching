# Environment
- Language: python 3.7.5
- Modules: 
    - OpenCV
    - NumPy
    - Matplotlab
    - Numba
    - gdown
    - SciPy
    - tqdm


We use pipenv to manage our package. If your don't have pipenv , run this: `$ pip install pipenv`

# Install the dependency
`$ pipenv install`

# Run the code
`$ pipenv run python src/main.py [-d id] [-p path] [-n]`

There are three arguments:
- `-d`: Use the images download from cloud, use id `0`~`2` to specify the image set
-  `-p`: Follow the path of image set in local path. Note that you cannot use `-d` and `-p` at the same time.
- `-n`: If the images didn't be shotted clockwise, use the argument to reorder the images. Note that this option may massively increase the execute time.
