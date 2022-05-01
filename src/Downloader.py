from os import makedirs, remove
from gdown import download
from zipfile import ZipFile


def unzip(zipfile, extract_path):
    with ZipFile(zipfile, "r") as zip_ref:
        zip_ref.extractall(extract_path)

def download_from_gdrive(name, link, path, type):

    url = f'https://drive.google.com/uc?id={link}'
    zipfile_name = f'{path+name}/temp.zip'
    extract_path = f'{path+name}/{type}'

    makedirs(path+name, exist_ok=True)
    download(url=url, output=zipfile_name, quiet=False)
    unzip(zipfile_name, extract_path)
    remove(zipfile_name)



if __name__ == '__main__':

    from argparse import ArgumentParser
    from Global.envs import ALBUM_NAMES as NAMES
    from Global.envs import ALBUM_LINKS as LINKS
    from Global.envs import ALBUM_TYPES as TYPES
    from Global.envs import ALBUM_PATH as PATH

    parser = ArgumentParser()
    parser_help = " ".join([f'[{i}] {name}' for i, name in enumerate(NAMES)])
    parser.add_argument("Album_ID", type=int, help=parser_help)
    args = parser.parse_args()
    album_id = args.Album_ID

    if album_id >= len(NAMES):
        print(f'Please enter a valid id between 0~{len(NAMES)-1}')
        exit()

    album_id = 0
    name = NAMES[album_id]
    link = LINKS[album_id]
    path = PATH
    type = TYPES[0]

    download_from_gdrive(name, link, path, type)


