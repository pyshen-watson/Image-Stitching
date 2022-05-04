from os import makedirs, remove
from gdown import download
from zipfile import ZipFile


def unzip(zipfile: str, extract_path: str) -> None:

    with ZipFile(zipfile, "r") as zip_ref:
        zip_ref.extractall(extract_path)

def download_all(link:str, out_dir:str) -> None:

    makedirs(out_dir, exist_ok=True)

    url = f'https://drive.google.com/uc?id={link}'
    zipfile_name = f'{out_dir}temp.zip'

    download(url=url, output=zipfile_name, quiet=False)
    unzip(zipfile_name, out_dir)
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

    link = LINKS[album_id]
    out_dir = f'{PATH}{NAMES[album_id]}/{TYPES[0]}'

    download_all(link, out_dir)


