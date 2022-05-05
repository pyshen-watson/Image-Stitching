from Global.envs import ALBUM_PATH  as PATH
from Global.envs import ALBUM_NAMES as NAMES
from Global.envs import ALBUM_LINKS as LINKS
from Global.envs import ALBUM_TYPES as TYPES
from Global.utils import checkSlash, print_exit
from Downloader import download_all
from Reprojector import reproject_all
from Detector import detect_all
from Descriptor import describe_all
from Matcher import match_all
from Stiticher import stitch_all
from Optimizer import optimize


def set_parser():

    from argparse import ArgumentParser
    parser = ArgumentParser(prog='main.py', description='Stitch the images')

    download_help = "Download the images online: " + " ".join([f'[{i}] {name}' for i, name in enumerate(NAMES)])
    parser.add_argument('--download', '-d',type=int, required=False, help=download_help)

    path_help = "The path to the directory contents images"
    parser.add_argument('--path', '-p', type=str, default='', required=False, help=path_help)

    ordered_help = "If the images have been ordered"
    parser.add_argument('--nonordered', '-n', action='store_true', help=ordered_help)
    parser.set_defaults(nonordered=False)

    return parser.parse_args()

if __name__ == '__main__':

    args = set_parser()
    download_id, image_path = args.download, args.path
    hasOrder = not args.nonordered


    if len(image_path) > 0:

        if download_id:
            print_exit("Please use only one argument.")

        image_path = checkSlash(image_path)
        name = image_path.split('/')[-1]
        dirs = [ f'{image_path}{t}/'for t in TYPES[1:]]

        reproject_all(image_path, dirs[0])
        detect_all(dirs[0], dirs[1])
        describe_all(dirs[0], dirs[1], dirs[2])
        match_all(dirs[0], dirs[2], dirs[3], hasOrder)
        stitch_all(dirs[0], dirs[3], dirs[4])
        optimize(dirs[4], dirs[5])

    else:
        if not download_id:
            print_exit("Please use at least one argument.")
        if download_id < 0 or download_id >= len(NAMES):
            print_exit(f"Please use a valid id for 0 to {len(NAMES)-1} ")

        path = PATH
        name = NAMES[download_id]
        link = LINKS[download_id]
        type = TYPES[0]
        dirs = [ f'{PATH}/{name}/{t}/'for t in TYPES]

        download_all(link, dirs[0])
        reproject_all(dirs[0], dirs[1])
        detect_all(dirs[1], dirs[2])
        describe_all(dirs[1], dirs[2], dirs[3])
        match_all(dirs[1], dirs[3], dirs[4], hasOrder)
        stitch_all(dirs[1], dirs[4], dirs[5])
        optimize(dirs[5], dirs[6])
