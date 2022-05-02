from os import getcwd
ALBUM_PATH = getcwd().replace('/src','/')
ALBUM_NAMES = [
    'Balcony',
    'Lake',
]
ALBUM_LINKS = [
    '1iNbX_1-1XHSTabKGYwV6m9MWilC1XCfs',
    '1XL0kYusA5Tr6GfcQ22kxsIWE8vjWtYlo',
]
ALBUM_TYPES = [
    '0_Original',
    '1_Reprojected',
    '2_Keypoint',
    '3_Desciption'
]

REPROJECTION_FOCAL_LENGTH = 3450
REPROJECTION_RESIZE_SCALE = 0.5

DETECT_BLUR_KSIZE = 11
DETECT_BLUR_SIGMA = 5
DETECT_GRAD_KSIZE = 5
DETECT_SLIDE_KSIZE = 11
DETECT_SLIDE_SIGMA = 5
DETECT_K = 0.05
DETECT_R_THRESHOLD = 1
DETECT_ANMS_RADIUS = 36
DETECT_ANMS_NUMBER = 500

DESCRIBE_SLIDE_KSIZE = DETECT_SLIDE_KSIZE
DESCRIBE_GRAD_KSIZE = DETECT_GRAD_KSIZE
DESCRIBE_BUCKET_NUMBER = 36
DESCRIBE_BLUR_KSIZE = DESCRIBE_SLIDE_KSIZE
DESCRIBE_BLUR_SIGMA = DETECT_BLUR_SIGMA
DESCRIBE_PATCH_KSIZE = 40
DESCRIBE_PATCH_SIGMA = 4.5
DESCRIBE_PATCH_SMPSIZE = 8
