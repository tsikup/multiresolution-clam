import os

SCALE_FACTOR = 32
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = 'jpg'
IMAGE_EXT = 'png'
SRC_TRAIN_EXT = "svs"
DEST_TRAIN_EXT = IMAGE_EXT
IMAGE_PREFIX = 'TCGA-'
TRAIN_PREFIX = IMAGE_PREFIX

PROJECT = 'TCGA-BRCA'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')
HTML_DIR = os.path.join(OUTPUT_DIR, 'html')
DATA_DIR = os.path.join(ROOT_DIR, 'data', PROJECT)
SRC_TRAIN_DIR = os.path.join(DATA_DIR, 'Imaging', 'Training')

FILTER_DIR = os.path.join(OUTPUT_DIR, 'unsorted_filtered_tiles')
ALL_FILTERS_DIR = FILTER_DIR
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, 'unsorted')
THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, 'unsorted_thumbnails')
FILTER_HTML_DIR = os.path.join(HTML_DIR, 'filters')

STATS_DIR = os.path.join(OUTPUT_DIR, 'training_slide_stats')
DEST_TRAIN_DIR = os.path.join(OUTPUT_DIR, 'training_' + DEST_TRAIN_EXT)
DEST_TRAIN_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, 'training_thumbnail_' + THUMBNAIL_EXT)
DEST_TRAIN_FILTER_DIR = os.path.join(OUTPUT_DIR, 'training_filter_' + DEST_TRAIN_EXT)
DEST_TRAIN_FILTER_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, 'training_filter_thumbnail_' + THUMBNAIL_EXT)

TILE_SUMMARY_DIR = os.path.join(OUTPUT_DIR, 'tile_summary')
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)

TILE_DIR = os.path.join(OUTPUT_DIR, "tiles_" + DEST_TRAIN_EXT)
TILE_SUFFIX = "tile"
TILE_DATA_DIR = os.path.join(OUTPUT_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"

TOP_TILES_SUFFIX = "top_tile_summary"
TOP_TILES_DIR = os.path.join(OUTPUT_DIR, TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
TOP_TILES_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(OUTPUT_DIR, TOP_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(OUTPUT_DIR,
                                                   TOP_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
NUM_TOP_TILES = 50

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = os.path.join(ASSETS_DIR, "fonts/Arial-Bold.ttf")
SUMMARY_TITLE_FONT_PATH = os.path.join(ASSETS_DIR, "fonts/Courier-New-Bold.ttf")
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330

TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_PAGINATION_SIZE = 50

FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = False
FILTER_RESULT_TEXT = 'filtered'
TILE_SUMMARY_SUFFIX = 'tile_summary'

DEBUG = False


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


seg_labels = dict(Background=-1, Tissue=0, Tumor=1)
