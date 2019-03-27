import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?
USE_PLACE365_PRETRAINED = True

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'vcr1images')
print(VCR_IMAGES_DIR)
VCR_ANNOTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")