# sat-keypoint-matching
OpenCV Benchmark for keypoint matching of templates in sattelite imagery 


## Setup

### Install OpenCV Contrib

```bash
pip install opencv-contrib-python
```
## Usage

```bash
python tempmatchbenchmark.py -t big.tif -q small.png -o output/

options:
  -h, --help         show this help message and exit
  -t TEMPLATE_NAME   Template image
  -q QUERY_NAME      Query image
  -o OUTPUT_PATH     Output directory
  -v                 Increase output verbosity
  -p                 Removes shear from affine2d transform, Use only if the image is scanned
  --matches          Shows the matching result and the good matches
  --rgb              Extract multichannel features in RGB space instead of grayscale
  --tres TRESHOLD    Minimum good matches to pass the validation test
  --grid PATCH_SIZE  Width in pixels of the window in the detection grid
  --pts MAX_PTS      Maximum number of detection points in a single patch
  --nms NMS_PTS      Maximum number of Non Max Suppression points across the whole image