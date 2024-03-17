# Synthetic data generator - automated pipeline
Synthetic data generator used to create synthetic photos by taking data from a source photo based on the database stored in COCO and pasting it into the output photo.
## Algorithm
The program implements pipline architecture for creating new photos. The loaded data will be processed in individual steps to obtain a new photo with a randomly selected number of objects in it. Pasting new objects is done using mask processing, which results in a more natural final appearance. The program additionally generates a file with the segmentation of pasted objects in the [COCO](https://cocodataset.org) format. The individual parts of the process are illustrated in a diagram.
## Pipeline diagram
![Pipeline diagram](docs/Pipeline_architecture.svg)
## Preparing the environment
```bash
python3 -m venv venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
## Input arguments
* `coco` - path to input coco file, from which object will be gain,
* `library` - path to directory where photos from coco file are stored,
* `output` - path to output folder where photos and coco file will be saved,
* `prefix` - prefix for names for created photos - *e.g. `agpd_photo_` will generate photos named agpd_photo_0.jpg, agpd_photo_1.jpg ...*,
* `number` - number of photos to generate - *default = 100*.
## Program start
```bash
python3 main.py --coco /path/to/coco/file/annotations.json \
--library /path/to/library/images \
--input /path/to/backgrounds/directory \
--output /path/to/output/directory \
--prefix prefix_for_generated_photos_ \
--number 5
```
## Blending parameters
* Dilatation length = 050/255
* Gaussian blur kernel (smooth mask) = 075/555
* Threshold value =  130/255
* Max value with thresh binary = 255/255
* Blur length = 150/255
## GUI interface version
For fully manual version of the program with graphical user interface switch to branch [**main**](https://github.com/AgniechaP/synthetic_data_generation/tree/main). 
#### Agnieszka Piórkowska, Miłosz Gajewski
##### Politechnika Poznańska 2023