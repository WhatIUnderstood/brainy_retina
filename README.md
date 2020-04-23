# README #

Retina simulation from digital camera source. The following outputs can be generated from one image:
- S,M,L cones output.
- Parvo output by simulating the midget cells outputs
- Magno output by simulating the parasol cells. Temporal aspects are not yet simulated

This project generate a library that use GPU acceleration by using cuda. No pure CPU support is done for now.

If you want more details, check this [what I understood page](https://blog.whatiunderstood.com/pages/artificial_intelligence/sensors/retina/retina_modelisation.html)

### Build ###
Requirements:
 - docker
 - nvidia card with a driver version >= 430 (I did not try below). You do not need cuda to be installed as we compile in a docker

 To check if you match the requirement try to run the following command:
`sudo docker run --rm --runtime=nvidia nvidia/cuda:10.0-devel nvcc --version`

This project can be build in a docker environment. To start it run
`./scripts/start_dev.sh`

```
mkdir build
cd build
cmake ..
make
```

### Run the example ###
An example is provided
`./gpuretina_test -i ../videos/<some_video>.mp4`

