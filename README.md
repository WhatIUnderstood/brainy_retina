# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

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

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact