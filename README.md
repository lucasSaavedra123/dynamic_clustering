# Decoding dynamic membrane protein aggregation with supervised Graph-based deep learning

In this repository, all software related with REFERENCE_TO_PAPER (including simulation software) can be found.

## Installation

During the whole development of the project, we used different machines with different operating systems. Also, some of these have modern GPUs to train. However, some users may not have special hardware and not be allow to run the code. We provide different YML files to run the software on different machines, depending on the available hardware and software. To install this repository, you will need to have Conda installed. YML files descriptions are below:

<ul>
  <li><code>environment-cpu.yml</code>: Install with this file if you don't have GPUs.</li>
  <li><code>environment-titan.yml</code>:Install with this file if you use a GPU (this YML worked with TITAN V).</li>
  <li><code>environment-m1-2021.yml</code>:Install with this file if you use a Mac M1 2021 which includes a powerful GPU for Deep Learning.</li>
</ul>

Run the following command:

``` 
conda env create -n dynamic_clustering --file SELECTED_YML_FILE
```


Then, activate the created environment:


``` 
conda activate dynamic_clustering
```

## How to simulate

In order to reproduce datasets, please see generate_datasets.py.
```generate_datasets.py```. This script works as an example and it is the script that we used to generated datasets in our work.

We strongly recommend to run the simulations with cythonbuilder. According to our experience, simulations are faster. Run the following commands to cythonize the code:

``` 
mv Cluster.py Cluster.pyx
mv Experiment.py Experiment.pyx
mv Particle.py Particle.pyx
mv hypo.py hypo.pyx
mv RetentionProbabilities.py RetentionProbabilities.pyx
mv TrajectoryDisplacementGenerator.py TrajectoryDisplacementGenerator.pyx
mv utils.py utils.pyx

cythonbuilder build
```

If you want to recover the original files, run the following commands:

```
mv Cluster.pyx Cluster.py
mv Experiment.pyx Experiment.py
mv Particle.pyx Particle.py
mv hypo.pyx hypo.py
mv RetentionProbabilities.pyx RetentionProbabilities.py
mv TrajectoryDisplacementGenerator.pyx TrajectoryDisplacementGenerator.py
mv utils.pyx utils.py

rm *.so *.pyi
rm -r ./ext
```

## Docker Image for Dynamic Simulations

The proyect was executed on Windows to avoid Linux virtualizations. However, we noticed that these simulation run faster on Linux systems. We used [Docker](https://www.docker.com/) to run simulations and we used the following code to run it:

```
docker build -t dynamic_simulations .
docker run -i -v dynamic_simulation_files:/usr/src/app/datasets --name simulation_container -t dynamic_simulations bash
```

Once the shell is open, follow the instructions of the first part.

Then, to retrieve .csv files, run the following command:
```
docker cp simulation_container:/usr/src/app/datasets/ ./
```
