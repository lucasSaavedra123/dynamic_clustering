# Live-cell clustering analysis for multi-level characterization of membrane protein aggregation

In this repository, simulation of Dynamic SMLM datasets can be found. In order to reproduce datasets, please see generate_datasets.py.
```generate_datasets.py```.

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

The proyect was executed on Windows to avoid Linux virtualizations. However, we noticed that these simulation run faster on Linux systems. We used Docker to run simulations and we used the following code to run it:

```
docker build -t dynamic_simulations .
docker run -i -v dynamic_simulation_files:/usr/src/app/datasets --name simulation_container -t dynamic_simulations bash
```

Once the shell is open, follow the instructions of the first part.

Then, to retrieve .csv files, run the following command:
```
docker cp simulation_container:/usr/src/app/datasets/* ./
```
