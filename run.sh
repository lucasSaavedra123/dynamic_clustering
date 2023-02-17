mv Cluster.py Cluster.pyx
mv Experiment.py Experiment.pyx
mv Particle.py Particle.pyx
mv hypo.py hypo.pyx
mv RetentionProbabilities.py RetentionProbabilities.pyx
mv TrajectoryDisplacementGenerator.py TrajectoryDisplacementGenerator.pyx
mv utils.py utils.pyx

cythonbuilder build

echo "Running code"
python $1

mv Cluster.pyx Cluster.py
mv Experiment.pyx Experiment.py
mv Particle.pyx Particle.py
mv hypo.pyx hypo.py
mv RetentionProbabilities.pyx RetentionProbabilities.py
mv TrajectoryDisplacementGenerator.pyx TrajectoryDisplacementGenerator.py
mv utils.pyx utils.py

rm *.so *.pyi
rm -r ./ext
