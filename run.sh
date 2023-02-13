cp Cluster.py Cluster.pyx
cp Experiment.py Experiment.pyx
cp Particle.py Particle.pyx
cp hypo.py hypo.pyx
cp RetentionProbabilities.py RetentionProbabilities.pyx
cp TrajectoryDisplacementGenerator.py TrajectoryDisplacementGenerator.pyx
cp utils.py utils.pyx

cythonbuilder build

echo "Running code..."
python $1

rm *.pyx *.so *.pyi
rm -r ./ext
