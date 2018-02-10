# Bypass packages from external environment to conda environment
packages=('tensorflow')
for pack in "${packages[@]}"
do
	src="/home/pythonAdmin/local/python/lib/python2.7/site-packages/$pack"
	dest="$CONDA_PREFIX/lib/python2.7/site-packages/"
	echo "$src adding symbolic link to $dest"
	ln -s $src $dest
done
