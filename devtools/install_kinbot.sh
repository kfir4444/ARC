# temporarily change directory to install software, and move one directory up in the tree
pushd ..

# clone the repo in the parent directory and update it
echo "Cloning/Updating KinBot..."
git clone https://github.com/zadorlab/KinBot
cd KinBot || exit
git fetch origin
git checkout master
git pull origin master

# Add to PYTHONPATH
echo "Adding KinBot to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "export PYTHONPATH=$$PYTHONPATH:$(pwd)" >> ~/.bashrc

# Restore the original directory
echo "Done installing Kinbot."
popd || exit
