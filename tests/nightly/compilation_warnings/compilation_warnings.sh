
runme() {
	cmd=$*
	echo "$cmd"
	$cmd
	ret=$?
	if [[ ${ret} != 0 ]]; then
		echo " "
		echo "ERROR: Return value non-zero for: $cmd"
		echo " "
		exit 1
	fi
}
g++ -dumpversion
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install time g++-5
sudo update-alternatives --remove-all gcc 
sudo update-alternatives --remove-all g++
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
g++ -dumpversion

runme make clean >/dev/null
runme mkdir build
echo "Starting make"
runme /usr/bin/time -f "%e" make -j$(nproc) &> build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py build/compile_output.txt
