
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

sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get -y install time g++-5
runme make clean >/dev/null
runme mkdir build
echo "Starting make"
runme CC="g++-5" CXX="g++-5" /usr/bin/time -f "%e" make -j$(nproc) &> build/compile_output.txt
head -10 build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py build/compile_output.txt
