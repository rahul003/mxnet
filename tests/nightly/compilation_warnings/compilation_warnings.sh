
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
sudo apt-get install time g++-5
runme make clean >/dev/null
runme mkdir build
echo "Starting make-default"
runme /usr/bin/time -f "%e" make -j$(nproc) &> build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py build/compile_output.txt

export CC=gcc-5
export CXX=g++-5

runme make clean >/dev/null
runme mkdir build
echo "Starting make-5"
runme /usr/bin/time -f "%e" make -j$(nproc) &> build/compile_output.txt
echo "Finished make. Now processing output"
python tests/nightly/compilation_warnings/process_output.py build/compile_output.txt
