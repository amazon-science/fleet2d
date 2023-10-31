# Fleet2D

Fleet2D is a simulation framework for testing long time-horizon algorithms.

## Running locally
To run Fleet2D use the following steps.

### Set up
To install Python dependencies locally run the following command:
```
pip3 install -r requirements.txt
```
### Prerequisites

Fleet2D depends on an OpenCV installation that supports both Python and C++ development. The easiest way to achieve this is usually by installing the Python OpenCV libraries (via the `requirements.txt` installation above. And then installing the same corresponding OpenCV version using your system's package manager.

For example, currently the requirement is `opencv-python~=4.6.0`, then using your distribution's package manager (e.g. `apt`, `yum`, `brew`, etc) specify version `4.6.0` of OpenCV's C++ development libraries.

Another method might be to simply build and install OpenCV from source. We've found this to be the most straightforward on MacOS.

### Build
To build locally run the following command:
```
python3 setup.py build
```
Once the command has completed compiling any C++ files that have been updated, a build directory is created. Usually this is the `build` directory in the package's root. Navigate to this directory with the following command:
```
cd build/lib\*
```
Once there, export the path for Python to be able to recognize the freshly built library:
```
export PYTHONPATH=`pwd`
```
Then go back to your original workspace directory (`cd -` should work). You're now ready to run the test bed!

### Run simulation
To run the simulation using the config file located at `src/f2d/configs/config_simple.yaml` enter the following command from the package home directory:
```
python3 src/f2d/simulation/simulation.py src/f2d/configs/config_simple.yaml
```

### Development iteration
Once set up, a typical build/run command after each code change might look like
```
python3 setup.py build && python3 src/f2d/simulation/simulation.py src/f2d/configs/config_simple.yaml
```
Or, the command could be set up in PyCharm by editing the runtime configuration and adding the appropriate path/command there.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
