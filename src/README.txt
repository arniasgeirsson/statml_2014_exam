-------------------------------------------------------------------------------

I have used Python 2.7.6 to implement my solutions to the exam questions.
I have tested the implementations using Python 2.7.6 64-bit version on
Windows7, and Python 2.7.6 on OSX Mavericks. I cannot guarantee that my
implementations work using any other than the mentioned combinations of
Python versions and operating systems.


---- Dependencies

Furthermore I have used the following Python packages in my implementations

- Scipy v 0.13.3
 - matplotlib [2]
 - numpy [3]
 - python-dateutil [4]
 - pyparsing [5]
 - six [6]
- Scikit-learn v 0.14.1

It is recommended to install everything at once the ScipySuperpack [1]
Once again I cannot guarantee that my implementations are backwards compatible
(or forward compatible) with the described packages.


---- Files

This hand-in contains a number of folders and files.

'data/' contains all the given data files and are untouched, but are included
for the sake of convenience.

'modules/common.py' is a file which contains a set of commonly used functions
throughout my implementations.

'modules/config.py' is a small configurations file which is described below.

'main.py' is the main file to run all the code and is further described below.

The following files are the individual files implementing my solution to each
exam question. These can also be run individually.
'question1.py'
'question2.py'
'question3.py'
'question4.py'
'question5.py'
'question7.py'


---- Main

Once Python and the dependencies have been installed my implementations can be
run with Python.

As stated I have included a main file 'main.py' which can be run to reproduce
all the results shown in my report.

Note that with my levels of skill using Python and matplotlib I was not able
to make sure that the generated plots stay when running multiple files
(through the main file).
Hence I refer the reader to run these files on their own, if the reader
wishes to see the plots (without them closing once they open).
The files in question are 'question4.py' & 'question5.py'.


---- Configurations

My solution comes with a small configurations file 'modules/config.py'
which allows you to make small changes.

The most interesting configurations are the 'recompute', 'showfigs' and
'savefigs'.
If 'recompute' is set to False then the files affected use the hardcoded
optimized hyperparameters, the ones presented in the report, when doing
grid search. If the value is set to True then all the defined possible
hyperparameters are used in the grid search and hence takes much longer time to
run.

On my fastest machine it takes about 30-40 seconds to run the main file with
'recompute' set to False, and about 16-18 minutes to run when set to True.

The 'showfigs' and 'savefigs' values determine whether or not the generated
plots should be printed to the screen and saved to some files, respectively.

The configurations file also defines the file paths for the generated plots,
should they be saved to the filesystem, and I suggest the reader to make sure
that the defined paths are what he or she desires if he or she wants to save
the plots.


---- Notes

I am aware that the file 'questions7.py' raises an warning, although this
warning is raised due to the fact that at least one class in the Variabel
Stars data set that only has one data point. I decided not to do something
about the warning to remind the one running the code of this fact and that one
should considered the impact it has on the classifiers.


-------------------------------------------------------------------------------
[1]: http://fonnesbeck.github.io/ScipySuperpack/
[2] http://matplotlib.org/
[3] http://www.numpy.org/
[4] http://labix.org/python-dateutil
[5] http://pyparsing.wikispaces.com/
[6] https://pypi.python.org/pypi/six

