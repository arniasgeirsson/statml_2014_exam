# --------------------------------------------------------------------------- #
"Configuration file used to determine some simple configurations"
# --------------------------------------------------------------------------- #

# This value determines whether or not the files should recompute all the
# values. These are all the Jaakkola values, all the values found through
# gridsearch, etc.
recompute = False

# This parameter corresponds to the n_jobs parameter used in many of the
# scikit learn methods, and denotes how many threads the method can use to
# perform the respected method, although does not work on windows, and is
# therefore set to 1 (so it has no effect) in the handin.
# Feel free to change to speed up the computations if it works on your machine.
n_jobs = 1

# This parameter corresponds to the respective parameter used in many of the
# functions of scikit learn. It determines how much is printing when running
# the scikit learn functions.
verbose = 0

# This parameter corresponds to the respective parameter used when cross
# validating in scikit learn. It determines the level of the fold of the cross
# validation.
cv = 5

# This value determines if the code should show the generated plots upon run.
showfigs=True

# This value determines if the code should save the generated plots upon run.
savefigs=False

# The following values define the paths where the different plots should be
# saved to.
# Any folder specified must exist before running, i.e. they are not created
# automatically.
fig2_4_1_esPath='../Report/Figures/2_4_1-combined_eigen.png'
fig2_4_2_scPath='../Report/Figures/2_4_2-scatter.png'

fig2_5_1_scPath='../Report/Figures/2_5_1-scatterandcenter.png'
