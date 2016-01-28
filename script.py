import numpy as np
import math
import argparse
import os.path

print("Start")

##############################################################
# Keywords to be parsed
##############################################################
#parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d, epilog=" ")
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=" ")
#parser.add_argument("-bsf", help="biasfactor used for the well-tempered target distribution")
# Temperature
parser.add_argument("-tempKelvin", type=float, default=-1., help="Temperature in Kelvin")
parser.add_argument("-tempkbt", type=float, default=-1.,help="Temperature in kbt")
# Input files
parser.add_argument("-colvar", default="COLVAR", help="filename containing reweighting CVs and static bias")
parser.add_argument("-numberOfFiles", type=int, default=1, help="number of files from which to reweight")
parser.add_argument("-maxColvarSize", type=int, default=10000000, help="max lines in individual colvar files ")
parser.add_argument("-rewcol", type=int, default=2, help="column in colvar file containing the CV to be reweighted (first column = 1) (default: %(default)s)")
parser.add_argument("-biascol", type=int, default=1, help="column in colvar file containing the variational bias (first column = 1) (default: %(default)s)")
parser.add_argument("-readStride", type=int, default=1, help="stride for reading the input file (default: %(default)s)")
parser.add_argument("-ignore", type=int, default=0, help="ignore these many lines in colvar (default: %(default)s)")
# CV parameters
parser.add_argument("-min", type=float,  help="minimum values of the new CV in colvar file")
parser.add_argument("-max", type=float,  help="maximum values of the new CV in colvar file")
parser.add_argument("-step", type=float, default=1., help="number of bins for the reweighted FES (default: %(default)s)")
# Output files
parser.add_argument("-outfile", default="fes_rew.dat", help="output FES filename (default: %(default)s)")
parser.add_argument("-outStride", type=int, default=0, help="stride in the output of the fes (default: %(default)s)")
# Error calculation
parser.add_argument("-bootstrapSamples", type=int, default=100, help="number of samples in the estimation of the error through bootstrapping (default: %(default)s)")
parser.add_argument("-significanceLevel", type=float, default=0.05, help="significance level in the bootstrap confidence interval (default: %(default)s)")
# Use weights
parser.add_argument('-weightsFlag', action='store_true')
# Test error convergence
parser.add_argument('-errorConvergence', action='store_true')
parser.add_argument("-errorConvergenceBin", type=int, default=1, help="calculate error convergence for this bin (default: %(default)s)")
parser.add_argument("-errorConvergenceMaxSample", type=int, default=100, help="max sample for which to calculate the error convergence (default: %(default)s)")
parser.add_argument("-errorConvergenceSampleStride", type=int, default=10, help="sample stride in the calculation of the error convergence (default: %(default)s)")
#parser.add_argument("-v", "--verbose", action='store_true', help="be verbose")
##############################################################


##############################################################
# Error classes
##############################################################
class InputError(Exception):
    """
    Raised when a input-related error happens.
    """
    pass
##############################################################

##############################################################
# Parsing inputs
##############################################################

print("Parsing...")

args = parser.parse_args()

# Input files

numberOfFiles=args.numberOfFiles
if (numberOfFiles<1):
	raise InputError("The number of input files should be greater than one.")
colvarFileName=args.colvar
if (numberOfFiles==1):
	if (not(os.path.isfile(colvarFileName))):
		raise InputError("The input file " + colvarFileName + " was not found.")
else:
	for i in xrange(numberOfFiles):
		if (not(os.path.isfile( colvarFileName + "." + str(int(i)) )) ):
			raise InputError("The input file " + colvarFileName + " was not found.")
maxColvarSize=args.maxColvarSize
if (maxColvarSize<0):
	raise InputError("The maximum colvar size must be greater than zero.")
# Parameters for the biased CV. 
CVmin=args.min
CVmax=args.max
step=args.step
# Other parameters
tempKelvin=args.tempKelvin
tempkbt=args.tempkbt
if (tempKelvin<0):
	temp=tempkbt
elif (tempkbt<0):
	kb=0.0083144621
	temp=kb*tempKelvin
else:
	raise InputError("You must define one and only one positive temperature either in Kelvin or KbT.")
print("The temperature is set to " + str(float(temp)) + " KbT.")
# Parameters for calculation of new FES
newCVcolumn=args.rewcol-1
biasColumn=args.biascol-1

# Stride
outStride=args.outStride
readStride=args.readStride
ignore=args.ignore

# Output
outputFesFileName=args.outfile

# Error calculation
bootstrapSamples=args.bootstrapSamples
if (bootstrapSamples<1):
	raise InputError("The number of bootstrap samples must be greater or equal to one")
significanceLevel=args.significanceLevel
if ( (significanceLevel<0.) or (significanceLevel>1.) ):
	raise InputError("The significance level must be in the interval [0,1]")

# Use weights
weightsFlag=args.weightsFlag
if (weightsFlag):
	print("Using weights based on the bias column")

# Error convergence
errorConvergence=args.errorConvergence
errorConvergenceBin=args.errorConvergenceBin         
errorConvergenceMaxSample=args.errorConvergenceMaxSample   
errorConvergenceSampleStride=args.errorConvergenceSampleStride

if (errorConvergence):
	print("Calculating error convergence for bin " + str(errorConvergenceBin) )
	print("Error convergence maximum sample size: " + str(errorConvergenceMaxSample) )
	print("Error convergence sample stride: " + str(errorConvergenceSampleStride)  )

print("Finished parsing")
##############################################################


##############################################################
# Define bootstrap function
##############################################################
# Based on http://people.duke.edu/~ccc14/pcfb/analysis.html
def bootstrap(data, num_samples, alpha, data_weights=None):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for a weighted average."""
    n = len(data)
    stat=np.zeros(num_samples)
    for i in xrange(num_samples):
        idx = np.random.randint(0, n, n)
        samples = data[idx]
	if (weightsFlag):
        	weights = data_weights[idx]
        	stat[i]=np.average(samples, 0, weights) 
	else:
        	stat[i]=np.mean(samples, 0) 
    stat = np.sort(stat)
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])
##############################################################

print("Initializing...")

########################################################
# General initialization
########################################################
# ignore warnings about log(0) and /0
np.seterr(all='ignore')
# Intialize files
#colvar1=np.genfromtxt(colvarFileName+".0")
#colvar2=np.genfromtxt(colvarFileName+".1")
#colvar3=np.genfromtxt(colvarFileName+".2")
#colvar4=np.genfromtxt(colvarFileName+".3")
# Size of files
#minSizeColvar=min(colvar1.shape[0],colvar2.shape[0],colvar3.shape[0],colvar4.shape[0])
#maxSizeColvar=max(colvar1.shape[0],colvar2.shape[0],colvar3.shape[0],colvar4.shape[0])
# Other parameters
beta=1./temp
# Denominator of histogram
#denom=0. 
# Parameters for the calculation of the new FES
Nbin=int((CVmax-CVmin)/step)+1
CVvector=np.linspace(CVmin,CVmax,Nbin)
# Matrix for bootstraping
samples=np.zeros([numberOfFiles*(maxColvarSize-ignore)/readStride,Nbin])
# Define arrays 
if (not(errorConvergence)):
	# If the histogram with errors is calculated
	bootLow=np.zeros([Nbin])
	bootHigh=np.zeros([Nbin])
	histogram=np.zeros([Nbin])	
else:
	# If the convergence of the error of one bin is calculated
	numSamples=int( (errorConvergenceMaxSample-errorConvergenceSampleStride)/errorConvergenceSampleStride )
	bootLow=np.zeros([numSamples])
	bootHigh=np.zeros([numSamples])
	sampleSize=np.zeros([numSamples])
	histogram=np.zeros([numSamples])	

if (weightsFlag):
	weights=np.zeros([numberOfFiles*(maxColvarSize-ignore)/readStride])
########################################################

print("Initialization completed")

########################################################
# Start calculation
########################################################

print("Starting calculation...")

counts=0

for i in range(numberOfFiles):
	# i is file number
	print("Reading file " + str(i+1))
	if (numberOfFiles==1):
		colvarFile=np.genfromtxt(colvarFileName)
	else:
		colvarFile=np.genfromtxt(colvarFileName + "." + str(int(i)))
	print("Finished reading file " + str(i+1))
	sizeColvar=colvarFile.shape[0]
	numColumns=colvarFile.shape[1]
	# Perform some checks on the input parameters
	if (ignore>sizeColvar):
		raise InputError("The number of lines to ignore cannot be greater than the size of the file.")
	if (newCVcolumn>(numColumns-1)):
		raise InputError("The number of column corresponding to the CV is greater than the number of columns in the file.")
	if (weightsFlag):
		if (biasColumn>(numColumns-1)):
			raise InputError("The number of column corresponding to the bias is greater than the number of columns in the file.")
	# Do the calculation for this file
	for j in range(ignore,sizeColvar,readStride):
		# j is the position in the file
		CVvalue=colvarFile[j,newCVcolumn]
		if ( (CVvalue>=CVmin) and (CVvalue<=CVmax) ):
			k=int( (CVvalue-CVmin)*(Nbin-1) / (CVmax-CVmin) )
			# k is the bin in the histogram
			if (weightsFlag):
				tmp = np.exp(beta*colvarFile[j,biasColumn])
			#histogram[k] += tmp 
			#denom += tmp
			samples[counts,k]=1
			if (weightsFlag):
				weights[counts]=tmp
			counts += 1
	# Free memory explicitly
	del colvarFile

######################################################
# Trim samples and weights arrays
######################################################
samples = samples[:counts,:]
if (weightsFlag):
	weights = weights[:counts]	
print("Counts: " + str(int(counts)))
######################################################

if (not(errorConvergence)):
	######################################################
	# Calculate error with bootstrap
	######################################################
	print("Bootstrapping")
	if (weightsFlag):
		for i in range(Nbin):
			bootLow[i], bootHigh[i] = bootstrap(samples[:,i], bootstrapSamples, significanceLevel, weights[:])
	else:
		for i in range(Nbin):
			bootLow[i], bootHigh[i] = bootstrap(samples[:,i], bootstrapSamples, significanceLevel)
	print("Finished bootstrapping")
	######################################################
	
	
	######################################################
	# Normalize histogram and convert into free energy
	######################################################
	#histogram /= denom
	if (weightsFlag):
		for i in range(Nbin):
			histogram[i]=np.average(samples[:,i], 0, weights)
	else:
		for i in range(Nbin):
			histogram[i]=np.mean(samples[:,i], 0)
	newFES = -temp*np.log(histogram)
	offset = np.min(newFES)
	newFES -= offset
	newFESmin = -temp*np.log(bootLow)-offset
	newFESmax = -temp*np.log(bootHigh)-offset
	
	######################################################
	
	print("Calculation completed")
	
	########################################################
	# Output everything
	########################################################
	np.savetxt(outputFesFileName,np.transpose(np.vstack([CVvector,histogram,bootLow,bootHigh,newFES,newFESmin,newFESmax])))
	########################################################

else:
	######################################################
	# Calculate error with bootstrap
	######################################################
	print("Bootstrapping")
	counter=0
	if (weightsFlag):
		for i in range(errorConvergenceSampleStride,errorConvergenceMaxSample,errorConvergenceSampleStride):
			bootLow[counter], bootHigh[counter] = bootstrap(samples[:,errorConvergenceBin], i, significanceLevel, weights[:])
			sampleSize[counter]=i
			counter += 1
	else:
		for i in range(errorConvergenceSampleStride,errorConvergenceMaxSample,errorConvergenceSampleStride):
			bootLow[counter], bootHigh[counter] = bootstrap(samples[:,errorConvergenceBin], i, significanceLevel)
			sampleSize[counter]=i
			counter += 1
	print("Finished bootstrapping")
	######################################################
		
	######################################################
	# Get histogram value for the chosen bin
	######################################################
	#histogram /= denom
	if (weightsFlag):
		histogramValue=np.average(samples[:,errorConvergenceBin], 0, weights)
	else:
		histogramValue=np.mean(samples[:,errorConvergenceBin], 0)
	for i in range(numSamples):
		histogram[i]=histogramValue
	######################################################
	

	########################################################
	# Output everything
	########################################################
	np.savetxt(outputFesFileName,np.transpose(np.vstack([sampleSize,histogram,bootLow,bootHigh])))
	########################################################


