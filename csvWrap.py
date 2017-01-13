import csv
def writeCSV(filename,array,verbose=True):
	try:
		with open(filename,'w') as csvFile:
			csvWriter=csv.writer(csvFile,delimiter=',')
			csvWriter.writerow(array)
			if verbose:
				print('wrote '+filename)
	except:
		print('unable to write to '+filename)

def readCSV(filename,verbose=True):
	try:
		with open(filename,'r') as csvFile:
			csvReader=csv.reader(csvFile,delimiter=',')
			if verbose:
				print('read '+filename)
			ret=[]
			for i in csvReader:
				ret.append(i)
			if len(ret)==1:
				return ret[0]
			else:
				return ret
	except:
		print('failed to open '+filename)
