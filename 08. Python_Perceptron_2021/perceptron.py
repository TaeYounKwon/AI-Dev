import csv
theta = 0.2
alpha = 1


# Read Data from File 
# Just need to type the file name
# Ex) data.txt
# Combine the readable FileName
file=input('Put the file Name(ex: data.txt): ')
fileName=[]
fileName.append('.\\')
fileName.append(file)
newFileName=''.join(fileName)
trainingFile = open(newFileName)
samples = []
t = []
exampleCount = 0

for example in trainingFile:# example will be 1,1,1 
   samples.append([])
   raw = example.split(',')# raw will be '1','1','1\n'
   for i in range(len(raw) - 1):
      samples[exampleCount].append(float(raw[i])) # sample will be [1.0, 1.0]
   t.append(float(raw[len(raw)-1])) # t-value will be 1.0
   exampleCount = exampleCount + 1

# Initialize Weights in Bias
print('')
bias = 0
w = []
dimensions = len(samples[0])
for i in range(dimensions):
   w.append(0)

bigLoop = True
count_while=0
sampleNumber=0

while bigLoop:
	w_start=w[:]
	bias_start=bias	
	count=0
	loop=True
	while loop:
    		
		for sampleNumber in range(len(samples)):
			w_before=w[:]
			print('    ------------------------------')		    		
			print('    Sample: ',count)
			print('    Old Values: ',w_before,' ',bias)
			x = samples[sampleNumber]
			y_in = sum([a*b for a,b in zip(x,w)])+bias
			if y_in > theta:
				y = 1
			else:
				if y_in >= -theta and y_in <= theta:
					y = 0
				else:
					y = -1
			if y != t[sampleNumber]:
				for i in range(dimensions):
    					w[i] = w[i] + alpha * (t[sampleNumber]-y) * x[i] # weight changed equals to learning rate*(target val - perceptron output)*input val
				bias = bias + alpha * t[sampleNumber]
			print('    New Values: ',w,' ',bias,'\n')
			# Test Condition
			count+=1
			if(w_before==w): 				
				break
		w_end=w
		bias_end=bias	
		loop=False
		
	count_while=count_while+1		
	print('....................................')
	print('Epoch: ', count_while)
	print('Previous Values: ',w_start,' ',bias_start)
	print('Current Values: ',w_end,' ',bias_end)
	print('....................................')
	if(w_start==w_end):
		bigLoop=False
