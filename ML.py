import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

#global variables
theta = []
iterations = 3000
learning_rate = 0.1
epsilom = 0.000000000000000001

def linReg(X, Y):
    X=np.transpose(X)
    Xt=np.transpose(X)
    #=X^T
    XtX=np.matmul(Xt,X)
    #=(X^T X)
    a, b = XtX.shape
    i = np.eye(a, a)
    invXtX=np.linalg.lstsq(XtX, i, rcond=None)[0]
    #=(X^T X)^(-1)
    invXtXXt=np.matmul(invXtX,Xt)
    #=(X^T X)^(-1) X^T
    global theta
    theta=np.matmul(invXtXXt,Y)
    #=(X^T X)^(-1) X^T Y

# Activation Function
def sigmaf(x):
    return 1/(1 + np.exp(-x))

def d_sigmaf(x):
    return (1 - sigmaf(x)) * sigmaf(x)

# Loss Functions 
def loss(y, a):
    global epsilom
    return -(y*np.log(a+epsilom) + (1-y)*np.log(1-a+epsilom))

def d_loss(y, a):
    global epsilom
    return (a - y)/((a+epsilom)*(1 - a+epsilom))

#Layer class was derived from an example of neural networks created by
#    Adarsh Menon encountered while browsing towardsdatascience.com.
#    Coded example available at: "https://gist.githubusercontent.com/adarsh1021/f08f58580852310886f0c97164cafe64/raw/97b6818e76eb483c821478566f0d04399320fbae/layerClassBackprop.py"
class Layer:
    
    activate = {
        "sigmoid": (sigmaf, d_sigmaf)
    }

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs) #Gaussian Distribution in np.random.randn
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activate.get(activation)
        
    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z)
        return self.A

    def backprop(self, dA):
        #By use of gradient descent method
        dZ = np.multiply(self.d_act(self.Z), dA)
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)
        global learning_rate
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

        return dA_prev

def neurnet(x,y):
    x=np.array(x)
    class_y=[]
    for it in range(len(y)):
        j=0
        while (1.58+0.42800000000000005*(j+1)<y[it]):
                    j+=1
        class_y.append(j)
    y=np.array([class_y])
    m,n = x.shape
    global iterations

    layer = Layer(16, 1, "sigmoid") #single layer neural network with sigmoid activation function

    for i in range(iterations):
        # Feedforward
        A = x
        A = layer.feedforward(A)

        # Backpropagation
        dA = d_loss(y, A)
        dA = layer.backprop(dA)

        # Predict
        A = layer.feedforward(x)
    global theta
    W0=np.transpose(layer.W)
    theta=(W0)

with open(r'C:\Users\CriticalHit\Documents\JupyterLab\smart_grid_stability_augmented.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    i=int(input('how many lines do you want to train?'))
    boolk=0
    for row in csv_reader:
        if line_count == 0:
            #useful: print(f'Column names are {", ".join(row)}')
            ones=[]
            x0=[]
            x02=[]
            x03=[]
            x04=[]
            x05=[]
            x1=[]
            x12=[]
            x13=[]
            x14=[]
            x15=[]
            x2=[]
            x22=[]
            x23=[]
            x24=[]
            x25=[]
            y=[]
            line_count += 1
        #this is line 36
        elif line_count <=i+1:
            #use power balance consumer5, price elasticity producer8 and price elasticity consumer9 to power balance producer4
            ones.append(1)
            pbc=float(row[5])
            x0.append(pbc)
            pbc2=pbc*pbc
            x02.append(pbc2)
            pbc3=pbc2*pbc
            x03.append(pbc3)
            pbc4=pbc3*pbc
            x04.append(pbc4)
            pbc5=pbc4*pbc
            x05.append(pbc5)
            pep=float(row[8])
            x1.append(pep)
            pep2=pep*pep
            x12.append(pep2)
            pep3=pep2*pep
            x13.append(pep3)
            pep4=pep3*pep
            x14.append(pep4)
            pep5=pep4*pep
            x15.append(pep5)
            pec=float(row[9])
            x2.append(pec)
            pec2=pec*pec
            x22.append(pec2)
            pec3=pec2*pec
            x23.append(pec3)
            pec4=pec3*pec
            x24.append(pec4)
            pec5=pec4*pec
            x25.append(pec5)
            y.append(float(row[4]))
            line_count += 1
        else:
            if (boolk==0):
                why=np.array(y)
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter3D(x0, x1, y, c=y, cmap='Greens');
                plt.show()
                algo=input("1: Linear Regression || 2: Neural Network")
                k=int(input('how many lines would you like to test?'))
                stime = time.time()
                x=[ones,x0,x02,x03,x04,x05,x1,x12,x13,x14,x15,x2,x22,x23,x24,x25]
                if (algo=="1"):
                    linReg(x,why)
                else:
                    neurnet(x,why)
                etime=time.time()
                boolk=1
                errornum=0
                errorden=0
                ypred=[]
                yact=[]
                correct=0
                incorrect=0
                relerror_array=[]
                diff_array=[]
                confusion_matrix=np.zeros((10,10))
                out_of_bounds=0
                line_count += 1
            elif line_count <=(i+k+2):
                a0=float(row[5])
                X0=(a0)
                X02=(a0*a0)
                X03=pow(a0,3)
                X04=pow(a0,4)
                X05=pow(a0,5)
                a1=float(row[8])
                X1=(a1)
                X12=(a1*a1)
                X13=pow(a1,3)
                X14=pow(a1,4)
                X15=pow(a1,5)
                a2=float(row[9])
                X2=(a2)
                X22=(a2*a2)
                X23=pow(a2,3)
                X24=pow(a2,4)
                X25=pow(a2,5)
                variables=(np.array([[1,X0,X02,X03,X04,X05,X1,X12,X13,X14,X15,X2,X22,X23,X24,X25]]))
                solution=np.matmul(variables,theta)
                if (algo=="1"):
                    prediction=((solution[0]))
                else:
                    prediction=(solution[0][0])
                strprediction=str(prediction)
                actual=float(row[4])
                stractual=str(actual)
                relerror=round(abs(float(row[4])-prediction)/float(row[4])*100,2)
                errornum+=relerror
                errorden+=1
                relerror_array.append(relerror)
                strrelerror=str(relerror)+"%"
                if (algo=="1" and relerror>50):
                    diff_array.append([X0,X1,X2])
                jp=0
                while (1.58+0.42800000000000005*(jp+1)<prediction):
                    jp+=1
                ypred.append(jp)
                ja=0
                while (1.58+0.42800000000000005*(ja+1)<actual):
                    ja+=1
                yact.append(ja)
                if (ja>9 or jp>9):
                    out_of_bounds+=1
                else:
                    confusion_matrix[jp][ja]+=1
                if (jp==ja):
                    correct+=1
                else:
                    incorrect+=1
                #print("Prediction: "+strprediction+"\n"+"\t"+"Actual: "+stractual+"\n"+"\t"+"Relative Error: "+strrelerror)
                line_count += 1
            else:
                print("time taken to run: "+str(round(etime-stime,3))+"s")
                ave=round(errornum/errorden, 2)
                print("Correct Classification Ratio:"+"\t"+str(correct)+":"+str(incorrect))
                print("Out of Classification Bounds: "+str(out_of_bounds))
                print("Confusion Matrix: ")
                print(confusion_matrix)
                if (algo=="1"):
                    print("Average Relative Error: "+str(ave)+"%")
                    difficult=input("Would you like to see difficulty points? Y/N ")
                    if (difficult=="Y"):
                        print(diff_array)
                    print("Plot showing relative error at each testing point:")
                    plt.plot(range(len(relerror_array)), relerror_array)
                break
    #useful: print(f'Processed {line_count-1} lines of data')