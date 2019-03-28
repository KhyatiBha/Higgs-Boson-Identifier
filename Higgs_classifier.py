import numpy as np
import pandas as pd
import time
import sklearn
from tkinter.filedialog import *
root = Tk()
root.geometry('500x500')
root.title('AI Project')
global dfile

def fhand():
    global dfile
    dfile= askopenfile()
    def des():
    root.destroy()

def main():
    data = pd.read_csv('training.csv')
    shape=(data.shape)
    mylabe2 = Label(text='Data Shape:', font=('arial', 12, 'bold')).pack(anchor=W)
    mylabe2 = Label(text=shape, font=('arial', 10, 'bold')).pack(anchor=W)
    data = data.drop(['EventId', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_lep_eta_centrality', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta','PRI_jet_subleading_phi'],axis=1)

    from sklearn.model_selection import train_test_split
    X = data.drop(['Label'], axis=1)
    Y = data['Label']
    X_int_train, X_test, y_int_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    X_train, X_crossvalid, y_train, y_crossvalid = train_test_split(X_int_train, y_int_train, test_size = 0.33, random_state = 42)
    print(X_train.shape)
    print(X_crossvalid.shape)
    print(X_test.shape)

    #Neural network
    X_train = X_train.values
    X_crossvalid = X_crossvalid.values
    X_test = X_test.values
    y_train = y_train.values
    y_crossvalid = y_crossvalid.values
    y_test = y_test.values
    y_train = np.where(y_train == 's', 0, 1)
    y_test = np.where(y_test == 's', 0, 1)

    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(32, input_shape=(21,), activation='relu'))
    model.add(Dense(12, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    start_time = time.time()
    model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=2)
    traintime=str(time.time() - start_time)
    mylab = Label(text='Training Time in (s):', font=('arial', 12, 'bold')).pack(anchor=W)
    mylab1 = Label(text=traintime, font=('arial', 10, 'bold')).pack(anchor=W)
    mylab11 = Label(text=' ').pack(anchor=W)

    mylabe3 = Label(text='Training scores Accuracy:', font=('arial', 12, 'bold')).pack(anchor=W)
    scores = model.evaluate(X_train, y_train)
    acc= (scores[1] * 100)
    mylabe5 = Label(text=acc, font=('arial', 10, 'bold')).pack(anchor=W)
    mylabe4 = Label(text=' ').pack(anchor=W)

    mylabe6 = Label(text='Test scores Accuracy::', font=('arial', 12, 'bold')).pack(anchor=W)
    test_scores = model.evaluate(X_test, y_test)
    accu=(test_scores[1] * 100)
    mylabe5 = Label(text=accu, font=('arial', 10, 'bold')).pack(anchor=W)

mylabel= Label(text='Identification of Higgs Boson', font=('arial',20, 'bold')).pack()
btn = Button(text='Choose Dataset', width=60, command=fhand).pack()
btn2= Button(text='Next',width=60,command=main).pack()
btn2= Button(text='Close',width=60,command=des).pack()
root.mainloop()