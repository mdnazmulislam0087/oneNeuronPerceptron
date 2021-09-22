"""
Author: Nazmul
email: nazmul.bangladesh06@gmail.com
"""


from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data,eta,epochs,fileName,plotName):


    df = pd.DataFrame(data)

    print(df)

    X,y = prepare_data(df)



    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=fileName)
    save_plot(df,plotName, model)

if __name__== '__main__':
    XOR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,0],
    }
    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    main(data=XOR,eta=ETA,epochs=EPOCHS,fileName="xor.model",plotName="xor.png")
