
"""
Author: Nazmul
email: nazmul.bangladesh06@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
import os
import logging

plt.style.use("fivethirtyeight")

# Save model function
def save_model(model, filename):
  """It is used to save the trained model

  Args:
      model (python object): trained model to
      filename (string): Path to save the trained model
  """
  logging.info(f"Saved the trained model{filename}")


  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model, filePath)

# Prepare data function

def prepare_data(df):
  """It is used to separate the dependent and independent features

  Args:
      df (pd.DataFrame): It's the pandas dataframe

  Returns:
      tuple: It return the tuples of dependent and independent variable(X,y)
  """

  logging.info("Preparing the data by segregrating independent and dependent variables")
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y


def save_plot(df, file_name, model):
  """It is used to create to save plot for models..

  Args:
      df (pd.DataFrame): Input is pandas dataframe
      file_name (string): File name of the plot (and,or,xor e.t.c for example and.png)
      model (python object): Name of the model for example and/or/xor e.t.c
  """
  def _create_base_plot(df):
    logging.info("Creating the base plot")
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    logging.info("Plotting the decision boundary")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    #print(xx1)
    #print(xx1.ravel())
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()

  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f"Saving the plots at {plotPath}")

