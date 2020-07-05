#!/usr/bin/env python3

"""
<< train_mean_CNN.py >>


Arguments
---------



Output
------


"""

from sys import argv,exit
import argparse
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten
)
from tensorflow.keras.layers import (
    Conv2D,
    AveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    """
    Run the script from the command line.
    """

    """
    # Print docstring if only the name of the script is given
    if len(argv) < 2:
        print(__doc__)
        exit(0)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Options for train_min_CNN.py", add_help=True)

    required = parser.add_argument_group("required arguments")
    required.add_argument('-cu','--coal_units', action="store", type=float, default=1.0,
                            metavar='\b', help="branch scaling in coalescent units")

    args = parser.parse_args()
    cu   = args.coal_units
    """

    data = np.load("heliconius_min_data.npz")
    xtrain,xval,ytrain,yval = (
        data['xtrain'],
        data['xval'],
        data['ytrain'],
        data['yval']
    )
    del data

    model = Sequential()
    model.add(
        Conv2D(
            #8, kernel_size=(4,2),
            12, kernel_size=(4,2),
            #strides=(2,1),
            activation='relu',
            input_shape=(xtrain.shape[1],xtrain.shape[2],xtrain.shape[3])
        )
    )
    model.add(
        AveragePooling2D(
            pool_size=(2,1),
            strides=(2,1)
        )
    )
    model.add(Dropout(0.2))

    model.add(
        Conv2D(
            #16, kernel_size=(4,2),
            24, kernel_size=(4,2),
            #strides=(2,1),
            activation='relu'
        )
    )
    model.add(
        AveragePooling2D(
            pool_size=(2,1),
            strides=(2,1)
        )
    )
    model.add(Dropout(0.2))

    model.add(
        Conv2D(
        #24, kernel_size=(4,2),
        36, kernel_size=(4,2),
        activation='relu'
        )
    )
    model.add(
        AveragePooling2D(
            pool_size=(2,1),
            strides=(2,1)
        )
    )
    model.add(Dropout(0.2))

    model.add(
        Conv2D(
        #32, kernel_size=(4,2),
        48, kernel_size=(4,2),
        activation='relu'
        )
    )
    model.add(
        AveragePooling2D(
            pool_size=(2,1),
            strides=(2,1)
        )
    )
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(60, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
    )
    print(model.summary())
    #exit(0)

    callbacks = [
        EarlyStopping(monitor='val_loss'),
        ModelCheckpoint(
            filepath='heliconius_cnn_min.mdl',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    model.fit(
        xtrain, ytrain,
        batch_size=32,
        epochs=10,
        verbose=1,
        callbacks=callbacks,
        validation_data=(xval,yval)
    )
