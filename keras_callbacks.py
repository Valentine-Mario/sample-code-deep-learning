import keras

callbacks_list =[
    #interupt when model improving stops
keras.callbacks.EarlyStopping(
    #monitior acc
    monitior="acc",
    #stop after 1 epoch of non improvement
    patience=1
),

#save to my_model.h5 only when validation loss improves
keras.callbacks.ModelCheckpoint(
filepath='my_model.h5',
monitor='val_loss',
save_best_only=True,
)
]

model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))

#reduce learning rate when val loss has stopped improving
callbacks_list2=[
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss'
        #Divides the learning rate by 10 when triggered
        factor=0.1,
        #trigger whn val loss does not improve after 10 epochs
        patience=10
    )
]

model.fit(x, y,
epochs=10,
batch_size=32,
callbacks=callbacks_list,
validation_data=(x_val, y_val))

#using tensorboard in callback
callbacks = [
keras.callbacks.TensorBoard(
log_dir='my_log_dir', #save log in this directory
histogram_freq=1, #Records activation histograms every 1 epoch
embeddings_freq=1, #recording embedding data every 1 epoch
)
]