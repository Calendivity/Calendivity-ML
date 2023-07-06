from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import tensorflow as tf
import os

def train_model_embedding(model, dataloader, loss_fn, optim, epochs, save_dir):
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    for epoch in range(epochs):
        loss_total = torch.tensor(0.)
        for x,y in tqdm(dataloader):
            out = model(x)
            loss = loss_fn(out, y.float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_total +=loss.item()
        writer.add_scalar("loss", loss_total, epoch)
        if (epoch+1)%5 == 0:
            print("epochs :", epoch+1, " | loss :", loss.item())
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))



def train_model_durasi(model, train_data, loss_fn, optim, epochs, save_dir, val_data=None):
    model.compile(
        # loss = lambda x, dist : - dist.log_prob(x),
        # loss ='mse',
        loss = loss_fn,
        # optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        optimizer = optim,
        metrics=['mae', 'mse']
    )

    # logdir = 'drive/MyDrive/Bangkit/model_durasi/logs'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'weights'),
        save_best_only=True,
        save_weights_only=True
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(save_dir, 'logs'))

    if val_data != None:
        model.fit(train_data, epochs=epochs,validation_data = val_data,
                    callbacks=[tensorboard, checkpoint])
    else:
        model.fit(train_data, epochs=128, #validation_data = test_data,
                    callbacks=[tensorboard, checkpoint])



