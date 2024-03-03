#This file was not used to create the models used in this assignment. The .ipnyb file used on Google Colab was

from utilities import *

# Train
captions_json = './training_label.json'
feat_folder = './MLDS_hw2_1_data/training_data/feat/'

train_dataset = TrainDataset(feat_folder, captions_json)
trainloader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=0, shuffle=True, collate_fn=Collate(pad_idx=0))

learning_rate = 1e-4
epochs=23
input_size=4096
hidden_size=512
vocabs = train_dataset.vocab
vocab_size=len(vocabs.wtoi)
embed_dim=256
LOAD_MODEL = False

if LOAD_MODEL == True:
    model = torch.load('./models/Model1.h5')
    model.to(device)
else:
    model = S2VTMODEL(input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size, embed_dim=embed_dim)
    model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocabs.wtoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)


modelloss=[]

model.train()
step=0
bestloss = 20
for epoch in range(0, epochs):
    trainloss = 0
    step = 0
    for idx, features, captions in trainloader:
        if step%10==0:
            print(f'Epoch: {epoch+1}  Step: {step+1}')
        step+=1
        features = features.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        outputs, seq_pred = model(features, captions, 'train')
        targets = captions[:, 1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        trainloss += loss.item()
        loss.backward()
        optimizer.step()
    avgloss = trainloss/len(trainloader)
    modelloss.append(avgloss)
    if epoch%1==0:
        print('Epoch: {}    TotalLoss: {}'.format(epoch, avgloss))
    if avgloss < bestloss:
        bestloss = avgloss
        try:
          torch.save(model, f"{'Model1'}.h5")
        except:
          print(model)
