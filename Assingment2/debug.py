import numpy as np
import torch
import torchvision as tv

model = tv.models.resnet50(pretrained= True)
data = torch.randn(1,3,64,64)
labels = torch.randn(1,1000)

prediction = model(data)
print('the size of prediction is:', prediction.size())
iteration = 200

for itr in range(iteration):
    mse = torch.nn.MSELoss(reduce=None, size_average=False)

    optim = torch.optim.SGD(model.parameters(), lr= 1e-2, momentum=0.9)
    optim.zero_grad()
    output = model(data)
    loss = mse(output, labels)
    loss.backward()
    optim.step()
    print('loss = ',loss)

output= output.detach().numpy()
labels = labels.detach().numpy()
print('accuracy = ', np.mean(abs(output-labels)<=0.01))

