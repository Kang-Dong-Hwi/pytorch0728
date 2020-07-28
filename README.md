# pytorch0728



### mag, phase nomalization
-------
<br>
(257, 382) size의  numpy array 2개를  위아래로 붙인 뒤  ( 257*2, 382 ) size로 만들어 주고  <br>
( 257 * 2 * 382 )개의 sample에 대한 평균, 표준편차로 정규화해주었습니다.  <br>
 
정규화된 numpy array (z) 의 shape을 (2, 257, 382)로 바꿔주고 <br>
반환값은 tuple type으로 <br>
z[0]은 정규화된 left magnitude, z[1]은 정규화된 right magnitude 입니다.  

~~~python
def Mag_normalization( L, R ):
    
    samples = np.concatenate((L,R), axis=0)
    mu = np.mean( samples )
    sigma = np.std( samples )

    z = (samples-mu) / sigma
    z = z.reshape(2, L.shape[0], L.shape[1])
    
    return z[0], z[1]
    
~~~


left phase와 right phase의 정규화는 따로 해주었습니다. 
~~~python
def Phase_normalization( phase ):
    mu = np.mean( phase )
    sigma = np.std( phase )
    
    z = ( phase - mu ) / sigma
    return z
~~~

<br><br><br>

### model 학습
----

training dataset을 model에 적용하여 학습하는 코드입니다.
loss, accuracy를 표시하는 코드 외에 필요한 것만 남긴 부분입니다.

~~~python
# cifar10 코드 참고하였습니다.

torch.manual_seed(100)
criterion = nn.CrossEntropyLoss().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


model.train()
EPOCHS = 50
for epoch in range(EPOCHS):
    
    for i, (data, label) in enumerate(train_dataset):
        (data, label) = (data.to('cuda'), label.to('cuda'))

        #zero the parameter gradients
        optimizer.zero_grad()        

        # forward + backward = optimize
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
~~~        
        
    
    
    
<br><br>    
        
전체코드입니다.
~~~python

torch.manual_seed(100)
criterion = nn.CrossEntropyLoss().to('cuda')


'''optimizer'''

#optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)




EPOCHS = 50
train_loss = []
train_acc  = []


# confusion matrix를 만들기 위해서 
# 먼저 int type의 random값이 있는 
# (1,) size torch tensor 초기화
torch_pred  = torch.empty((1,), dtype=torch.int32).to('cuda')
torch_label = torch.empty((1,), dtype=torch.int32).to('cuda')



model.train()
for epoch in range(EPOCHS):
    #print('epoch ' + str(epoch+1))
    total_loss = 0.0
    total_acc = 0
    
    
    for i, (data, label) in enumerate(train_dataset):
        (data, label) = (data.to('cuda'), label.to('cuda'))

        #zero the parameter gradients
        optimizer.zero_grad()        

        # forward + backward = optimize
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        

        """loss, accuracy"""
        # batch 정확도
        preds  = torch.max(output.data, 1)[1]
        corr  = (preds==label).sum().item()
        acc   = corr/BATCH_SIZE*100

        # epoch 손실도, 정확도
        total_loss += loss.item()
        total_acc += corr
    
        # batch 손실도, 정확도 출력, 저장
        train_loss.append(loss)
        train_acc.append(acc)
        #if i%5==0: print('\tLoss: {:.3f}\tAcc: {:.3f}'.format(loss.item(), acc))


        """ confusion matrix"""
        if epoch == EPOCHS-1:
            torch_pred  = torch.cat( [ torch_pred , preds.to( 'cuda', dtype=torch.int32) ], dim=0 )
            torch_label = torch.cat( [ torch_label, label.to( 'cuda', dtype=torch.int32) ], dim=0 )


    # epoch 손실도, 정확도 출력
    print('epoch' + str(epoch+1) + '  >> Loss: {:.3f} \tAcc: {:.3f}'.format( total_loss, total_acc/800*100 ))
    #print()
~~~



<br><br><br>

### 1. optimizer : Adagrad  ( lr=0.00001, weight_decay=0.9 )
-----
<table>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix1.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix1.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix2.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix2.png", height=200px, width=250px>   </td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix3.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix3.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix4.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix4.png", height=200px, width=250px>   </td>
  </tr>
</table>


Adagrad를 optim로  
epoch 50, batch 50으로 실행한 결과 입니다.  

training dataset, validation dataset에서 모두 처음 예측값으로만  



<br>
<br>
<br>

### 2. xavier_uniform, kaiming_uniform
-------
xavier_uniform
kaiming_uniform
<br>

~~~python
import re
import torch.nn as nn
import torch.nn.functional as F



class CNN (nn.Module):
    def __init__(self, INIT):
        super(CNN, self).__init__()
        ''' 4 * 257 * 382'''
        self.conv1 = nn.Conv2d( in_channels=  4, out_channels= 64, kernel_size = (7,7), stride = (2,2) )
        #relu
        #pooling
        
        ''' 64 * 62 * 94'''
        self.conv2 = nn.Conv2d( in_channels= 64, out_channels= 64, kernel_size = (3,3) )
        #relu
        #pooling


        ''' 64 * 30 * 46'''
        self.conv3 = nn.Conv2d( in_channels= 64, out_channels= 32, kernel_size = (3,3) )
        #relu
        #pooling


        ''' 32 * 14 * 22'''
        #flatten

        self.lay1  = nn.Linear( 32*14*22, 256)
        self.lay2  = nn.Linear( 256, 256 )
        self.lay3  = nn.Linear( 256, 64 )
        self.lay4  = nn.Linear( 64 , 11 )
        

        if INIT==1 :
            # xavier initialization
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.conv3.weight)
            
            # kaiming he initialization
            nn.init.kaiming_uniform_(self.lay1.weight)
            nn.init.kaiming_uniform_(self.lay2.weight)
            nn.init.kaiming_uniform_(self.lay3.weight)
            nn.init.kaiming_uniform_(self.lay4.weight)


        
    def forward(self, output):
        output = F.max_pool2d( F.relu( self.conv1(output) ),2 )
        output = F.max_pool2d( F.relu( self.conv2(output) ),2 )
        output = F.max_pool2d( F.relu( self.conv3(output) ),2 )
        
        output = output.view(-1, 32*14*22)
        
        output = F.relu( self.lay1(output) )
        output = F.dropout(output, training=self.training)
        output = F.relu( self.lay2(output) )
        output = F.dropout(output, training=self.training)
        output = F.relu( self.lay3(output) )
        output = F.dropout(output, training=self.training)
        output = F.log_softmax(self.lay4(output), dim=1)
        
        return output
~~~
-----------
<br><br>

<table>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix11.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix11.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/train_dataset_confusion_matrix12.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/confusion_matrix/validation_dataset_confusion_matrix12.png", height=200px, width=250px>   </td>
  </tr>
  

</table>

convolution층 사이에는 xavier_uniform  
linear층 사이에는 kaiming_uniform 추가



<br>
<br>
<br>

### 3. optimizer : Adam
-----------
<table>
  <tr>  <td colspan="5"> lr=10e-3 </td> </tr>
  <tr>
        <td colspan="2">initializer X</td> <td></td>
        <td colspan="2">initializer O</td>
  </tr>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>

  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix201.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix201.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix202.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix202.png", height=200px, width=250px>   </td>
  </tr>
  
  
  <tr>  <td colspan="5"> <br><br> lr=10e-4 </td> </tr>
  <tr>
        <td colspan="2">initializer X</td> <td></td>
        <td colspan="2">initializer O</td>
  </tr>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix203.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix203.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix204.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix204.png", height=200px, width=250px>   </td>
  </tr>
  
  
  
  <tr>  <td colspan="5"> <br><br>  lr=10e-5 </td> </tr>
  <tr>
        <td colspan="2">initializer X</td> <td></td>
        <td colspan="2">initializer O</td>
  </tr>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix205.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix205.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix206.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix206.png", height=200px, width=250px>   </td>
  </tr>
  
  
  
  <tr>  <td colspan="5"> <br><br>  lr=10e-6 </td> </tr>
  <tr>
        <td colspan="2">initializer X</td> <td></td>
        <td colspan="2">initializer O</td>
  </tr>
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td></td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix207.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix207.png", height=200px, width=250px>   </td>
      <td></td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix208.png", height=200px, width=250px>        </td>
      <td>    <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix208.png", height=200px, width=250px>   </td>
  </tr>
  
</table>


Adam에서 lr이 큰 값에서 변수??초기화 효과?? 가 없었습니다.  
lr이 10e-6보다 작을 때 loss값이 오히려 높아지는 결과가 나왔습니다.
learning rate를 다양하게 해서 돌렸을V?? 때  
2*10e-5에서 loss값이 작게 측정되었습니다.


<br>
<br>
<br>

### Adam  ( lr = 2*10e-5 )
-------------
<table>
  <tr>  <td colspan="5"> lr=2*10e-5 </td> </tr>
  
  <tr>
        <td colspan="2">initializer X</td>
        <td colspan="2">initializer O</td>
  </tr>
  
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td>training dataset</td>
        <td>validation dataset</td>
  </tr>
  
  <tr>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix211.png", height=200px, width=250px>        </td>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix211.png", height=200px, width=250px>   </td>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix210.png", height=200px, width=250px>        </td>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix210.png", height=200px, width=250px>   </td>
  </tr>
  
  <tr>
      <td colspan="4"> <br><br> epoch 200, initializerO </td>
  </tr>
  
  <tr>
        <td>training dataset</td>
        <td>validation dataset</td>
        <td colspan="2">training loss, accuracy</td>
  </tr>
  
  <tr>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive01/train_dataset_confusion_matrix214.png", height=200px, width=250px>        </td>
      <td>   <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/gdrive02/validation_dataset_confusion_matrix214.png", height=200px, width=250px>   </td>
      <td colspan="2">  <img src="https://github.com/Kang-Dong-Hwi/pytorch0727/blob/master/Adam%20(1).png", height=250px, width=360px>                             </td> 
  </tr>

</table>


