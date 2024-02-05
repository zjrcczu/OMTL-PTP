

def get_ethucy_split(dataset):

     if dataset=='eth':
          seqs = ['biwi_eth']
     elif dataset == 'hotel':
          seqs = ['biwi_hotel']
     elif dataset == 'univ':
          seqs = ['students001','students003']
     elif dataset == 'zara1':
          seqs = ['crowds_zara01']
     elif dataset == 'zara2':
          seqs = ['crowds_zara02']




     train, val,test = [], [], []
     for seq in seqs:

          train.append(f'{seq}')
          val.append(f'{seq}')
          test.append(f'{seq}')
     return train, val, test