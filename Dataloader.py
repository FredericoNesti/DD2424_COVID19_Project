import PIL
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import urllib3
import urllib.request
import tarfile
from torchvision.utils import save_image

import os

train_dataset = ImageFolder(root=args.train_set, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

class DATABASE(Dataset):
    def __init__(self,batch,device='cuda',split = [0.6,0.1,0.3]):
        '''
        :param batch: batch size
        :param device: default cuda
        :param split: proportional split for train,validation and test. Default: [0.6,0.1,0.3]
        '''
        self.batch = batch
        self.split = split
        self.device = device
        self.database = []
        self.links_gz = ['https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
                      'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
                      'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
                      'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
                      'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
                      'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
                      'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
                      'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
                      'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
                      'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
                      'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
                      'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz']


    def split_database(self):
        return train_set, val_set, test_set




    def add_source_gz(self,extra_source):
        '''
        Use this method only for "gz" format
        :param extra_source:
        :return:
        '''
        self.links_gz.append(extra_source)


    def preprocess(self):


    def add_to_database(self):


    def load_database_to_server(self):
        



    def load_dataset(self,source):
        # in case dataset is online
        '''
        http = urllib3.PoolManager()


        save = []
        for idx, link in enumerate(links):
            print('downloading', idx, '...')
            fn = 'images_%02d.tar.gz' % (idx + 1)
            ftpstream = urllib.request.urlopen(link + '/' + fn)
            thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
            tar = tarfile.open(thetarfile, "r:gz")
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()

                    print('concluded', idx)
                    save.append(content)
                    print('')
        '''

        # in case dataset is inside machine
        self.database.append(ImageFolder(root=source, transform=ToTensor()))




    def augmentate(self, type):

        if type == 'rotation':

        elif type == 'translation':

        elif type == 'horizontal flip':

        elif type == 'intensity shift':

        else:
            print('Enter a valid type.')
