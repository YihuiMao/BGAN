import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
#from vis_tools import save_img,make_img
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb
from torch.autograd import Variable


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES']="0"

img_dir = './data/train/'
img_dir_1 = './data/val/'
img_shape = (3, 128, 128)  
num_epochs = 20
batch_size = 12
lr_rate = 0.0002  
betas = (0.5, 0.99)  
lambda_pixel = 10  
lambda_latent = 0.5  
lambda_kl = 0.01  
latent_dim = 8  
gpu_id = "cuda:0"
criterion_mse = nn.MSELoss()




fixed_z = Variable(torch.randn(100, 5, 8).type(torch.cuda.FloatTensor), requires_grad = True)
fixed_z_1 = Variable(torch.randn(100, 1, 8).type(torch.cuda.FloatTensor), requires_grad = True)


def make_img_1(d_loader, G, z, index, img_num, img_size):
    loader = iter(d_loader)
    if os.path.isdir("./image_demo_1") == False:
        os.mkdir("./image_demo_1")
    id_ = 0
    for idx, data in enumerate(loader):

        if (idx < 3):
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor), norm(rgb_tensor)

            edge_img = Variable(edge_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
            rgb_img = Variable(rgb_tensor.type(torch.cuda.FloatTensor), requires_grad = True)

            for i in range(edge_img.shape[0]):
                fig = plt.figure(figsize = (15, 10))
                fig.add_subplot(1, 7, 1)
                edge_display = np.transpose(np.array(edge_img[i].data.detach().cpu()), (1, 2, 0))
                plt.imshow(edge_display)
                plt.axis('off')
                plt.title("input")

                fig.add_subplot(1, 7, 2)
                rgb_display = np.transpose(np.array(rgb_img[i].data.detach().cpu()), (1, 2, 0))
                plt.imshow(rgb_display)
                plt.axis('off')
                plt.title("gt")

                for j in range(img_num):
                    img = edge_img[i].unsqueeze(dim = 0)
                    z_ = z[i, j, :].unsqueeze(dim = 0)
                    out_img = G(img, z_)
                    fig.add_subplot(1, 7, j + 3)
                    sample_image = np.transpose(np.array(out_img[0].data.detach().cpu()), (1, 2, 0))
                    plt.imshow(sample_image)
                    plt.axis('off')
                    plt.title("sample " + str(j + 1))
                id_ = id_ + 1
                plt.savefig('./image_demo_1/' +str(index)+"_"+str(id_) + ".png")

                plt.show(block = False)
                plt.pause(0.5)
                plt.close()

def make_img(d_loader,G,z,img_num,img_size):
    loader = iter(d_loader)
    if os.path.isdir("./image_demo")==False: 
        os.mkdir("./image_demo")   
    id_=0
    for idx, data in enumerate(loader):

        if(idx<3):
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor), norm(rgb_tensor)

            edge_img = Variable(edge_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
            rgb_img = Variable(rgb_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
            

            for i in range(edge_img.shape[0]):
                fig = plt.figure(figsize=(15, 10))
                fig.add_subplot(1, 7, 1)
                edge_display=np.transpose(np.array(edge_img[i].data.detach().cpu()), (1, 2, 0))
                plt.imshow(edge_display)
                plt.axis('off')
                plt.title("input")
                
                fig.add_subplot(1, 7, 2)
                rgb_display=np.transpose(np.array(rgb_img[i].data.detach().cpu()), (1, 2, 0))
                plt.imshow(rgb_display)
                plt.axis('off')
                plt.title("gt")
                
                for j in range(img_num):
                    img = edge_img[i].unsqueeze(dim=0)
                    z_ = z[i, j, :].unsqueeze(dim=0)
                    out_img = G(img, z_)
                    fig.add_subplot(1, 7, j+3)
                    sample_image=np.transpose(np.array(out_img[0].data.detach().cpu()), (1, 2, 0))
                    plt.imshow(sample_image)
                    plt.axis('off')
                    plt.title("sample "+str(j+1))
                id_=id_+1
                plt.savefig('./image_demo/'+str(id_)+".png")
                
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()   


def save_img(d_loader,G,z,img_num,img_size):
    loader = iter(d_loader)
    if os.path.isdir("./image_input")==False: 
        os.mkdir("./image_input")
    if os.path.isdir("./image_rgb")==False: 
        os.mkdir("./image_rgb") 
    if os.path.isdir("./image_sample")==False: 
        os.mkdir("./image_sample")   
    # if os.path.isdir("./image_sample2")==False: 
    #     os.mkdir("./image_sample2") 
    # if os.path.isdir("./image_sample3")==False: 
    #     os.mkdir("./image_sample3") 
    # if os.path.isdir("./image_sample4")==False: 
    #     os.mkdir("./image_sample4")
    # if os.path.isdir("./image_sample5")==False: 
    #     os.mkdir("./image_sample5")
    
    id_=0     
    for idx, data in enumerate(loader):

        
        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor), norm(rgb_tensor)

        edge_img = Variable(edge_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
        rgb_img = Variable(rgb_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
            

            
        for i in range(edge_img.shape[0]):

            edge_display=np.transpose(np.array(edge_img[i].data.detach().cpu()), (1, 2, 0))
            plt.imshow(edge_display)
            plt.savefig('./image_input/'+str(id_)+".png")
            
            edge_display=np.transpose(np.array(rgb_img[i].data.detach().cpu()), (1, 2, 0))
            plt.imshow(edge_display)
            plt.savefig('./image_rgb/'+str(id_)+".png")
 
            for j in range(img_num):
                img = edge_img[i].unsqueeze(dim=0)
                z_ = z[i, j, :].unsqueeze(dim=0)
                out_img = G(img, z_)
                sample_image=np.transpose(np.array(out_img[0].data.detach().cpu()), (1, 2, 0))
                plt.imshow(sample_image)
                plt.savefig('./image_sample/'+str(id_)+".png")
            id_=id_+1
            print("id/200",id_)








def L1_loss(pred, target):

    return torch.mean(torch.abs(pred - target))



def norm(image):
    return (image / 255.0 - 0.5) * 2.0

def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0

weight_update = 0
D_loss_list=[]
G_loss_list=[]
onlyG_loss_list=[]
KL_loss_list=[]
wd_list=[]
img_recon_loss_list=[]

torch.manual_seed(1)
np.random.seed(1)


dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size = batch_size)
test_dataset = Edge2Shoe(img_dir_1)
test_loader = data.DataLoader(test_dataset, batch_size = batch_size)


mae_loss = torch.nn.L1Loss().to(gpu_id)



generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)
Disc_VAR = Discriminator().to(gpu_id)
Disc_LR = Discriminator().to(gpu_id)


optim_Encoder = torch.optim.Adam(encoder.parameters(), lr = lr_rate, betas = betas)
optim_Gen = torch.optim.Adam(generator.parameters(), lr = lr_rate, betas = betas)
optim_Disc_VAR = torch.optim.Adam(Disc_VAR.parameters(), lr = lr_rate, betas = betas)
optim_Disc_LR = torch.optim.Adam(Disc_LR.parameters(), lr = lr_rate, betas = betas)


valid = 1;
fake = 0

# Training
total_steps = len(loader) * num_epochs
step = 0
for e in range(num_epochs):
    start = time.time()
    print("-------------------")
    for idx, data in enumerate(loader):

        edge_tensor, rgb_tensor = data
        edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)

        edge_img = Variable(edge_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
        rgb_img = Variable(rgb_tensor.type(torch.cuda.FloatTensor), requires_grad = True)
        
        cVAE_img=edge_img
        cVAE_gt=rgb_img
        cLR_img=edge_img
        cLR_gt=rgb_img
        size_=cVAE_img.shape[0]



        

        mu, var_ = encoder(cVAE_gt)
        std = torch.exp(var_ / 2)
        random_latent_z = Variable(torch.randn(size_, latent_dim).type(torch.cuda.FloatTensor), requires_grad = True)
        encoded_z = (random_latent_z * std) + mu

        F_cVAE = generator(cVAE_img, encoded_z)

        RD_cVAE_1, RD_cVAE_2 = Disc_VAR(cVAE_gt)
        FD_cVAE_1, FD_cVAE_2 = Disc_VAR(F_cVAE)




        
        loss_1_D_loss_cVAE_1=criterion_mse(RD_cVAE_1, Variable(torch.ones(RD_cVAE_1.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        loss_2_D_loss_cVAE_1=criterion_mse(FD_cVAE_1, Variable(torch.zeros(FD_cVAE_1.size()).type(torch.cuda.FloatTensor), requires_grad = False)) 
        D_loss_cVAE_1 =  loss_1_D_loss_cVAE_1 +  loss_2_D_loss_cVAE_1
        
        loss_1_D_loss_cVAE_2=criterion_mse(RD_cVAE_2, Variable(torch.ones(RD_cVAE_2.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        loss_2_D_loss_cVAE_2=criterion_mse(FD_cVAE_2, Variable(torch.zeros(FD_cVAE_2.size()).type(torch.cuda.FloatTensor), requires_grad = False)) 
        D_loss_cVAE_2 =  loss_1_D_loss_cVAE_2 +  loss_2_D_loss_cVAE_2       
        


        random_latent_z =  Variable(torch.randn(size_, latent_dim).type(torch.cuda.FloatTensor), requires_grad = True)


        F_cLR = generator(cLR_img, random_latent_z)


        RD_cLR_1, RD_cLR_2 = Disc_LR(cLR_gt)
        FD_cLR_1, FD_cLR_2 = Disc_LR(F_cLR)
        
        loss_1_D_loss_cLR_1=criterion_mse(RD_cLR_1, Variable(torch.ones(RD_cLR_1.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        loss_2_D_loss_cLR_1=criterion_mse(FD_cLR_1, Variable(torch.zeros(FD_cLR_1.size()).type(torch.cuda.FloatTensor), requires_grad = False)) 
        D_loss_cLR_1 =  loss_1_D_loss_cLR_1 +  loss_2_D_loss_cLR_1
        
        loss_1_D_loss_cLR_2=criterion_mse(RD_cLR_2, Variable(torch.ones(RD_cLR_2.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        loss_2_D_loss_cLR_2=criterion_mse(FD_cLR_2, Variable(torch.zeros(FD_cLR_2.size()).type(torch.cuda.FloatTensor), requires_grad = False)) 
        D_loss_cLR_2 =  loss_1_D_loss_cLR_2 +  loss_2_D_loss_cLR_2 



        D_loss = D_loss_cVAE_1 + D_loss_cLR_1 + D_loss_cVAE_2 + D_loss_cLR_2

  
        optim_Disc_VAR.zero_grad()
        optim_Disc_LR.zero_grad()
        optim_Gen.zero_grad()
        optim_Encoder.zero_grad()
        D_loss.backward()
        optim_Disc_VAR.step()
        optim_Disc_LR.step()



        mu, var_ = encoder(cVAE_gt)
        std = torch.exp(var_ / 2)
        random_latent_z =  Variable(torch.randn(size_, latent_dim).type(torch.cuda.FloatTensor), requires_grad = True)
        encoded_z = (random_latent_z * std) + mu

        F_cVAE = generator(cVAE_img, encoded_z)
        FD_cVAE_1, FD_cVAE_2 = Disc_VAR(F_cVAE)

        Gloss_cVAE_1 = criterion_mse(FD_cVAE_1, Variable(torch.ones(FD_cVAE_1.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        Gloss_cVAE_2 = criterion_mse(FD_cVAE_2, Variable(torch.ones(FD_cVAE_2.size()).type(torch.cuda.FloatTensor), requires_grad = False))



        random_latent_z = Variable(torch.randn(size_, latent_dim).type(torch.cuda.FloatTensor), requires_grad = True)


        F_cLR = generator(cLR_img, random_latent_z)
        FD_cLR_1, FD_cLR_2 = Disc_LR(F_cLR)

        GLoss_cLR_1 = criterion_mse(FD_cLR_1, Variable(torch.ones(FD_cLR_1.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        GLoss_cLR_2 = criterion_mse(FD_cLR_2, Variable(torch.ones(FD_cLR_2.size()).type(torch.cuda.FloatTensor), requires_grad = False))



        random_latent_z = Variable(torch.randn(size_, latent_dim).type(torch.cuda.FloatTensor), requires_grad = True)

 
        F_cLR = generator(cLR_img, random_latent_z)
        FD_cLR_1, FD_cLR_2 = Disc_LR(F_cLR)
        GLoss_cLR_1 = criterion_mse(FD_cLR_1, Variable(torch.ones(FD_cLR_1.size()).type(torch.cuda.FloatTensor), requires_grad = False))
        GLoss_cLR_2 = criterion_mse(FD_cLR_2, Variable(torch.ones(FD_cLR_2.size()).type(torch.cuda.FloatTensor), requires_grad = False))


        G_loss = Gloss_cVAE_1 + Gloss_cVAE_2 + GLoss_cLR_1 + GLoss_cLR_2


        KL_div = lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(var_) - var_ - 1))


        img_recon_loss = lambda_pixel * L1_loss(F_cVAE, cVAE_gt)

        EG_loss = G_loss + KL_div + img_recon_loss
        optim_Disc_VAR.zero_grad()
        optim_Disc_LR.zero_grad()
        optim_Gen.zero_grad()
        optim_Encoder.zero_grad()

        EG_loss.backward(retain_graph = True)
        optim_Encoder.step()
        optim_Gen.step()



        mu_, var__ = encoder(F_cLR.detach())

        onlyG_loss = lambda_latent * L1_loss(mu_, random_latent_z)

        optim_Disc_VAR.zero_grad()
        optim_Disc_LR.zero_grad()
        optim_Gen.zero_grad()
        optim_Encoder.zero_grad()
        onlyG_loss.backward()
        optim_Gen.step()
        


        print("iter",idx," ",D_loss.item(),EG_loss.item(),onlyG_loss.item())
        wd_list.append(weight_update/1000)
        D_loss_list.append(D_loss.item())
        G_loss_list.append(G_loss.item())
        onlyG_loss_list.append(onlyG_loss.item())
        KL_loss_list.append(KL_div.item())
        img_recon_loss_list.append(img_recon_loss.item())
        if weight_update%2000==0:
            make_img_1(test_loader, generator, fixed_z,weight_update,img_num=5, img_size=128)
        weight_update = weight_update + 1
        # if weight_update==500:
        #     plt.plot(wd_list, D_loss_list)
        #     plt.title("Discriminator Loss")
        #     plt.xlabel("weight update per 100 iters ")
        #     plt.ylabel("Loss")
        #     plt.show()
        #     plt.savefig('Discriminator_Loss.png')
        #     plt.plot(wd_list, G_loss_list)
        #     plt.title("Generator loss")
        #     plt.xlabel("weight update per 100 iters ")
        #     plt.ylabel(" Loss")
        #     plt.show()
        #     plt.savefig('Generator_Loss.png')
        #
        #     plt.plot(wd_list, KL_loss_list)
        #     plt.title("KL Loss")
        #     plt.xlabel("weight update per 100 iters ")
        #     plt.ylabel("Loss")
        #     plt.show()
        #     plt.savefig('KL_Loss.png')
        #
        #     plt.plot(wd_list,img_recon_loss_list)
        #     plt.title("||B − G(A, z)||,  Reconstruction of ground truth image")
        #     plt.xlabel("weight update per 100 iters")
        #     plt.ylabel("loss")
        #     plt.show()
        #     plt.savefig('recon_gt_img.png')
        #
        #     plt.plot(wd_list, onlyG_loss_list, label="Only G, reconstruction of random latent")
        #     plt.title("Only G, reconstruction of random latent")
        #     plt.xlabel("weight update per 100 iters")
        #     plt.ylabel("loss")
        #     plt.show()
        #     plt.savefig('recon_random_z.png')
torch.save(generator,"./checkpoint.pth") 
make_img(test_loader, generator, fixed_z,img_num=5, img_size=128)
save_img(test_loader, generator, fixed_z_1,img_num=1, img_size=128)
plt.show()
plt.plot(wd_list, D_loss_list)
plt.title("Discriminator Loss")
plt.xlabel("weight update ")
plt.ylabel("Loss")
plt.savefig('Discriminator_Loss.png')
plt.show()
# plt.pause(0.5)
# plt.close() 
# plt.savefig('Discriminator_Loss.png')




plt.plot(wd_list, G_loss_list)
plt.title("Generator loss")
plt.xlabel("weight update ")
plt.ylabel(" Loss")
plt.savefig('Generator_Loss.png')
plt.show()
# plt.pause(0.5)
# plt.close() 
# plt.savefig('Generator_Loss.png')



plt.plot(wd_list, KL_loss_list)
plt.title("KL Loss")
plt.xlabel("weight update ")
plt.ylabel("Loss")
plt.savefig('KL_Loss.png')
plt.show()
# plt.pause(0.5)
# plt.close()
#plt.savefig('KL_Loss.png')
 


plt.plot(wd_list, img_recon_loss_list)
plt.title("||B, G(A, z)||,  Reconstruction of ground truth image")
plt.xlabel("weight update")
plt.ylabel("loss")
plt.savefig('recon_gt_img.png')
plt.show()
#plt.savefig('recon_gt_img.png')


plt.plot(wd_list, onlyG_loss_list, label = "Only G, reconstruction of random latent")
plt.title("Only G, reconstruction of random latent")
plt.xlabel("weight update")
plt.ylabel("loss")
plt.savefig('recon_random_z.png')
plt.show()
#plt.savefig('recon_random_z.png')
 



