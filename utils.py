import inspect
from tqdm import tqdm
import sys
from swin_transformer import SSLHead
import torch
import numpy as np

def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    return loss

def create_bgnn():
    model = pvig_ti_224_gelu()
    return model

def create_net(args):
    
    init_signature = inspect.signature(SwinTransformer.__init__)
    model_init_params = {name for name, _ in init_signature.parameters.items() if name != 'self'}
    model_params = {k: args[k] for k in model_init_params if k in args}
    model = SwinTransformer(**model_params)
    
    return model

def create_pre_trained_Swin_VIT(args):
    
    model = SSLHead(args)
    
    return model


def create_SFCN(args):
    
    init_signature = inspect.signature(SFCN.__init__)
    model_init_params = {name for name, _ in init_signature.parameters.items() if name != 'self'}
    model_params = {k: args[k] for k in model_init_params if k in args}
    print("Current Model Architecture is : SFCN")
    print("Here are Parameters setting : ")
    print(model_params)
    model = SFCN(**model_params)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch,distribution,corr):
    
    if distribution:
        model.train()
        accu_loss = torch.zeros(1).to(device) 
        accu_num = torch.zeros(1).to(device)   
        optimizer.zero_grad()

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)
        for step, batch in enumerate(data_loader):
            if not corr:
                image,y,bc_label,age = batch[0].float().to(device), batch[1].float().to(device),batch[2].float().to(device), batch[3].float().to(device)
                preds = model(image)
            else:
                image,y,bc_label,age,gender = batch[0].float().to(device), batch[1].float().to(device),batch[2].float().to(device), batch[3].float().to(device),batch[4].long().to(device)
                preds = model(image,gender)
    
            loss = my_KLDivLoss(preds,y)
            loss.backward()
            accu_loss += loss.detach()

            dl_io_sentence = "[train epoch {}] current loss: {:.3f}".format(epoch,loss.item())
            data_loader.desc = dl_io_sentence


            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()
            
        return accu_loss.mean().item()
    
    else:
        model.train()
        loss_function = torch.nn.MSELoss()
        epoch_loss = 0
        step = 0
        cur_ds = data_loader.dataset
        cur_dl = data_loader
        data_loader = tqdm(data_loader, file=sys.stdout)
        
        for batch_data in data_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].float().to(device)
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32).unsqueeze(dim=-1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(cur_ds) // cur_dl.batch_size
            dl_io_sentence = f"[train epoch {step}/{epoch_len}, train_loss: {loss.item():.4f}]"
            data_loader.desc = dl_io_sentence
            
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        return epoch_loss
            
#         for step, batch in enumerate(data_loader):
#             inputs = torch.tensor(batch[0], dtype=torch.float32).to(device)
#             labels = torch.tensor(batch[1], dtype=torch.float32).to(device)

#             sample_num += inputs.shape[0]
#             pred = model(inputs)

#             loss = loss_function(pred, labels)
#             loss.backward()
#             accu_loss += loss.detach()

#             dl_io_sentence = "[train epoch {}] current loss: {:.3f}".format(epoch,loss.item())
#             data_loader.desc = dl_io_sentence


#             if not torch.isfinite(loss):
#                 print('WARNING: non-finite loss, ending training ', loss)
#                 sys.exit(1)

#             optimizer.step()
#             optimizer.zero_grad()
            
#             return accu_loss.mean().item()
        
                                 
                        
@torch.no_grad()
def evaluate(model, data_loader, device, epoch,distribution,corr):
    
    if distribution:
        model.eval()

        accu_num = torch.zeros(1).to(device)  
        accu_loss = torch.zeros(1)  

        sample_num = 0
        data_loader = tqdm(data_loader, file=sys.stdout)

        val_ba = []
        val_age = []

        for step, batch in enumerate(data_loader):
            if  not corr:
                image,dist_label,bc_label,age_label = batch[0].float().to(device), batch[1].float().to(device),batch[2].float().to(device), batch[3].float().to(device)
                pred_dist = torch.exp(model(image))
            else:
                image,dist_label,bc_label,age_label,gender = batch[0].float().to(device), batch[1].float().to(device),batch[2].float().to(device), batch[3].float().to(device),batch[4].long().to(device)
                pred_dist = torch.exp(model(image,gender))
            
            pred_age = torch.bmm(pred_dist.unsqueeze(1), bc_label.unsqueeze(1).transpose(1, 2)).squeeze().float()
            
            cur_mae = torch.mean(torch.abs(age_label - pred_age).float()).cpu()
            accu_loss += cur_mae

            val_ba.extend(pred_age.cpu().tolist())
            val_age.extend(age_label.cpu().tolist())

        corr_mat = np.corrcoef(val_ba, val_age)
        corr = corr_mat[0,1]
        mae = np.mean(np.abs(np.array(val_ba) - np.array(val_age)))

        io_sentence = "[valid epoch {}] MAE: {:.3f} Corr: {:.3f}".format(epoch + 1,mae,corr)
        print(io_sentence)

        return corr, mae
    
    else:
        model.eval()
        mae_sum = 0.0
        metric_count = 0
        val_function = torch.nn.L1Loss()
        
        val_ba = []
        val_age = []

        
        data_loader = tqdm(data_loader, file=sys.stdout)
        
        for val_data in data_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_inputs = val_images.to(device=device, dtype=torch.float)
            val_labels = val_labels.to(device=device, dtype=torch.float).unsqueeze(dim=-1)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = val_function(val_outputs,val_labels)
                metric_count += 1
                mae_sum += value
                
        metric = mae_sum / metric_count
        val_ba.extend(val_outputs.view(-1,).cpu().tolist())
        val_age.extend(val_labels.view(-1,).cpu().tolist())
        corr_mat = np.corrcoef(np.array(val_ba),np.array(val_age))
        corr = corr_mat[0,1]
        io_sentence = "[valid epoch {}] MAE: {:.3f} Corr: {:.3f}".format(epoch + 1,metric,corr)
        print(io_sentence)
        
        return corr, metric

        
#         model.eval()

#         accu_num = torch.zeros(1).to(device)  
#         accu_loss = torch.zeros(1)  

#         sample_num = 0
#         data_loader = tqdm(data_loader, file=sys.stdout)

#         val_ba = []
#         val_age = []

#         for step, batch in enumerate(data_loader):
#             val_images, val_labels = batch[0].to(device), batch[1].to(device)
#             val_images = torch.tensor(batch[0], dtype=torch.float32).to(device)
#             val_labels = torch.tensor(batch[1], dtype=torch.float32).to(device)
#             sample_num += val_images.shape[0]

#             val_preds = model(val_images.to(device)).cpu().squeeze().numpy()
#             val_labels = val_labels.cpu().squeeze().numpy()

#             accu_loss += np.abs(val_preds - val_labels)

#             val_ba.append(val_preds)
#             val_age.append(val_labels)
#             data_loader.desc = "[valid epoch {}] MAE: {:.3f},".format(epoch,np.mean(np.abs(val_preds - val_labels)).item())

#         corr_mat = np.corrcoef(val_ba, val_age)
#         corr = corr_mat[0,1]
#         mae = np.mean(np.abs(np.array(val_ba) - np.array(val_age)))

#         io_sentence = "[valid epoch {}] MAE: {:.3f} Corr: {:.3f}".format(epoch,mae,corr)
#         print(io_sentence)

#         return corr, mae