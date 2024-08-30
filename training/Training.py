#This running script is adapted and edited from ProteinMPNN project.
import os.path
import torch
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import numpy as np
import multiprocessing
from utils_Antibody import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
from model_utils_Antibody import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
from tqdm import tqdm
import torch

def main():
    import json, time, os, sys, glob, os.path
    import shutil
    import warnings
    import numpy as np  
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed    
    from utils_Antibody import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils_Antibody import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    cwd = os.getcwd()
    print("Current working directory: {0}".format(cwd))

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available)

    path_for_outputs = "./training_output"# Training parameters
    base_folder = time.strftime(path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = [] 
    previous_checkpoint = ''
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    path_for_training_data = "./training_sets" # Training parameters
    data_path = path_for_training_data
    params = {
        "LIST": f"{data_path}/list_All_6578_data.csv", 
        "VAL": f"{data_path}/valid_train_all_6578.txt",
        "TEST": f"{data_path}/test_na_all_6578.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": 100000, 
        "HOMO": 0.70 
    }# Training parameters

    print(data_path)
    print(params)

    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory': False,
                  'num_workers': 4} 
    debug = False
    if debug is True:
        num_examples_per_epoch = 50
        max_protein_length = 1000
        batch_size = 1000
    train, valid, test = build_training_clusters(params, debug)
    print(len(test), len(valid), len(train))
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    
    hidden_dim = 128 # Training parameters
    num_encoder_layers = 3 # Training parameters
    num_decoder_layers = 3 # Training parameters
    num_neighbors = 48 # Training parameters
    dropout = 0.2 # Training parameters
    backbone_noise = 0.5 # Training parameters

    model = ProteinMPNN(node_features=hidden_dim, 
                        edge_features=hidden_dim, 
                        hidden_dim=hidden_dim, 
                        num_encoder_layers=num_encoder_layers, 
                        num_decoder_layers=num_encoder_layers, 
                        k_neighbors=num_neighbors, 
                        dropout=dropout, 
                        augment_eps=backbone_noise)
    model.to(device)
    if PATH:
        checkpoint = torch.load(PATH, weights_only=True)
        total_step = checkpoint['step'] 
        epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0
    optimizer = get_std_opt(model.parameters(), hidden_dim, total_step)
    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    max_protein_length = 10000 # Training parameters
    num_examples_per_epoch = 1000000 # Training parameters
    batch_size = 1000 # Training parameters
    num_epochs = 200 # Training parameters
    reload_data_every_n_epochs = 4 # Training parameters
    gradient_norm = -1.0 # Training parameters
    save_model_every_n_epochs = 2 # Training parameters
    mixed_precision = True # Training parameters
    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        for i in tqdm(range(3)):
            q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
            p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))

        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result() 
        dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length) 
        dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)
        loader_train = StructureLoader(dataset_train, batch_size=batch_size)
        loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)
        reload_c = 0 
        for e in tqdm(range(num_epochs)):
            t0 = time.time()
            e = epoch + e
            model.train()
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            if e % reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=max_protein_length)
                    loader_train = StructureLoader(dataset_train, batch_size=batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=max_protein_length)
                    loader_valid = StructureLoader(dataset_valid, batch_size=batch_size)
                    q.put_nowait(executor.submit(get_pdbs, train_loader, 1, max_protein_length, num_examples_per_epoch))
                    p.put_nowait(executor.submit(get_pdbs, valid_loader, 1, max_protein_length, num_examples_per_epoch))
                reload_c += 1
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                elapsed_featurize = time.time() - start_batch
                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                
                if mixed_precision is True:
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
           
                    scaler.scale(loss_av_smoothed).backward()
                     
                    if gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    if gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)

                    optimizer.step()
                
                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                
                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                for _, batch in enumerate(loader_valid):
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, device)
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    mask_for_loss = mask*chain_M
                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
    
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}')
            
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges': num_neighbors,
                        'noise_level': backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges': num_neighbors,
                        'noise_level': backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)

if __name__ == '__main__':
    main()
