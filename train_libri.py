import argparse
from argparse import ArgumentParser
import logging
import time

import torch
from torch.optim import SGD, Adam
import torch.nn.functional as F
from tqdm import tqdm

from dev.loaders import LibriSpeech4SpeakerRecognition, LibriSpeechSpeakers
from dev.models import RawAudioCNN, ALR, TDNN, SpectrogramCNN, DoubleModelCNN
from dev.utils import infinite_iter

from hparams import hp
import pdb, sys, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def _is_cuda_available():
    return torch.cuda.is_available()




def _get_device():
    return torch.device("cuda" if _is_cuda_available() else "cpu")


def noise_augmenter(inputs, labels, epsilon):
    """ Data augmentation with additive white uniform noise"""
    a = torch.rand([])
    noise = torch.rand_like(inputs)
    noise = noise.to(inputs.device)
    noise = 2 * a * epsilon * noise - a * epsilon
    noisy = inputs + noise
    inputs = torch.cat([inputs, noisy])
    labels = torch.cat([labels, labels])
    return inputs, labels


device = _get_device()


def main(args):  

    logging.basicConfig(filename=args.log, level=logging.DEBUG)

    if args.model_ckpt is None:
        ckpt = f"model/libri_model_raw_audio_{time.strftime('%Y%m%d%H%M')}.pt"

    else:
        ckpt = args.model_ckpt

    generator_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers
    }

    # Step 1: load data set
    data_resolver = LibriSpeechSpeakers(hp.data_root, hp.data_subset)

    train_data = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        subset="train",
        project_fs=hp.sr,
        wav_length=args.wav_length,
    )
    train_generator = torch.utils.data.DataLoader(train_data, **generator_params)

    val_data = LibriSpeech4SpeakerRecognition(
        root=hp.data_root,
        url=hp.data_subset,
        train_speaker_ratio=hp.train_speaker_ratio,
        train_utterance_ratio=hp.train_utterance_ratio,
        subset="val",
        project_fs=hp.sr,
        wav_length=args.wav_length,
    )
    val_generator = torch.utils.data.DataLoader(val_data, **generator_params)

    if args.model_type=='cnn':
        model = RawAudioCNN(num_class=data_resolver.get_num_speakers())
    elif args.model_type=='tdnn':
        model = TDNN(data_resolver.get_num_speakers())
    elif args.model_type=='double':
        cnn_audio = RawAudioCNN(num_class=data_resolver.get_num_speakers())
        cnn_spec = SpectrogramCNN(num_class=data_resolver.get_num_speakers())
        model = DoubleModelCNN(data_resolver.get_num_speakers(), cnn_audio, cnn_spec)
        if args.double_model_ckpt is not None:
            if args.cnn_audio_model_ckpt is None or args.cnn_spec_model_ckpt is None:
                model.cnn_audio = RawAudioCNN(num_class=data_resolver.get_num_speakers())
                model.cnn_spec = SpectrogramCNN(num_class=data_resolver.get_num_speakers())
            else:
                model.cnn_audio = torch.load(args.cnn_audio_model_ckpt)
                second_ckpt_model = torch.load(args.cnn_spec_model_ckpt)
                # save the state dict of the second model
                model.cnn_spec = SpectrogramCNN(num_class=data_resolver.get_num_speakers())
                model.cnn_spec.load_state_dict(second_ckpt_model.state_dict())
                if args.freeze_cnn:
                    model.freeze_cnns()


    else:
        logging.error('Please provide a valid model architecture type!')
        sys.exit(-1)
        
    print(model)
        
    if _is_cuda_available():
        model.to(device)
        logging.info(device)


    alr = ALR()
    
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9)
    elif args.optimizer=='adam':
        print()
        print('Using Adam optimizer\n')
        optimizer = Adam(model.parameters(), lr=1e-3, betas=(.5, .999))


    # Step 3: train
    model.train()
    batch_idx = 0
    loss_epoch = []
    acc_epoch = []
    for batch_data in tqdm(infinite_iter(train_generator), total=args.n_iters):
        batch_idx += 1
        inputs, labels = (x.to(device) for x in batch_data)
        model.train()
        if args.freeze_cnn:
            model.freeze_cnns()
        if args.epsilon > 0:
            inputs, labels = noise_augmenter(inputs, labels, args.epsilon)

        real_feature = model.encode(inputs)
        outputs = model.predict_from_embeddings(real_feature)
        class_loss = criterion(outputs, labels)

        if args.alr_weight > 0:
            input_adv = alr.get_adversarial_perturbations(model, inputs, labels)
            adv_feature = model.encode(input_adv)
            output_adv = model.predict_from_embeddings(adv_feature)
            alr_outputs = alr.get_alp(inputs, input_adv, outputs, output_adv, labels)
            alr_loss = alr_outputs[0].mean()
            loss = class_loss + args.alr_weight * alr_loss
            
            if args.cr_weight > 0:
                cr_loss = F.mse_loss(real_feature, adv_feature)
                loss += cr_loss
        else:
            loss = class_loss

        # Model computations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # training accuracy 
        acc_ = np.mean((torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy())
        loss_epoch.append(loss.item())
        acc_epoch.append(acc_)

        # validation accuracy
        message = f"It [{batch_idx}] train-loss: {loss.item():.4f} \ttrain-acc (batch): {acc_:.4f}"
        
        if args.alr_weight > 0:
            message += f"\talr={alr_loss.item():.4e} "

        if args.cr_weight > 0:
            message += f"\tcr={cr_loss.item():.4e} "
        
        print(message, end="\r")
        logging.info(message)

        # Checkpointing
        if batch_idx % args.save_every == 0:
            # log validation accuracy
            model.eval()
            val_acc = []
            for val_batch in val_generator:
                inputs, labels = (x.to(device) for x in val_batch)
                outputs = model(inputs)
                val_acc.append(np.mean((torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy()))
            val_acc = np.mean(val_acc)
            message = f"\nValidation accuracy: {val_acc}"
            print(message)
            logging.info(message)
            torch.save(model, ckpt + f"_{batch_idx}.tmp")
            torch.save(optimizer, ckpt + f"_{batch_idx}.optimizer.tmp")
            print()

        # Termination
        if batch_idx > args.n_iters:
            break
        # Termination -- if n_epochs are provided
        if args.n_epochs is not None:
            done_ = batch_idx//len(train_generator)
            if batch_idx%len(train_generator)==0:
                msg_ = f"Epoch {done_}: loss = {np.mean(loss_epoch)} acc = {np.mean(acc_epoch)}"
                print(msg_)
                logging.info(msg_) 
                loss_epoch=[]
                acc_epoch=[]
            if done_ >= args.n_epochs:
                msg_ = "Finishing training based on n_epochs provided..."
                print(msg_)
                logging.info(msg_)
                break

    logging.info("Finished Training")
    torch.save(model, ckpt)


def parse_args():
    name = "DoubleModelCNN"
    name="CNN_Vocoded_clean"
    parser = ArgumentParser("Speaker Classification model on LibriSpeech dataset", \
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-mn", "--model_name", type=str, default=name, help="Model name")
    parser.add_argument(
        "-m", "--model_ckpt", type=str, default=f"model/{name}", help="Model checkpoint")
    parser.add_argument(
        "-g", "--log", type=str, default=f"model/train_logs/train_{name}.log", help="Experiment log")
    parser.add_argument(
        "-mt", "--model_type", type=str, default='cnn', help="Model type: cnn or tdnn")
    parser.add_argument(
        "-l", "--wav_length", type=int, default=80000,
        help="Max length of waveform in a batch")
    parser.add_argument(
        "-n", "--n_iters", type=int, default=5000,
        help="Number of iterations for training"
    )
    parser.add_argument(
        "-ne", "--n_epochs", type=int, default=100,
        help="Number of epochs for training. Optional. Ignored if not provided."
    )
    parser.add_argument(
        "-s", "--save_every", type=int, default=500, help="Save after this number of gradient updates"
    )
    parser.add_argument(
        "-e", "--epsilon", type=float, default=0,
        help="Noise magnitude in data augmentation; set it to 0 to disable augmentation")
    parser.add_argument(
        "-w", "--alr_weight", type=float, default=0,
        help="Weight of the adversarial Lipschitz regularizer"
    )
    parser.add_argument(
        "-c", "--cr_weight", type=float, default=0,
        help="Weight of consistency regularizer"
    )
    parser.add_argument(
        "-opt", "--optimizer", type=str, default='adam',
        help="Optimizer: sgd, adam")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=128,
        help="Batch size")
    parser.add_argument(
        "-nw", "--num_workers", type=int, default=8,
        help="Number of workers related to pytorch data loader")
    parser.add_argument(
        "-dm", "--double_model_ckpt", type=bool, default=True,
        help="Checkpoint for double model"
    )
    parser.add_argument(
        "-ca", "--cnn_audio_model_ckpt", type=str, default="model/clean_4000_96.7.tmp",
        help="Checkpoint for cnn_audio model"
    )
    parser.add_argument(
        "-cs", "--cnn_spec_model_ckpt", type=str, default="model/CNN_Vocoded_clean_4000_94.8.tmp",
        help="Checkpoint for cnn_spec model"
    )
    parser.add_argument(
        "-fc", "--freeze_cnn", type=bool, default=True,
        help="Freeze the CNNs"
    )

    args = parser.parse_args()

    # check if model_cpkt is default or not
    model_name = args.model_name
    if model_name != name:
        if args.model_ckpt == f"model/{name}/{name}":
            args.model_ckpt = f"model/{model_name}/{model_name}"
        if args.log == f"model/train_logs/train_{name}.log":
            args.log = f"model/train_logs/train_{model_name}.log"

    if os.path.exists(args.model_ckpt):
        logging.error("Model checkpoint already exists. Please provide a new model checkpoint")
        sys.exit(-1)
    else:
        os.makedirs(os.path.dirname(args.model_ckpt), exist_ok=True)

    #clean log file
    with open(args.log, "w") as f:
        f.write("") 



    return args


if __name__ == "__main__":
    main(parse_args())
