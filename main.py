from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dataloader import *
from nets.mgn import MGN_Net
from utils.eval_metrics import segment_level, event_level


def train(args, model, train_loader, optimizer, criterion, criterion_ce, epoch):
    model.train()

    for batch_idx, sample in enumerate(train_loader):

        global_step = batch_idx + len(train_loader) * epoch

        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')

        optimizer.zero_grad()
        aud_cls_prob, vis_cls_prob, output, a_prob, v_prob, a_frame_prob, v_frame_prob = model(audio, video, video_st)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        # label smoothing
        a = 1.0
        v = 0.9 
        Pa = a * target + (1 - a) * 0.5
        Pv = v * target + (1 - v) * 0.5

        cls_target = torch.arange(0,25).long().cuda()

        # individual guided learning
        loss_av = criterion(output, target)
        loss_a = criterion(a_prob, Pa)
        loss_v = criterion(v_prob, Pv)

        # class aware prediction
        loss_cls_aud = criterion_ce(aud_cls_prob, cls_target)
        loss_cls_vis = criterion_ce(vis_cls_prob, cls_target)

        loss =  loss_cls_aud + loss_cls_vis + loss_av + loss_a + loss_v

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss av: {:.6f}\tLoss a: {:.6f}\tLoss v: {:.6f}\tLoss cls_aud: {:.6f}\tLoss cls_vis: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), loss_av.item(), loss_a.item(), loss_v.item(), loss_cls_aud.item(), loss_cls_vis.item()))


def eval(model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'),sample['video_st'].to('cuda'), sample['label'].to('cuda')
            _, _, output, a_prob, v_prob, a_frame_prob, v_frame_prob  = model(audio, video, video_st)

            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)

            Pa = a_frame_prob[0, :, :].cpu().detach().numpy()
            Pv = v_frame_prob[0, :, :].cpu().detach().numpy()

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)         # [10, 25]
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)         # [10, 25]

            # extract audio GT labels
            GT_a = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num >0:
                for i in range(num):

                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            GT_v =np.zeros((25, 10))

            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v                 # [25, 10]

            # obtain prediction matrices
            SO_a = np.transpose(Pa)             # [25, 10]
            SO_v = np.transpose(Pv)             # [25, 10]
            SO_av = SO_a * SO_v                 # [25, 10]

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    print('Audio Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_a))))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_v))))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_av))))

    avg_type = (100 * np.mean(np.array(F_seg_av))+100 * np.mean(np.array(F_seg_a))+100 * np.mean(np.array(F_seg_v)))/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_av))))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))

    return avg_type, avg_type_event

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Multimodal Grouping Network for AVVP')
    parser.add_argument(
        "--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument(
        "--model", type=str, default='MGN_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='MGN_Net',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    
    # Model settings
    parser.add_argument(
        "--unimodal_assign", type=str, default='soft', help="type of unimodal grouping assignment")
    parser.add_argument(
        "--crossmodal_assign", type=str, default='soft', help="type of crossmodal grouping assignment")

    parser.add_argument(
        '--dim', type=int, default=128, help='dimensionality of features')
    parser.add_argument(
        '--depth_aud', type=int, default=3, help='depth of audio transformers')
    parser.add_argument(
        '--depth_vis', type=int, default=3, help='depth of visual transformers')
    parser.add_argument(
        '--depth_av', type=int, default=6, help='depth of audio-visual transformers')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    if args.model == 'MGN_Net':
        model = MGN_Net(args).to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        criterion = nn.BCELoss()
        criterion_ce = nn.CrossEntropyLoss()
        best_F = 0
        
        for epoch in range(0, args.epochs):
            train(args, model, train_loader, optimizer, criterion, criterion_ce, epoch)
            scheduler.step()
            F, F_event = eval(model, val_loader, args.label_val)

            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

        # testing
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        eval(model, test_loader, args.label_test)

    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([
                ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        eval(model, test_loader, args.label_val)
    else:
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
                                               ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        eval(model, test_loader, args.label_test)
if __name__ == '__main__':
    main()