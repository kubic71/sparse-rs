from torchvision import transforms
import pathlib
from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
from rs_attacks import RSAttack
import gvision_model
import argparse

def get_data_loader(path="dataset/Cats"):
    size = 299
    imagenet = ImageFolder(path,
                           transforms.Compose([
                            transforms.Resize(size),
                            transforms.CenterCrop(size),
                            transforms.ToTensor()
                           ])
                           )
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=1, shuffle=False, num_workers=1)
    return imagenet_loader

    # return np.array(x_test, dtype=np.float32), np.array(y_test)


parser = argparse.ArgumentParser()

parser.add_argument('--norm', type=str, choices=['L0', 'patches', 'frames',
            'patches_universal', 'frames_universal'], default='L0')

parser.add_argument('--k', default=2000, type=float)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--loss', type=str, default='margin')
parser.add_argument('--attack', type=str, default='rs_attack')
parser.add_argument('--n_queries', type=int, default=1000)
parser.add_argument('--targeted', action='store_true')
parser.add_argument('--target_class', type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--constant_schedule', action='store_true')
parser.add_argument('--save_dir', type=str, default='./results')

# Sparse-RS parameter
parser.add_argument('--p_init', type=float, default=.3)
parser.add_argument('--resample_period_univ', type=int)
parser.add_argument('--loc_update_period', type=int)

args = parser.parse_args()

args.resample_loc = args.resample_period_univ
args.update_loc_period = args.loc_update_period

param_run = '{}_{}_{}_1_{}_nqueries_{:.0f}_pinit_{:.2f}_loss_{}_k_{:.0f}_targeted_{}_targetclass_{}_seed_{:.0f}'.format(
            args.attack, args.norm, "gvision", 1, args.n_queries, args.p_init,
            args.loss, args.k, args.targeted, args.target_class, args.seed)

logsdir = '{}/logs_{}_{}'.format(args.save_dir, args.attack, args.norm)
pathlib.Path(logsdir).mkdir(parents=True, exist_ok=True)

test_img = 'cat'
if test_img == 'cat':
    loader = get_data_loader(path="dataset/Cats")
    label_set = ['cat']

elif test_img == 'shark':
    loader = get_data_loader(path="dataset/Shark")
    label_set = ["Shark", "Fin", "Water", "Fish", "Carcharhiniformes", "Lamnidae", "Lamniformes"]

testiter = iter(loader)
x_test, _ = next(testiter)
y_test = torch.zeros(x_test.shape[0])


# test_img = 'shark'

save_loc = f"experiments/{args.norm}/{test_img}_k={args.k}_nqueries={args.n_queries}_constant_schedule={args.constant_schedule}_p-init={args.p_init}"
model = gvision_model.GVisionModel(exp_name=param_run, save_location=save_loc, correct_labelset=label_set)
adversary = RSAttack(model, norm=args.norm, eps=int(args.k), verbose=True, n_queries=args.n_queries,
            p_init=args.p_init, log_path='{}/log_run_{}_{}.txt'.format(logsdir, str(datetime.now())[:-7], param_run),
            loss=args.loss, targeted=args.targeted, seed=args.seed, constant_schedule=args.constant_schedule,
            data_loader=None, resample_loc=args.resample_loc)




qr_curr, adv = adversary.perturb(x_test, y_test)
print(qr_curr, adv)

