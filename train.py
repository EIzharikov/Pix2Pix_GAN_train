"""
Generative Adversarial Networks: pix2pix & CycleGAN
Style Transfer on Facade Dataset

Part 1: pix2pix (paired image-to-image translation) - Isola et al., 2016
Part 2: CycleGAN (unpaired image-to-image translation) - Zhu et al., 2017

Usage: python train.py
Results saved to results/ folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import urllib.request
import zipfile
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

os.makedirs('results', exist_ok=True)

# ==================== DATASET ====================

IMG_SIZE = 256
BATCH_SIZE = 4
BATCH_SIZE_CYC = 1
SKIP_PIX2PIX = False
SKIP_CYCLEGAN = False
LR = 2e-4; BETAS = (0.5, 0.999); LAMBDA_L1 = 100
NUM_EPOCHS_P2P = 100

data_url = 'https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip'
data_dir = 'facades_data'

if not os.path.exists(data_dir):
    print('Downloading dataset...')
    urllib.request.urlretrieve(data_url, 'facades.zip')
    with zipfile.ZipFile('facades.zip', 'r') as z:
        z.extractall(data_dir)
    os.remove('facades.zip')
    print('Done!')
else:
    print('Dataset already exists')

image_files = sorted(glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
label_files = sorted(glob(os.path.join(data_dir, '**', '*.png'), recursive=True))
print(f'Found {len(image_files)} images and {len(label_files)} labels')

class FacadesDataset(Dataset):
    def __init__(self, image_files, label_files, img_size=IMG_SIZE, augment=True):
        self.image_files = image_files
        self.label_files = label_files
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        real_img = Image.open(self.image_files[idx]).convert('RGB')
        seg_mask = Image.open(self.label_files[idx]).convert('RGB')
        if self.augment and np.random.random() > 0.5:
            real_img = real_img.transpose(Image.FLIP_LEFT_RIGHT)
            seg_mask = seg_mask.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(seg_mask), self.transform(real_img)

n = len(image_files)
split = int(0.9 * n)
train_dataset = FacadesDataset(image_files[:split], label_files[:split], augment=True)
val_dataset = FacadesDataset(image_files[split:], label_files[split:], augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

# Save dataset samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    img = Image.open(image_files[i])
    label = Image.open(label_files[i])
    axes[0, i].imshow(img); axes[0, i].set_title(f'Real photo {i}'); axes[0, i].axis('off')
    axes[1, i].imshow(label); axes[1, i].set_title(f'Segmentation mask {i}'); axes[1, i].axis('off')
plt.suptitle('Facades Dataset: Real Photos and Segmentation Masks', fontsize=14)
plt.tight_layout(); plt.savefig('results/01_dataset_samples.png', dpi=150); plt.close()
print('Saved dataset samples')

# ==================== PIX2PIX MODELS ====================

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if normalize: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch)]
        if dropout: layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        return torch.cat([self.model(x), skip], dim=1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.down1 = UNetDown(in_ch, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)
        self.up1 = UNetUp(512, 512, dropout=True)
        self.up2 = UNetUp(1024, 512, dropout=True)
        self.up3 = UNetUp(1024, 512, dropout=True)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, stride=2, padding=1), nn.Tanh())
    def forward(self, x):
        d1=self.down1(x); d2=self.down2(d1); d3=self.down3(d2); d4=self.down4(d3)
        d5=self.down5(d4); d6=self.down6(d5); d7=self.down7(d6); d8=self.down8(d7)
        u1=self.up1(d8,d7); u2=self.up2(u1,d6); u3=self.up3(u2,d5); u4=self.up4(u3,d4)
        u5=self.up5(u4,d3); u6=self.up6(u5,d2); u7=self.up7(u6,d1)
        return self.final(u7)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        def block(ic, oc, norm=True):
            l = [nn.Conv2d(ic, oc, 4, stride=2, padding=1)]
            if norm: l.append(nn.BatchNorm2d(oc))
            l.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*l)
        self.model = nn.Sequential(
            block(in_ch, 64, norm=False), block(64, 128), block(128, 256),
            nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(512, 1, 4, padding=1))
    def forward(self, a, b):
        return self.model(torch.cat([a, b], dim=1))

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)

def denorm(x):
    return ((x + 1) / 2).clamp(0, 1)

# ==================== TRAIN PIX2PIX ====================
if not SKIP_PIX2PIX:
    print('\n' + '='*60)
    print('PART 1: Training pix2pix')
    print('='*60)

    generator = GeneratorUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    init_weights(generator); init_weights(discriminator)

    opt_G = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    opt_D = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)
    crit_GAN = nn.BCEWithLogitsLoss()
    crit_L1 = nn.L1Loss()

    print(f'Generator params: {sum(p.numel() for p in generator.parameters()):,}')
    print(f'Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}')

    G_losses, D_losses = [], []
    t0 = time.time()

    for epoch in range(NUM_EPOCHS_P2P):
        eG, eD = 0, 0
        for masks, reals in train_loader:
            masks, reals = masks.to(device), reals.to(device)
            fakes = generator(masks)

            # Train D
            opt_D.zero_grad()
            pred_real = discriminator(masks, reals)
            pred_fake = discriminator(masks, fakes.detach())
            loss_D = (crit_GAN(pred_real, torch.ones_like(pred_real)) + crit_GAN(pred_fake, torch.zeros_like(pred_fake))) * 0.5
            loss_D.backward(); opt_D.step()

            # Train G
            opt_G.zero_grad()
            pred_fake = discriminator(masks, fakes)
            loss_G = crit_GAN(pred_fake, torch.ones_like(pred_fake)) + crit_L1(fakes, reals) * LAMBDA_L1
            loss_G.backward(); opt_G.step()

            eG += loss_G.item(); eD += loss_D.item()

        G_losses.append(eG/len(train_loader)); D_losses.append(eD/len(train_loader))
        if (epoch+1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS_P2P}] G: {G_losses[-1]:.4f} D: {D_losses[-1]:.4f} ({elapsed:.0f}s)')

    print(f'pix2pix training done in {time.time()-t0:.0f}s')

    # Save pix2pix loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(G_losses, label='Generator'); plt.plot(D_losses, label='Discriminator')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('pix2pix Training Losses')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('results/02_pix2pix_losses.png', dpi=150); plt.close()

    # Save pix2pix results
    generator.eval()
    masks_v, reals_v = next(iter(val_loader))
    masks_v = masks_v[:4].to(device); reals_v = reals_v[:4]
    with torch.no_grad():
        fakes_v = generator(masks_v).cpu()
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        axes[0,i].imshow(denorm(masks_v[i].cpu()).permute(1,2,0).numpy()); axes[0,i].set_title('Input mask'); axes[0,i].axis('off')
        axes[1,i].imshow(denorm(fakes_v[i]).permute(1,2,0).numpy()); axes[1,i].set_title('Generated'); axes[1,i].axis('off')
        axes[2,i].imshow(denorm(reals_v[i]).permute(1,2,0).numpy()); axes[2,i].set_title('Ground Truth'); axes[2,i].axis('off')
    plt.suptitle('pix2pix Results on Validation Set', fontsize=16)
    plt.tight_layout(); plt.savefig('results/03_pix2pix_results.png', dpi=150); plt.close()
    generator.train()
    print('Saved pix2pix results')

# ==================== CYCLEGAN MODELS ====================

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch))
    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_res=6):
        super().__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, 64, 7, bias=False), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        c = 64
        for _ in range(2):
            model += [nn.Conv2d(c, c*2, 3, stride=2, padding=1, bias=False), nn.InstanceNorm2d(c*2), nn.ReLU(inplace=True)]
            c *= 2
        for _ in range(n_res):
            model += [ResidualBlock(c)]
        for _ in range(2):
            model += [nn.ConvTranspose2d(c, c//2, 3, stride=2, padding=1, output_padding=1, bias=False), nn.InstanceNorm2d(c//2), nn.ReLU(inplace=True)]
            c //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class CycleDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        def blk(ic, oc, norm=True):
            l = [nn.Conv2d(ic, oc, 4, stride=2, padding=1)]
            if norm: l.append(nn.InstanceNorm2d(oc))
            l.append(nn.LeakyReLU(0.2, inplace=True))
            return l
        self.model = nn.Sequential(*blk(in_ch,64,False), *blk(64,128), *blk(128,256), *blk(256,512),
            nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(512, 1, 4, padding=1))
    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size; self.data = []
    def push_and_pop(self, data):
        result = []
        for el in data:
            el = el.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(el); result.append(el)
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, self.max_size)
                result.append(self.data[idx].clone()); self.data[idx] = el
            else:
                result.append(el)
        return torch.cat(result, 0)

# ==================== TRAIN CYCLEGAN ====================

if not SKIP_CYCLEGAN:
    print('\n' + '='*60)
    print('PART 2: Training CycleGAN')
    print('='*60)

    G_XY = CycleGenerator().to(device)
    F_YX = CycleGenerator().to(device)
    D_X = CycleDiscriminator().to(device)
    D_Y = CycleDiscriminator().to(device)
    init_weights(G_XY); init_weights(F_YX); init_weights(D_X); init_weights(D_Y)

    LAMBDA_CYC = 10.0; LAMBDA_ID = 5.0
    NUM_EPOCHS_CYC = 100
    train_loader_cyc = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    opt_G_cyc = optim.Adam(list(G_XY.parameters()) + list(F_YX.parameters()), lr=LR, betas=BETAS)
    opt_DX = optim.Adam(D_X.parameters(), lr=LR, betas=BETAS)
    opt_DY = optim.Adam(D_Y.parameters(), lr=LR, betas=BETAS)

    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - NUM_EPOCHS_CYC // 2) / (NUM_EPOCHS_CYC // 2 + 1)
    sched_G = optim.lr_scheduler.LambdaLR(opt_G_cyc, lr_lambda)
    sched_DX = optim.lr_scheduler.LambdaLR(opt_DX, lr_lambda)
    sched_DY = optim.lr_scheduler.LambdaLR(opt_DY, lr_lambda)

    crit_mse = nn.MSELoss(); crit_cyc = nn.L1Loss(); crit_id = nn.L1Loss()
    buf_X = ReplayBuffer(); buf_Y = ReplayBuffer()

    total_params = sum(p.numel() for p in G_XY.parameters()) + sum(p.numel() for p in F_YX.parameters()) + sum(p.numel() for p in D_X.parameters()) + sum(p.numel() for p in D_Y.parameters())
    print(f'Total CycleGAN params: {total_params:,}')

    hist = {'G':[], 'DX':[], 'DY':[], 'cyc':[], 'id':[]}
    t0 = time.time()

    for epoch in range(NUM_EPOCHS_CYC):
        eG, eDX, eDY, eCyc, eId = 0, 0, 0, 0, 0
        for real_X, real_Y in train_loader_cyc:
            real_X, real_Y = real_X.to(device), real_Y.to(device)

            # Train Generators
            opt_G_cyc.zero_grad()
            id_Y = G_XY(real_Y); loss_idY = crit_id(id_Y, real_Y) * LAMBDA_ID
            id_X = F_YX(real_X); loss_idX = crit_id(id_X, real_X) * LAMBDA_ID

            fake_Y = G_XY(real_X)
            pY = D_Y(fake_Y); valid_Y = torch.ones_like(pY)
            loss_G_XY = crit_mse(pY, valid_Y)

            fake_X = F_YX(real_Y)
            pX = D_X(fake_X); valid_X = torch.ones_like(pX)
            loss_G_YX = crit_mse(pX, valid_X)

            rec_X = F_YX(fake_Y); loss_cyc_X = crit_cyc(rec_X, real_X) * LAMBDA_CYC
            rec_Y = G_XY(fake_X); loss_cyc_Y = crit_cyc(rec_Y, real_Y) * LAMBDA_CYC

            loss_G_total = loss_G_XY + loss_G_YX + loss_cyc_X + loss_cyc_Y + loss_idX + loss_idY
            loss_G_total.backward(); opt_G_cyc.step()

            # Train D_Y
            opt_DY.zero_grad()
            fake_Y_buf = buf_Y.push_and_pop(fake_Y.detach())
            fake_lbl_Y = torch.zeros_like(D_Y(fake_Y_buf))
            loss_DY = (crit_mse(D_Y(real_Y), valid_Y) + crit_mse(D_Y(fake_Y_buf), fake_lbl_Y)) * 0.5
            loss_DY.backward(); opt_DY.step()

            # Train D_X
            opt_DX.zero_grad()
            fake_X_buf = buf_X.push_and_pop(fake_X.detach())
            fake_lbl_X = torch.zeros_like(D_X(fake_X_buf))
            loss_DX = (crit_mse(D_X(real_X), valid_X) + crit_mse(D_X(fake_X_buf), fake_lbl_X)) * 0.5
            loss_DX.backward(); opt_DX.step()

            eG += loss_G_total.item(); eDX += loss_DX.item(); eDY += loss_DY.item()
            eCyc += (loss_cyc_X + loss_cyc_Y).item(); eId += (loss_idX + loss_idY).item()

        sched_G.step(); sched_DX.step(); sched_DY.step()
        nn = len(train_loader_cyc)
        hist['G'].append(eG/nn); hist['DX'].append(eDX/nn); hist['DY'].append(eDY/nn)
        hist['cyc'].append(eCyc/nn); hist['id'].append(eId/nn)

        if (epoch+1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS_CYC}] G:{hist["G"][-1]:.3f} DX:{hist["DX"][-1]:.3f} DY:{hist["DY"][-1]:.3f} Cyc:{hist["cyc"][-1]:.3f} ({elapsed:.0f}s)')

    print(f'CycleGAN training done in {time.time()-t0:.0f}s')
    torch.save(G_XY.state_dict(), 'results/G_XY.pth')
    torch.save(F_YX.state_dict(), 'results/F_YX.pth')
    # Save CycleGAN loss plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist['G'], label='Generator Total'); axes[0].set_title('Generator Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist['DX'], label='D_X'); axes[1].plot(hist['DY'], label='D_Y'); axes[1].set_title('Discriminator Losses'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(hist['cyc'], label='Cycle'); axes[2].plot(hist['id'], label='Identity'); axes[2].set_title('Cycle & Identity Losses'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle('CycleGAN Training Losses', fontsize=14)
    plt.tight_layout(); plt.savefig('results/04_cyclegan_losses.png', dpi=150); plt.close()

    # Save CycleGAN results
    G_XY.eval(); F_YX.eval()
    masks_v, reals_v = next(iter(val_loader))
    masks_v = masks_v[:4].to(device); reals_v = reals_v[:4].to(device)
    with torch.no_grad():
        fake_photos = G_XY(masks_v)
        rec_masks = F_YX(fake_photos)
        fake_masks = F_YX(reals_v)
        rec_photos = G_XY(fake_masks)

    # Forward cycle: mask -> photo -> mask
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ['Input Mask', 'Generated Photo', 'Recovered Mask', 'Real Photo']
    data = [masks_v, fake_photos, rec_masks, reals_v]
    for r in range(4):
        for c in range(4):
            axes[r,c].imshow(denorm(data[r][c]).cpu().permute(1,2,0).numpy())
            if c == 0: axes[r,c].set_ylabel(titles[r], fontsize=12)
            axes[r,c].axis('off')
    plt.suptitle('CycleGAN: Mask -> Photo -> Mask (Cycle)', fontsize=16)
    plt.tight_layout(); plt.savefig('results/05_cyclegan_forward.png', dpi=150); plt.close()

    # Reverse cycle: photo -> mask -> photo
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ['Real Photo', 'Generated Mask', 'Recovered Photo', 'Real Mask']
    data = [reals_v, fake_masks, rec_photos, masks_v]
    for r in range(4):
        for c in range(4):
            axes[r,c].imshow(denorm(data[r][c]).cpu().permute(1,2,0).numpy())
            if c == 0: axes[r,c].set_ylabel(titles[r], fontsize=12)
            axes[r,c].axis('off')
    plt.suptitle('CycleGAN: Photo -> Mask -> Photo (Cycle)', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/06_cyclegan_reverse.png', dpi=150); plt.close()

# ==================== COMPARISON ====================

# ==================== COMPARISON ====================
if not SKIP_PIX2PIX and not SKIP_CYCLEGAN:
    generator.eval(); G_XY.eval()
    masks_v, reals_v = next(iter(val_loader))
    masks_v = masks_v[:4].to(device); reals_v = reals_v[:4]
    with torch.no_grad():
        p2p_out = generator(masks_v).cpu()
        cyc_out = G_XY(masks_v).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    titles = ['Input Mask', 'pix2pix Output', 'CycleGAN Output', 'Ground Truth']
    data = [masks_v.cpu(), p2p_out, cyc_out, reals_v]
    for r in range(4):
        for c in range(4):
            axes[r,c].imshow(denorm(data[r][c]).permute(1,2,0).numpy())
            if c == 0: axes[r,c].set_ylabel(titles[r], fontsize=12)
            axes[r,c].axis('off')
    plt.suptitle('Comparison: pix2pix vs CycleGAN', fontsize=16)
    plt.tight_layout(); plt.savefig('results/07_comparison.png', dpi=150); plt.close()

print('\n' + '='*60)
print('All results saved to results/ folder!')
print('='*60)
print('Files:')
for f in sorted(os.listdir('results')):
    print(f'  {f}')