import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))   

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        return self.scale * x / rms


class MLP_block(nn.Module):
    def __init__(self, dim1=7, dim2=96, d_model=512, d_out=512, dropout=0.1):
        super(MLP_block, self).__init__()
        self.mlp = nn.Sequential(
                RMSNorm([dim1, dim2]),
                torch.nn.Linear(dim2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                torch.nn.Linear(d_model, d_out),
                nn.Dropout(dropout)
            )
    
    def forward(self, x):
        return self.mlp(x)


class Multiscale_MLP(nn.Module):
    def __init__(self, configs):
        super(Multiscale_MLP, self).__init__()
        self.freq = configs.freq
        if self.freq == 't':
            self.enc_in = configs.enc_in+5
        elif self.freq == 'h':
            self.enc_in = configs.enc_in+4
        elif self.freq == 'd':
            self.enc_in = configs.enc_in+3
        else:
            self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model


        self.temporal_function = configs.temporal_function
        if self.temporal_function == 'down':
            self.kernel = configs.patch
            self.layers = len(self.kernel)
            self.down_method = configs.down_method
            if self.down_method == 'conv':
                self.down_sample = torch.nn.ModuleList([
                    torch.nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                        kernel_size=self.kernel[i],stride=self.kernel[i],padding=0, groups=self.enc_in)
                    for i in range(self.layers)])

            self.mlp = torch.nn.ModuleList([MLP_block(dim1=self.enc_in, dim2=self.seq_len//self.kernel[i], d_model=configs.d_model, d_out=configs.seq_len, dropout=configs.dropout)
                for i in range(self.layers)])

            self.rnn_ffn = MLP_block(dim1=self.enc_in, dim2=self.seq_len, d_model=configs.d_model, d_out=self.seq_len, dropout=configs.dropout)



        if self.temporal_function == 'patch':
            self.core = 64
            self.patch = configs.patch
            self.patch_num = [self.seq_len // i for i in self.patch]
            self.decomp = torch.nn.ModuleList([series_decomp(i+1) for i in self.patch])
            self.mlp = MLP_block(dim1=self.enc_in, dim2=self.seq_len, d_model=configs.d_model, d_out=self.seq_len,  dropout=configs.dropout)
            
            self.t_intra = torch.nn.ModuleList([
                MLP_block(dim1=self.patch_num[i], dim2=self.patch[i], d_model=self.core, d_out=self.patch[i], dropout=configs.dropout)
                for i in range(len(self.patch))])
            self.t_inter = torch.nn.ModuleList([
                MLP_block(dim1=self.patch[i], dim2=self.patch_num[i], d_model=self.core, d_out=self.patch_num[i], dropout=configs.dropout)
                for i in range(len(self.patch))])
                
            self.s_intra = torch.nn.ModuleList([
                MLP_block(dim1=self.patch_num[i], dim2=self.patch[i], d_model=self.core, d_out=self.patch[i], dropout=configs.dropout)
                for i in range(len(self.patch))])
            self.s_inter = torch.nn.ModuleList([
                MLP_block(dim1=self.patch[i], dim2=self.patch_num[i], d_model=self.core, d_out=self.patch_num[i], dropout=configs.dropout)
                for i in range(len(self.patch))])

            self.rnn_ffn = MLP_block(dim1=self.enc_in, dim2=configs.seq_len, d_model=configs.d_model, d_out=configs.seq_len, dropout=configs.dropout)
        
        self.linear = torch.nn.Conv1d(in_channels=configs.seq_len, out_channels=configs.d_model,
                        kernel_size=1,stride=1,padding=0,groups=1)

    def forward(self, x, x_mark_enc):

        B, L, C = x.shape
        if self.temporal_function == 'patch':
            add = torch.zeros([B, L, C], device=x.device)
            for i in range(len(self.patch)):
                if self.patch[i] == 1:
                    add = self.mlp(x.permute(0,2,1)).permute(0,2,1)
                else:
                    season, x_group = self.decomp[i](x)
                    x_group = x_group.permute(0,2,1)
                    x_group = x_group.reshape(B, C, self.patch_num[i], self.patch[i])
                    x_group = x_group + self.t_intra[i](x_group)
                    x_group = x_group.permute(0,1,3,2)
                    x_group = x_group + self.t_inter[i](x_group)
                    x_group = x_group.permute(0,1,3,2).reshape(B, C, -1).permute(0,2,1)
                    season = season.permute(0,2,1)
                    season = season.reshape(B, C, self.patch_num[i], self.patch[i])
                    season = season + self.s_intra[i](season)
                    season = season.permute(0,1,3,2)
                    season = season + self.s_inter[i](season)
                    season = season.permute(0,1,3,2).reshape(B, C, -1).permute(0,2,1)
                    add = add +  (x_group + season)
            x =  x + self.rnn_ffn(add.permute(0,2,1)).permute(0,2,1)

        if self.temporal_function == 'down':
            add = torch.zeros([B, C, L], device=x.device)
            for i in range(self.layers):
                tmp = x.permute(0,2,1)
                if self.down_method == 'conv':
                    tmp = self.down_sample[i](tmp)
                else:
                    tmp = torch.nn.AvgPool1d(kernel_size=self.kernel[i])(tmp) + torch.nn.MaxPool1d(kernel_size=self.kernel[i])(tmp)
                add = add + self.mlp[i](tmp) 
            x = x + self.rnn_ffn(add).permute(0,2,1)
        
        x = self.linear(x)
        
        return x



class ResBlock_RNN2(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock_RNN2, self).__init__()
        self.d_model = configs.d_model

        if configs.freq == 't':
            if configs.n_patch == -1:
                self.n_patch = int(math.sqrt(configs.enc_in+5)) 
            else:
                self.n_patch = configs.n_patch
            self.c_patch = (configs.enc_in+5) // self.n_patch + 1
        elif configs.freq == 'h':
            if configs.n_patch == -1:
                self.n_patch = int(math.sqrt(configs.enc_in+4)) 
            else:
                self.n_patch = configs.n_patch
            self.c_patch = (configs.enc_in+4) // self.n_patch + 1
        elif configs.freq == 'd':
            if configs.n_patch == -1:
                self.n_patch = int(math.sqrt(configs.enc_in+3)) 
            else:
                self.n_patch = configs.n_patch
            self.c_patch = (configs.enc_in+3) // self.n_patch + 1
        else:
            if configs.n_patch == -1:
                self.n_patch = int(math.sqrt(configs.enc_in)) 
            else:
                self.n_patch = configs.n_patch
            self.c_patch = configs.enc_in // self.n_patch + 1

        self.linear = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
        )

        self.norm_lstm = torch.nn.LayerNorm(self.d_model)
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear = nn.Sequential( 
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )

        self.lstm2 = torch.nn.GRU(input_size=self.d_model, hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear2 = nn.Sequential( 
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )
        self.linear2 = torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1)

        self.linear3 = torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1)


    def forward(self, x):
        batch, seq, channel = x.shape


        x = torch.cat((x, x[:,:,:(self.c_patch*self.n_patch-channel)]), dim=-1)
        x = x.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        residual = x
        x = x.reshape(batch*self.n_patch, self.d_model, self.c_patch).permute(0,2,1)
        out1, intra = self.lstm(self.norm_lstm(x))
        out1 = out1[:,:,:self.d_model] + out1[:,:,-self.d_model:]
        out1 = self.lstm_linear(out1.permute(0,2,1))
        out1 = out1.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        intra = torch.sum(intra.permute(1,0,2), dim=1, keepdim=True).reshape(batch, self.n_patch, self.d_model)
        inter, h = self.lstm2(intra) 
        inter = inter[:,:,:self.d_model] + inter[:,:,-self.d_model:]
        inter = self.lstm_linear2(inter.permute(0,2,1)).unsqueeze(-2)
        out = residual +  self.linear2(torch.mul(self.linear(residual), out1 + self.linear3(torch.mul(out1, inter) )) )
        out = out.reshape(batch, self.d_model, self.c_patch*self.n_patch).contiguous()[:,:,:channel]


        return out





class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.freq = configs.freq
        self.d_model = configs.d_model


        self.linear_trans = nn.Linear(configs.seq_len, configs.d_model)

        self.multiscale_mlp = Multiscale_MLP(configs)
        self.model = nn.ModuleList([ResBlock_RNN2(configs, seq_len=self.seq_len, groups=configs.d_model // int(math.pow(2,i)))
                                    for i in range(configs.e_layers)])

        self.projection = nn.Linear(configs.d_model, configs.pred_len)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        if self.freq == 't' or self.freq == 'h' or self.freq == 'd':
            x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc = self.multiscale_mlp(x_enc, x_mark_enc)
        
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)

        enc_out = self.projection((x_enc).transpose(1, 2)).transpose(1, 2)
        
        if self.use_norm:
            enc_out = enc_out  * stdev + means

        return enc_out[:,:,:self.enc_in]



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]   # [B, L, D]


