import torch
from model.encoder_protein_atom import Pocket_encoder
from model.decoder import GPTDecoder

class HoS_Gen(torch.nn.Module):
    def __init__(self, train_config, encoder_config, decoder_config, device):
        super(HoS_Gen, self).__init__()

        self.train_config=train_config
        self.encoder_config=encoder_config
        self.decoder_config=decoder_config

        self.encoder=Pocket_encoder(encoder_config,device)
        self.decoder=GPTDecoder(decoder_config)




    def forward(self, data, hgnn_data, smiles, lengths=None):

        Pocket_pre, all_features=self.encoder(data, hgnn_data)

        logits, loss, attn_maps = self.decoder(smiles, Pocket_pre, all_features, lengths)


        return logits, loss
    
    def sample_from_pocket(self, data, hgnn_data, smiles, lengths=None):
        Pocket_pre, all_features=self.encoder(data, hgnn_data)    # [16,256]   [batch_size，embedingg_len]

        logits, loss, attn_maps = self.decoder.conditioned_sample(smiles, Pocket_pre, all_features, lengths)

        return logits

 


