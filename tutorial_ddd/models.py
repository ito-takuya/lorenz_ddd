import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


class MLP(nn.Module):
    def __init__(self, ndim=16, nlayers=2, input_size=3, output_size=3):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.ndim = ndim
        self.nlayers = nlayers
        assert(nlayers>0)

        # input layer has linear mapping + relu
        self.input_layer = nn.Sequential(*[nn.Linear(input_size, ndim),nn.ReLU()])
        # output mapping only has linear mapping
        self.output_layer = nn.Linear(ndim,output_size)

        hidden = []
        for _ in range(nlayers):
            hidden.append(nn.Linear(ndim,ndim))
            hidden.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden(x)
        x = self.output_layer(x)
        return x

class Transformer(torch.nn.Module):
    """
    base transformer model
    Inputs are the tokens of the LST grid
    """
    def __init__(self,
                 input_dim=1, # dimension of input tokens (time series so just 1)
                 output_dim=1, # mask pretraining so just 1
                 nhead=1,
                 nlayers=1,
                 embedding_dim=16,
                 ):
        super(Transformer,self).__init__()

        # Define general parameters
        self.input_dim = input_dim
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.nhead = nhead

        self.w_embed = torch.nn.Linear(self.input_dim,self.embedding_dim) 

        self.blocks = torch.nn.Sequential()
        for _ in range(nlayers):
            self.blocks.append(TransformerBlock(embedding_dim,
                                                nhead=self.nhead
                                                ))

        self.w_out = torch.nn.Linear(self.embedding_dim,self.output_dim)
        
    def forward(self, X):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        embedding = self.w_embed(X)

        ####
        transformer_out = checkpoint_sequential(self.blocks, segments = len(self.blocks), input = embedding)
        outputs = self.w_out(transformer_out[:,:,:])

        return outputs

class TransformerBlock(torch.nn.Module):
    """
    Transformer block
    """
    def __init__(self,
                 embedding_dim,
                 nhead=1,
                 ):
        super(TransformerBlock,self).__init__()

        # Define general parameters
        self.nhead = nhead
        self.embedding_dim = embedding_dim
        
        # positional encoding
        self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,batch_first=True)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.embedding_dim*4),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim*4,self.embedding_dim),
            torch.nn.GELU()
        )

        # # layer norm; 1st is after attention (embedding dim); 2nd is after RNN 
        self.layernorm0 = torch.nn.LayerNorm(self.embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(self.embedding_dim)

    def forward(self, embedding):
        """
        Run a forward pass of a trial by input_elements matrix
        For each time window, pass each 
        input (Tensor): batch x seq_length x dim_input x time
        """
        attn_outputs, attn_out_weights = self.selfattention(embedding, embedding, embedding, need_weights=False)
        # attn_outputs = self.layernorm0(attn_outputs+embedding) # w resid connection
        transformer_out = self.mlp(attn_outputs)
        # transformer_out = self.layernorm1(transformer_out+attn_outputs) # w resid connection
        return transformer_out

