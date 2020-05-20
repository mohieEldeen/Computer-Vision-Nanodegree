import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.embedded = nn.Embedding(vocab_size,embed_size)
        
        self.lstm = nn.LSTM( embed_size , hidden_size , num_layers , batch_first=True , dropout = 0)
          
        self.fc1 = nn.Linear( hidden_size  , vocab_size)
        
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        
    
    def forward(self, features, captions):
        
        self.features = features.to(self.device)
        
        self.captions = captions.to(self.device)
        
        self.batch_size = self.captions.size()[0]
        
        
        self.captions = self.captions[: , :-1]
        
        self.input = self.embedded(self.captions)
        
        self.features = self.features.reshape( ( self.features.size()[0] , 1 , self.features.size()[1] ) )
        
        self.input = torch.cat( ( self.features , self.input ) , dim=1 )
            
        
        out,_ = self.lstm( self.input )
        
        out = self.fc1(out)
        
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        self.input_features= inputs.to(self.device)
        
        self.batch_size = self.input_features.size()[0]
        
        h0 = torch.zeros(( self.batch_size , 1 , self.hidden_size ) ,device=self.device)
        c0 = torch.zeros(( self.batch_size, 1 , self.hidden_size ) ,device=self.device)
        
        hidden = (h0,c0)
        
        output_words=list()
        
        
        for i in range (max_len) :
            
            out,hidden = self.lstm( self.input_features , hidden)
            
            out = self.fc1 (out)
            
            word_idx = torch.argmax(out,dim=2)
            
            #print( int( word_idx.item() ) )
            
            output_words.append( int( word_idx.item() ) )
            
            self.input_features = self.embedded(word_idx)
            
            #print(self.input_features.size())
            
            if word_idx == 1 :
                return output_words
        
        
        return output_words