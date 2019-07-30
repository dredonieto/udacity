import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout= 0.3, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        #self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size))
    
    def forward(self, features, captions):
        
        # Delete <end> to avoid predicting it when it appears in the input
        cap_embed = self.embedding(captions[:,:-1])
        
        # merge the embedded image with the embedded word vector
        #print(features.shape)
        #print(features.unsqueeze(1).shape)
        #print(cap_embed.shape)
        embeddings = torch.cat((features.unsqueeze(1), cap_embed), dim=1)
        
        self.hidden = self.init_hidden(features.shape[0])
        
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        output = self.linear(lstm_out)
        
        return output
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        weight = next(self.parameters()).data
        # initialize hidden state with zero weights, and move to GPU if available
        if (device == 'cuda'):
            hidden = (weight.new(1, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(1, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(1, batch_size, self.hidden_size).zero_(),
                      weight.new(1, batch_size, self.hidden_size).zero_())
            
        return hidden


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
            if (max_indice == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break
                
            if (len(output) == max_len):
                # We have reached to the maximum lenght of the sentence
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.embedding(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
        
        return output