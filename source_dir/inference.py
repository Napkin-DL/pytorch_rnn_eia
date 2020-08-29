
import json
import logging
import os
 
from rnn import RNNModel

import torch
import data
import pandas

 
JSON_CONTENT_TYPE = 'application/json'
 
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    This function is called by the Pytorch container during hosting when running on SageMaker with
    values populated by the hosting environment.

    This function loads models written during training into `model_dir`.
    """
    
    print("****model_dir : {}".format(model_dir))
    
    logger.info('Loading the model.')

    model_info = {}
    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        model_info = torch.load(f)
    print('model_info: {}'.format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))
    model = RNNModel(rnn_type=model_info['rnn_type'], ntoken=model_info['ntoken'],
                     ninp=model_info['ninp'], nhid=model_info['nhid'], nlayers=model_info['nlayers'],
                     dropout=model_info['dropout'], tie_weights=model_info['tie_weights'])
    
    
    model_data = torch.load('/opt/ml/model/code/corpus_hidden')
    hidden = model_data['hidden']

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        save_dir = os.path.join(model_dir, "model.pt")
        
        ## temp_input size should be as a same as input size
        temp_input = torch.randint(1,(25,20))
        model = torch.jit.trace(model.eval(), (temp_input, hidden))
        torch.jit.save(model, save_dir)
 
    return model

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        if input_data['temperature'] < 1e-3:
            raise Exception('\'temperature\' has to be greater or equal 1e-3')
        return input_data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(input_data, model):
    logger.info('Generating text based on input parameters.')
#     corpus = model['corpus']
#     model = model['model']


    model_data = torch.load('/opt/ml/model/code/corpus_hidden')
    hidden = model_data['hidden']
    corpus = model_data['corpus']
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))
    torch.manual_seed(input_data['seed'])

    ntokens = len(corpus.dictionary)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
#     hidden = model.init_hidden(1)
    
    model.to(device)
    model.eval()
 
    logger.info('Generating {} words.'.format(input_data['words']))
    result = []
    with torch.no_grad():  # no tracking history
        with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
            for i in range(input_data['words']):
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(input_data['temperature']).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx.squeeze())
                word = corpus.dictionary.idx2word[word_idx]
                word = word if type(word) == str else word.decode()
                if word == '<eos>':
                    word = '\n'
                elif i % 12 == 11:
                    word = word + '\n'
                else:
                    word = word + ' '
                result.append(word)
    return ''.join(result)
