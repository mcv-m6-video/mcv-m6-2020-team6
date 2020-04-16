from embedding_net.model import EmbeddingNet
import argparse
from embedding_net.utils import parse_net_params, plot_grapths
import time

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str,default='ai_city_challenge_resnext50.yml',
                        help="path to config file")
    parser.add_argument("--weights", type=str,default='C:\\Users\\Quim\\Desktop\\EmbeddingNet\\work_dirs\\ai_city_challenge\\weights\\best_model_resnet50_ai_city.h5',
                        help="path to trained model weights file")
    parser.add_argument("--encodings", type=str,default='C:\\Users\\Quim\\Desktop\\EmbeddingNet\\work_dirs\\ai_city_challenge\\weights\\encodings_resnet50_ai_city.pkl',
                        help="path to trained model encodings file")
    parser.add_argument("--image", type=str,default='C:\\Users\\Quim\\Desktop\\EmbeddingNet\\Dataset\\val', help="path to image file")
    opt = parser.parse_args()
    '''
    start_1 = time.time()
    config_path = 'ai_city_challenge_resnext50.yml'
    weights_path = 'C:\\Users\\Quim\\Desktop\\EmbeddingNet\\work_dirs\\ai_city_challenge\\weights' \
                   '\\best_model_resnet50_ai_city.h5 '
    encodings_path = 'C:/Users/Quim/Desktop/EmbeddingNet/work_dirs/ai_city_challenge/encodings' \
                     '/encodings_resnet50_ai_city.pkl '
    image_path = 'C:\\Users\\Quim\\Desktop\\image29_1.jpg'

    config = 'configs/ai_city_challenge_resnext50.yml'

    cfg_params = parse_net_params(config)
    model = EmbeddingNet(cfg_params, training=False)
    model.load_model(weights_path)
    model.load_encodings(encodings_path)
    end_1 = time.time()

    print('Load model: ', end_1-start_1)


    start = time.time()
    encoding = model.predict_encoding(image_path)
    print('image encoding: {}'.format(encoding))
    end = time.time()
    model_prediction = model.predict(image_path)
    print('Model prediction: {}'.format(model_prediction))

    total = (end - start)
    print('Prediction time: {}'.format(total))