from embedding_net.utils import parse_net_params, plot_grapths
from embedding_net.model_emb import EmbeddingNet

config_name = 'ai_city_challenge_resnext50'

cfg_params = parse_net_params('configs/{}.yml'.format(config_name))
model = EmbeddingNet(cfg_params, training=False)

model.generate_encodings(save_file_name='work_dirs/ai_city_challenge/encodings/encodings_without_training_{}.pkl'.format(config_name, config_name), max_num_samples_of_each_class=30, knn_k=1, shuffle=True)

