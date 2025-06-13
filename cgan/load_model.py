import pickle
from cgan.main import Generator

def load_for_sampling(checkpoint, device='cpu'):
    config = checkpoint['config']
    # Reconstruct generator with the saved config
    generator = Generator(**config['generator_kwargs'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()

    transformer = pickle.loads(checkpoint['transformer_pkl'])
    discrete_column_category_prob_flatten = pickle.loads(config['discrete_column_category_prob_flatten_pkl'])
    discrete_column_matrix_st = pickle.loads(config['discrete_column_matrix_st_pkl'])
    n_categories = config['n_categories']
    return generator, transformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories