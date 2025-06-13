# Ergänzung zur IGAN-Klasse (igan/synthesizers/igan.py)
import pickle
import torch
import numpy as np
from cgan.errors import InvalidDataError
import pandas as pd
from torch.nn import functional

def load_gmm(self, gmm_path):
    """Lädt ein vortrainiertes GMM-Modell"""
    try:
        with open(gmm_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.gmm = model_info['gmm']
        self.gmm_k = model_info['k']
        
        if self._verbose:
            print(f"GMM-Modell mit {self.gmm_k} Komponenten erfolgreich geladen.")
        
        return True
    except Exception as e:
        if self._verbose:
            print(f"Fehler beim Laden des GMM-Modells: {e}")
        return False

def generate_gmm_noise(self, batch_size, embedding_dim):
    """Generiert Noise-Samples aus dem GMM-Modell"""
    if not hasattr(self, 'gmm'):
        # Fallback auf normales Rauschen, wenn kein GMM verfügbar
        if self._verbose:
            print("Warnung: Kein GMM-Modell geladen, verwende Normalverteilung.")
        mean = torch.zeros(batch_size, embedding_dim, device=self._device)
        std = mean + 1
        return torch.normal(mean=mean, std=std)
    
    # Samples aus dem GMM ziehen
    noise_samples, _ = self.gmm.sample(batch_size)
    
    # Falls die Dimensionalität des GMM nicht mit embedding_dim übereinstimmt,
    # müssen wir anpassen
    if noise_samples.shape[1] != embedding_dim:
        if noise_samples.shape[1] < embedding_dim:
            # Mit zufälligen Werten auffüllen
            padding = np.random.normal(0, 1, (batch_size, embedding_dim - noise_samples.shape[1]))
            noise_samples = np.hstack([noise_samples, padding])
        else:
            # Dimensionen reduzieren
            noise_samples = noise_samples[:, :embedding_dim]
    
    # Zu Torch Tensor konvertieren und auf das entsprechende Device verschieben
    return torch.tensor(noise_samples, dtype=torch.float32).to(self._device)


def load_preprocessed_csv(trainpath, testpath, inverse=False):
    """
    Load preprocessed CSV files without using an info object.
    
    Args:
        trainpath (str): Path to the training CSV file
        testpath (str): Path to the test CSV file
        inverse (bool): Whether to return inverse transformation functions
        
    Returns:
        tuple: Data needed for training or sampling
    """
    import pandas as pd
    
    # Load train and test data
    train_df = pd.read_csv(trainpath)
    test_df = pd.read_csv(testpath)
    
    return transform_input_data(train_df, test_df, inverse)
    
def transform_input_data(train_df, test_df, inverse=False):
    """
    Load preprocessed CSV files without using an info object.
    
    Args:
        trainpath (str): Path to the training CSV file
        testpath (str): Path to the test CSV file
        inverse (bool): Whether to return inverse transformation functions
        
    Returns:
        tuple: Data needed for training or sampling
    """
    import pandas as pd
    import numpy as np
    

    
    # Remove excluded columns
    excluded_cols = ['combined_tks', 'id']
    for col in excluded_cols:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

    
    # Identify columns after exclusion
    all_columns = train_df.columns.tolist()
    
    # Group categorical columns by their prefix
    category_prefixes = ['category_', 'sub_category1_', 'sub_category2_', 'ticket_type_', 'business_service_']
    
    # Store column metadata for custom transformations
    column_metadata = {
        'categorical_groups': [],
        'numerical_columns': [],
        'target_column': 'impact' if 'impact' in all_columns else None
    }
    
    # Identify categorical columns and group them
    for prefix in category_prefixes:
        cols = [col for col in all_columns if col.startswith(prefix)]
        if cols:
            prefix_clean = prefix.rstrip('_')
            column_metadata['categorical_groups'].append((prefix_clean, sorted(cols)))
    
    # Identify numerical columns (all columns that aren't categorical)
    num_cols = [col for col in all_columns if not any(col.startswith(prefix) for prefix in category_prefixes)]
    column_metadata['numerical_columns'] = num_cols
    
    # Extract numerical features
    X_train_num = train_df[num_cols].values
    X_test_num = test_df[num_cols].values
    
    # Process categorical features - convert one-hot encoding to indices
    X_train_cat_indices = []
    X_test_cat_indices = []
    categories = []
    
    for group_name, group_cols in column_metadata['categorical_groups']:
        # Extract one-hot encoded columns for this group
        train_group = train_df[group_cols].values
        test_group = test_df[group_cols].values
        
        # Convert one-hot encoding to indices (argmax)
        train_indices = np.argmax(train_group, axis=1)
        test_indices = np.argmax(test_group, axis=1)
        
        # Handle all-zero rows
        all_zeros_train = (train_group.sum(axis=1) == 0)
        all_zeros_test = (test_group.sum(axis=1) == 0)
        
        if np.any(all_zeros_train) or np.any(all_zeros_test):
            # Add an extra category for "none selected"
            num_categories = len(group_cols) + 1
            train_indices[all_zeros_train] = len(group_cols)
            test_indices[all_zeros_test] = len(group_cols)
        else:
            num_categories = len(group_cols)
        
        X_train_cat_indices.append(train_indices)
        X_test_cat_indices.append(test_indices)
        categories.append(num_categories)
    
    # Stack all categorical features into a single matrix
    X_train_cat = np.column_stack(X_train_cat_indices) if X_train_cat_indices else np.empty((len(X_train_num), 0), dtype=np.int64)
    X_test_cat = np.column_stack(X_test_cat_indices) if X_test_cat_indices else np.empty((len(X_test_num), 0), dtype=np.int64)
    
    # Number of numerical features
    d_numerical = X_train_num.shape[1]
    
    # Create inverse transformation functions if required
    if inverse:
        # Create numerical inverse transformation (identity function)
        def num_inverse(numerical_data):
            """Identity transform for numerical data"""
            return numerical_data
        
        # Create categorical inverse transformation function that operates on raw indices
        def cat_inverse(categorical_indices):
            """Transform categorical indices back to one-hot encoding"""
            batch_size = categorical_indices.shape[0]
            result = {}
            
            # Process each categorical group
            for group_idx, (group_name, group_cols) in enumerate(column_metadata['categorical_groups']):
                # Get indices for this group
                indices = categorical_indices[:, group_idx].astype(int)
                
                # Create one-hot encoding for this group
                one_hot = np.zeros((batch_size, len(group_cols)))
                
                # Set 1s for valid indices (less than number of columns)
                for row_idx, col_idx in enumerate(indices):
                    if col_idx < len(group_cols):
                        one_hot[row_idx, col_idx] = 1.0
                
                # Store one-hot encoding with column names
                for col_idx, col_name in enumerate(group_cols):
                    result[col_name] = one_hot[:, col_idx]
            
            # Return the categorical data in DataFrame format
            return pd.DataFrame(result)
        
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical, num_inverse, cat_inverse, all_columns, column_metadata
    else:
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical
    

@torch.no_grad()
def split_num_cat_target(syn_data, pre_decoder, token_dim, column_metadata, num_inverse, cat_inverse, device=None):
    """
    Split synthesized data using the pre-trained decoder model.
    
    Args:
        syn_data: The synthesized data from the model
        pre_decoder: The pre-trained decoder model
        token_dim: Dimension of each token
        column_metadata: Metadata about columns
        num_inverse: Function to inverse transform numerical data
        cat_inverse: Function to inverse transform categorical data
        device: Optional device for torch tensors
        
    Returns:
        tuple: (syn_num, syn_cat, syn_target) - numerical, categorical and target data
    """
    import torch
    import numpy as np
    
    # Bestimmen, ob wir eine Zielspalte haben
    target_column = column_metadata.get('target_column')
    has_target = target_column is not None
    
    # Dimensionen erfassen
    numerical_columns = column_metadata['numerical_columns']
    categorical_groups = column_metadata['categorical_groups']
    
    # Umformen der synthetischen Daten für den Pre-Decoder
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    syn_data_tensor = torch.tensor(syn_data, device=device)
    
    # Reshape für den Decoder - von flacher Repräsentation zu tokenisierter Form
    syn_data_reshaped = syn_data_tensor.reshape(syn_data_tensor.shape[0], -1, token_dim)
    
    # Pre-Decoder anwenden, um numerische und kategorische Features zu erhalten
    norm_input = pre_decoder(syn_data_reshaped)
    x_hat_num, x_hat_cat = norm_input
    
    # Kategorische Vorhersagen in Indizes konvertieren
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))
    
    # Zu NumPy konvertieren
    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy() if syn_cat else np.array([])
    
    # Inverse Transformationen anwenden
    syn_num = num_inverse(syn_num)
    
    # Hier müssen wir cat_inverse anpassen, da es nun mit Indizes arbeitet, nicht mit One-Hot
    if hasattr(cat_inverse, "__call__"):
        syn_cat = cat_inverse(syn_cat)
    
    # Ziel extrahieren, falls vorhanden
    if has_target and target_column in numerical_columns:
        # Annahme: Ziel ist die erste numerische Spalte
        syn_target = syn_num[:, :1]  # Erste Spalte als Ziel
        syn_num = syn_num[:, 1:]     # Rest als Features
    else:
        # Keine Zielspalte oder Ziel ist kategorisch
        syn_target = np.zeros((syn_num.shape[0], 1))
    
    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat_df, syn_target, numerical_columns, target_column=None):
    """
    Reconstructs a DataFrame from synthesized numerical, categorical, and target data.
    
    Args:
        syn_num: Numerical data
        syn_cat_df: Categorical data (as DataFrame)
        syn_target: Target data
        numerical_columns: List of numerical column names
        target_column: Name of target column if any
        
    Returns:
        DataFrame: Recovered synthetic data
    """
    import pandas as pd
    import numpy as np
    
    # Create DataFrames for each component
    num_df = pd.DataFrame(syn_num, columns=[col for col in numerical_columns if col != target_column])
    
    # Target DataFrame (if applicable)
    if target_column:
        # Kopie erstellen um Warnung zu vermeiden
        target_values = syn_target.copy()
        
        # Runden
        target_values = np.round(target_values)
        
        # Auf Bereich begrenzen falls angegeben
        min_val, max_val = (0, 4) # impact 0 bis 4
        target_values = np.clip(target_values, min_val, max_val)
            
        # Zu Integer konvertieren
        target_values = target_values.astype(int).astype(str)
            
        target_df = pd.DataFrame(target_values, columns=[target_column])
        # Combine numerical and target
        result_df = pd.concat([target_df, num_df], axis=1)
    else:
        result_df = num_df

    # Konvertiere alle Spalten in syn_cat_df zu boolean 
    # (da alle Spalten in syn_cat_df One-Hot-Encoding-Spalten sind)
    for col in syn_cat_df.columns:
        syn_cat_df[col] = (syn_cat_df[col] > 0.5).astype(bool)
    
    # Combine with categorical data
    result_df = pd.concat([result_df, syn_cat_df], axis=1)
    
    return result_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat


def validate_discrete_columns(train_data, discrete_columns):
    """Check whether ``discrete_columns`` exists in ``train_data``.

    Args:
        train_data (numpy.ndarray or pandas.DataFrame):
            Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        discrete_columns (list-like):
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a Numpy array, this list should
            contain the integer indices of the columns. Otherwise, if it is
            a ``pandas.DataFrame``, this list should contain the column names.
    """
    if isinstance(train_data, pd.DataFrame):
        invalid_columns = set(discrete_columns) - set(train_data.columns)
    elif isinstance(train_data, np.ndarray):
        invalid_columns = []
        for column in discrete_columns:
            if column < 0 or column >= train_data.shape[1]:
                invalid_columns.append(column)
    else:
        raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

    if invalid_columns:
        raise ValueError(f'Invalid columns found: {invalid_columns}')

def validate_null_data(train_data, discrete_columns):
    """Check whether null values exist in continuous ``train_data``.

    Args:
        train_data (numpy.ndarray or pandas.DataFrame):
            Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        discrete_columns (list-like):
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a Numpy array, this list should
            contain the integer indices of the columns. Otherwise, if it is
            a ``pandas.DataFrame``, this list should contain the column names.
    """
    if isinstance(train_data, pd.DataFrame):
        continuous_cols = list(set(train_data.columns) - set(discrete_columns))
        any_nulls = train_data[continuous_cols].isna().any().any()
    else:
        continuous_cols = [i for i in range(train_data.shape[1]) if i not in discrete_columns]
        any_nulls = pd.DataFrame(train_data)[continuous_cols].isna().any().any()

    if any_nulls:
        raise InvalidDataError(
            'CTGAN does not support null values in the continuous training data. '
            'Please remove all null values from your continuous training data.'
        )


def sample_original_condvec(batch, discrete_column_category_prob_flatten, n_categories):
    category_freq = discrete_column_category_prob_flatten
    category_freq = category_freq[category_freq != 0]
    category_freq = category_freq / np.sum(category_freq)
    col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
    cond = np.zeros((batch, n_categories), dtype='float32')
    cond[np.arange(batch), col_idxs] = 1

    return cond


def apply_activate(data, transformer):
    """Apply proper activation function to the output of the generator."""
    data_t = []
    st = 0
    for column_info in transformer.output_info_list:
        for span_info in column_info:
            if span_info.activation_fn == 'tanh':
                ed = st + span_info.dim
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif span_info.activation_fn == 'softmax':
                ed = st + span_info.dim
                transformed = gumbel_softmax(data[:, st:ed], tau=0.2)
                data_t.append(transformed)
                st = ed
            elif span_info.activation_fn == 'identity':
                ed = st + span_info.dim
                data_t.append(torch.sigmoid(data[:, st:ed]))
                st = ed
            else:
                raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

    return torch.cat(data_t, dim=1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """Deals with the instability of the gumbel_softmax for older versions of torch.

    For more details about the issue:
    https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

    Args:
        logits […, num_features]:
            Unnormalized log probabilities
        tau:
            Non-negative scalar temperature
        hard (bool):
            If True, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
        dim (int):
            A dimension along which softmax will be computed. Default: -1.

    Returns:
        Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
    """
    for _ in range(10):
        transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
        if not torch.isnan(transformed).any():
            return transformed

    raise ValueError('gumbel_softmax returning NaN.')


def generate_cond_from_condition_column_info(condition_info, batch, n_categories, discrete_column_matrix_st):
    """Generate the condition vector."""
    vec = np.zeros((batch, n_categories), dtype='float32')
    id_ = discrete_column_matrix_st[condition_info['discrete_column_id']]
    id_ += condition_info['value_id']
    vec[:, id_] = 1
    return vec