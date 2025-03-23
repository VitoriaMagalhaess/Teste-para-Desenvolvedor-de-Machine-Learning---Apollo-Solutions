import pickle
import numpy as np

def load_data(filepath):
    """
    Carrega os dados do arquivo pickle.

    Args:
        filepath (str): Caminho para o arquivo pickle.

    Returns:
        dict: Dados carregados ou um dicionário vazio em caso de falha.
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except:
        return {}

def explore_hierarchy(data):
    """Explora a hierarquia dos dados e imprime estatísticas."""
    if not data:
        return

    syndrome_counts = {}
    subject_counts = {}
    image_counts = 0

    for syndrome_id, subjects in data.items():
        syndrome_counts[syndrome_id] = len(subjects)
        for subject_id, images in subjects.items():
            subject_counts[subject_id] = len(images)
            image_counts += len(images)

    print("Estatísticas da Hierarquia:")
    print(f"  Número de síndromes: {len(syndrome_counts)}")
    print(f"  Número de sujeitos: {len(subject_counts)}")
    print(f"  Número total de imagens: {image_counts}")

    print("\nContagem de sujeitos por síndrome:")
    for syndrome_id, count in syndrome_counts.items():
        print(f"    {syndrome_id}: {count}")

    print("\nContagem de imagens por sujeito (amostra):")
    for subject_id, count in list(subject_counts.items())[:5]:  
        print(f"    {subject_id}: {count}")

def flatten_data(data):
    if not data: 
        return np.array([]), np.array([])

    embeddings = []
    labels = []

    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                labels.append(syndrome_id)

    return np.array(embeddings), np.array(labels)

def handle_missing_data(embeddings):
    """Lida com dados faltantes (NaNs)."""
    return np.nan_to_num(embeddings)

if __name__ == "__main__":
    filepath = "/content/mini_gm_public_v0.1.p"  # Caminho do arquivo

    # Tenta carregar os dados
    data = load_data(filepath)
    
    # Se os dados forem carregados corretamente
    if data:
        explore_hierarchy(data)
        embeddings, labels = flatten_data(data)
        embeddings = handle_missing_data(embeddings)

        print(f"\nNúmero de embeddings: {len(embeddings)}")
        print(f"Número de labels: {len(labels)}")

        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Contagem de labels:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
