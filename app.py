from flask import Flask, request, jsonify, render_template
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer, tokenizer
import torch
import torch.nn.functional as F
import pandas as pd
import os
import shutil
from sklearn.decomposition import PCA



# Load the pre-trained model and preprocess function

model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained="openai")
model.eval()

# Initialize the tokenizer
tokenizer = get_tokenizer('ViT-B-32')

# Load the image embeddings DataFrame
df = pd.read_pickle('image_embeddings.pickle')

# Define the base directory where images are stored
base_dir = r'C:\Users\Victor\Downloads\CS 506\zjiang48-assignment-10\coco_images_resized'

# Ensure all image files are moved to the static folder
static_image_dir = os.path.join(os.getcwd(), 'static')
os.makedirs(static_image_dir, exist_ok=True)

# Copy images to the static folder and update DataFrame paths
for _, row in df.iterrows():
    src_path = os.path.join(base_dir, row['file_name'])  # Full path to the original file
    dest_path = os.path.join(static_image_dir, os.path.basename(src_path))  # Static folder
    if not os.path.exists(dest_path):  # Avoid re-copying files
        shutil.copy(src_path, dest_path)

# Update DataFrame paths to point to the static folder
df['file_name'] = df['file_name'].apply(lambda x: x.replace("\\", "/"))

# Precompute PCA embeddings
default_k = 50
pca = PCA(n_components=default_k)
clip_embeddings = torch.stack([torch.tensor(embed) for embed in df['embedding']])
clip_embeddings_np = clip_embeddings.numpy()  # Convert to NumPy for PCA
pca_embeddings = pca.fit_transform(clip_embeddings_np)
df['pca_embedding'] = list(pca_embeddings)


# Flask application setup
app = Flask(__name__)

# Helper function to calculate cosine similarity
def cosine_similarity(query_embedding, embeddings):
    return torch.matmul(query_embedding, embeddings.T)

def get_top_k_matches(query_embedding, embeddings, k=5):
    """
    Finds the top k matches based on cosine similarity.
    """
    similarities = cosine_similarity(query_embedding, embeddings).squeeze(0)
    top_k_indices = torch.topk(similarities, k=min(k, embeddings.size(0))).indices
    return [
        {
            "file_name": f"static/{df.iloc[idx.item()]['file_name']}",
            "score": similarities[idx.item()].item()
        }
        for idx in top_k_indices
    ]
    
# Function to compute PCA embeddings dynamically
def compute_pca_embeddings(embeddings, k):
    
    pca = PCA(n_components=k)
    embeddings_np = embeddings.numpy()  # Convert to NumPy
    reduced_embeddings = pca.fit_transform(embeddings_np)
    return torch.tensor(reduced_embeddings, dtype=torch.float32), pca


@app.route('/')
def home():
    return render_template("index.html")  # Front-end form

@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type')
    use_pca = request.form.get('use_pca', 'false').lower() == 'true'
    k = request.form.get('k', type=int, default=50)
    text_query = request.form.get('text_query')
    image_query_file = request.files.get('image_query')
    weight = request.form.get('weight', type=float, default=0.5)

    text_embedding = None
    image_embedding = None
    query_embedding = None

    # Process query based on type
    if query_type == "text":
        if not text_query:
            return jsonify({"error": "Text query selected but no text provided."}), 400
        tokens = tokenizer([text_query])
        with torch.no_grad():
            query_embedding = F.normalize(model.encode_text(tokens))

    elif query_type == "image":
        if not image_query_file:
            return jsonify({"error": "Image query selected but no image uploaded."}), 400
        image = preprocess(Image.open(image_query_file)).unsqueeze(0)
        with torch.no_grad():
            query_embedding = F.normalize(model.encode_image(image))

    elif query_type == "combined":
        if not text_query or not image_query_file:
            return jsonify({"error": "Combined query selected but one or both inputs are missing."}), 400
        tokens = tokenizer([text_query])
        with torch.no_grad():
            text_embedding = F.normalize(model.encode_text(tokens))
        image = preprocess(Image.open(image_query_file)).unsqueeze(0)
        with torch.no_grad():
            image_embedding = F.normalize(model.encode_image(image))
        query_embedding = F.normalize(
            weight * text_embedding + (1.0 - weight) * image_embedding
        )
    else:
        return jsonify({"error": "Invalid query type selected."}), 400

    # Compute PCA embeddings dynamically if required
    global_embeddings = torch.stack([torch.tensor(embed) for embed in df['embedding']])
    if use_pca:
        # Apply PCA to global and query embeddings
        pca_embeddings, pca = compute_pca_embeddings(global_embeddings, k)
        query_embedding_np = query_embedding.numpy().reshape(1, -1)  # Ensure proper shape
        query_embedding_pca = pca.transform(query_embedding_np)
        query_embedding = torch.tensor(query_embedding_pca, dtype=torch.float32)
    else:
        pca_embeddings = global_embeddings

    # Retrieve top 5 matches
    top_k_matches = get_top_k_matches(query_embedding, embeddings=pca_embeddings, k=k)

    return jsonify(top_k_matches)







if __name__ == '__main__':
    app.run(debug=True)
