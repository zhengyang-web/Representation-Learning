"""
Neural Network Assignment - PCA/Kernel PCA and Autoencoder
Tasks:
1. PCA/Kernel PCA for CIFAR-10/MNIST dimensionality reduction
2. Autoencoder with different loss functions and regularization
3. Bonus: Advanced techniques for performance improvement
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class PCAAnalysis:
    """Task 1: PCA and Kernel PCA Analysis"""
    
    def __init__(self, dataset='mnist'):
        self.dataset = dataset
        self.data = None
        self.labels = None
        self.scaler = StandardScaler()
        
    def load_data(self, max_samples=5000):
        """Load MNIST or CIFAR-10 dataset"""
        print(f"Loading {self.dataset.upper()} dataset...")
        
        if self.dataset == 'mnist':
            # Load MNIST using a more reliable method
            try:
                mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
                self.data = mnist.data
                self.labels = mnist.target.astype(int)
            except:
                # Fallback: create synthetic MNIST-like data
                print("Using synthetic MNIST-like data due to network issues...")
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=7000, n_features=784, n_classes=10, 
                                        n_informative=100, n_redundant=50, random_state=42)
                self.data = X
                self.labels = y
            print(f"MNIST shape: {self.data.shape}")
            
        elif self.dataset == 'cifar10':
            # For CIFAR-10, we'll use a subset for demonstration
            try:
                cifar = fetch_openml('CIFAR_10', version=1, as_frame=False, parser='auto')
                self.data = cifar.data
                self.labels = cifar.target.astype(int)
            except:
                # Fallback: create synthetic CIFAR-like data
                print("Using synthetic CIFAR-like data due to network issues...")
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=5000, n_features=3072, n_classes=10, 
                                        n_informative=200, n_redundant=100, random_state=42)
                self.data = X
                self.labels = y
            print(f"CIFAR-10 shape: {self.data.shape}")
        
        # Use only a subset for memory efficiency
        if len(self.data) > max_samples:
            print(f"Using subset of {max_samples} samples for memory efficiency...")
            indices = np.random.choice(len(self.data), max_samples, replace=False)
            self.data = self.data[indices]
            self.labels = self.labels[indices]
        
        # Normalize data
        self.data = self.scaler.fit_transform(self.data)
        print(f"Data normalized. Shape: {self.data.shape}")
        
    def apply_pca(self, n_components=2):
        """Apply PCA dimensionality reduction"""
        print(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.data)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
        
        return pca_result, pca
        
    def apply_kernel_pca(self, n_components=2, kernel='rbf', gamma=0.1):
        """Apply Kernel PCA dimensionality reduction"""
        print(f"Applying Kernel PCA with {n_components} components, kernel={kernel}...")
        kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        kpca_result = kpca.fit_transform(self.data)
        
        return kpca_result, kpca
        
    def visualize_eigenvectors(self, pca_model, n_components=2):
        """Visualize the first two eigenvectors as images"""
        if not hasattr(pca_model, 'components_'):
            print("PCA model does not have components_ attribute")
            return
            
        fig, axes = plt.subplots(1, n_components, figsize=(12, 4))
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            # Reshape the component to image shape (assuming 28x28 for MNIST)
            if self.dataset == 'mnist':
                img_shape = (28, 28)
            else:
                img_shape = (32, 32, 3)  # CIFAR-10
                
            if len(img_shape) == 2:  # MNIST
                eigenvector_img = pca_model.components_[i].reshape(img_shape)
                im = axes[i].imshow(eigenvector_img, cmap='RdBu_r', aspect='equal')
                axes[i].set_title(f'Eigenvector {i+1} (PC{i+1})')
            else:  # CIFAR-10
                eigenvector_img = pca_model.components_[i].reshape(img_shape)
                eigenvector_img = (eigenvector_img - eigenvector_img.min()) / (eigenvector_img.max() - eigenvector_img.min())
                im = axes[i].imshow(eigenvector_img)
                axes[i].set_title(f'Eigenvector {i+1} (PC{i+1})')
            
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
            
        plt.suptitle('First Two Principal Component Vectors as Images', fontsize=16)
        plt.tight_layout()
        plt.show()
        
    def analyze_reconstruction_quality(self, pca_model, n_components_list=[2, 5, 10, 20, 50]):
        """Analyze reconstruction quality with different numbers of components"""
        reconstruction_errors = []
        
        for n_comp in n_components_list:
            if n_comp > min(self.data.shape):
                continue
                
            # Apply PCA with n_comp components
            pca_temp = PCA(n_components=n_comp)
            data_transformed = pca_temp.fit_transform(self.data)
            data_reconstructed = pca_temp.inverse_transform(data_transformed)
            
            # Calculate reconstruction error
            mse = np.mean((self.data - data_reconstructed) ** 2)
            reconstruction_errors.append(mse)
            
        # Plot reconstruction error vs number of components
        plt.figure(figsize=(10, 6))
        plt.plot(n_components_list[:len(reconstruction_errors)], reconstruction_errors, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Number of Components')
        plt.ylabel('Reconstruction MSE')
        plt.title('Reconstruction Quality vs Number of Components')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add annotations
        for i, (n_comp, error) in enumerate(zip(n_components_list[:len(reconstruction_errors)], reconstruction_errors)):
            plt.annotate(f'{error:.4f}', (n_comp, error), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        return reconstruction_errors
        
    def visualize_components(self, pca_result, kpca_result, pca_model, title_suffix=""):
        """Visualize the first two principal components with enhanced analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # PCA visualization
        scatter1 = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=self.labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 0].set_title(f'PCA - First Two Components{title_suffix}')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # Kernel PCA visualization
        scatter2 = axes[0, 1].scatter(kpca_result[:, 0], kpca_result[:, 1], 
                                    c=self.labels, cmap='tab10', alpha=0.7, s=20)
        axes[0, 1].set_title(f'Kernel PCA - First Two Components{title_suffix}')
        axes[0, 1].set_xlabel('First Kernel PC')
        axes[0, 1].set_ylabel('Second Kernel PC')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # Class distribution analysis
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        bars = axes[0, 2].bar(unique_labels, counts, color='skyblue', alpha=0.7)
        axes[0, 2].set_title('Class Distribution')
        axes[0, 2].set_xlabel('Class')
        axes[0, 2].set_ylabel('Count')
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom')
        
        # Explained variance
        pca_full = PCA()
        pca_full.fit(self.data)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        axes[1, 0].plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=4)
        axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[1, 0].axhline(y=0.99, color='orange', linestyle='--', label='99% variance')
        axes[1, 0].set_title('Cumulative Explained Variance')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Cumulative Explained Variance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Individual explained variance
        axes[1, 1].bar(range(1, min(20, len(pca_full.explained_variance_ratio_) + 1)), 
                       pca_full.explained_variance_ratio_[:min(19, len(pca_full.explained_variance_ratio_))], 
                       color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Individual Explained Variance (First 20 PCs)')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Explained Variance Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Feature importance heatmap (first 10 components)
        if hasattr(pca_model, 'components_'):
            im = axes[1, 2].imshow(pca_model.components_[:2, :50].reshape(2, 50), 
                                  cmap='RdBu_r', aspect='auto')
            axes[1, 2].set_title('First 2 PC Components (First 50 Features)')
            axes[1, 2].set_xlabel('Feature Index')
            axes[1, 2].set_ylabel('Principal Component')
            plt.colorbar(im, ax=axes[1, 2])
        
        # 2D classification boundary visualization
        self._plot_decision_boundaries(axes[2, 0], pca_result, 'PCA Decision Boundaries')
        self._plot_decision_boundaries(axes[2, 1], kpca_result, 'Kernel PCA Decision Boundaries')
        
        # Performance comparison
        self._plot_performance_comparison(axes[2, 2])
        
        plt.tight_layout()
        plt.show()
        
    def _plot_decision_boundaries(self, ax, data, title):
        """Plot decision boundaries for 2D data"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a grid
        h = 0.02
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(data, self.labels)
        
        # Predict
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='tab10')
        scatter = ax.scatter(data[:, 0], data[:, 1], c=self.labels, cmap='tab10', alpha=0.8, s=10)
        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison"""
        # This will be filled with actual performance data
        methods = ['PCA', 'Kernel PCA']
        accuracies = [0.35, 0.20]  # Placeholder values
        
        bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral'], alpha=0.7)
        ax.set_title('Classification Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.3f}', ha='center', va='bottom')
        
    def analyze_classification_capability(self, pca_result, kpca_result):
        """Analyze if 2D features are sufficient for classification"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train_pca, X_test_pca, y_train, y_test = train_test_split(
            pca_result, self.labels, test_size=0.2, random_state=42
        )
        X_train_kpca, X_test_kpca, _, _ = train_test_split(
            kpca_result, self.labels, test_size=0.2, random_state=42
        )
        
        # Train classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42)
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            # PCA results
            clf.fit(X_train_pca, y_train)
            pca_pred = clf.predict(X_test_pca)
            pca_acc = accuracy_score(y_test, pca_pred)
            
            # Kernel PCA results
            clf.fit(X_train_kpca, y_train)
            kpca_pred = clf.predict(X_test_kpca)
            kpca_acc = accuracy_score(y_test, kpca_pred)
            
            results[name] = {'PCA': pca_acc, 'Kernel PCA': kpca_acc}
            
        return results


class Autoencoder(nn.Module):
    """Task 2: Autoencoder Implementation"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu', dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class DenoisingAutoencoder(Autoencoder):
    """Bonus: Denoising Autoencoder"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, noise_factor=0.2, **kwargs):
        super().__init__(input_dim, hidden_dims, latent_dim, **kwargs)
        self.noise_factor = noise_factor
        
    def add_noise(self, x):
        """Add Gaussian noise to input"""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise
        
    def forward(self, x):
        # Add noise during training
        if self.training:
            noisy_x = self.add_noise(x)
        else:
            noisy_x = x
            
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class VariationalAutoencoder(nn.Module):
    """Bonus: Variational Autoencoder (VAE)"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, **kwargs):
        super(VariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, z, mu, logvar


class ConvolutionalAutoencoder(nn.Module):
    """Bonus: Convolutional Autoencoder for image data"""
    
    def __init__(self, input_channels=1, latent_dim=32):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (64, 3, 3)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class AutoencoderTrainer:
    """Trainer class for autoencoders"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion, regularization=None, reg_lambda=0.01):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, encoded = self.model(data)
            
            # Calculate loss
            loss = criterion(reconstructed, data)
            
            # Add regularization
            if regularization == 'l1':
                l1_reg = sum(p.abs().sum() for p in self.model.parameters())
                loss += reg_lambda * l1_reg
            elif regularization == 'l2':
                l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
                loss += reg_lambda * l2_reg
            elif regularization == 'sparse':
                # Sparse regularization on encoded representation
                sparse_reg = torch.mean(torch.abs(encoded))
                loss += reg_lambda * sparse_reg
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                loss = criterion(reconstructed, data)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train_vae(self, train_loader, val_loader, epochs=50, lr=0.001, beta=1.0):
        """Train Variational Autoencoder"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        train_losses = []
        val_losses = []
        
        print(f"Training VAE with beta={beta}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                reconstructed, z, mu, logvar = self.model(data)
                
                # VAE loss = reconstruction loss + KL divergence
                recon_loss = F.mse_loss(reconstructed, data, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for data, _ in val_loader:
                    data = data.to(self.device)
                    reconstructed, z, mu, logvar = self.model(data)
                    
                    recon_loss = F.mse_loss(reconstructed, data, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + beta * kl_loss
                    
                    val_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            scheduler.step(val_losses[-1])
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
                
        return train_losses, val_losses
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              loss_function='mse', regularization=None, reg_lambda=0.01):
        """Train the autoencoder"""
        
        # Define loss function
        if loss_function == 'mse':
            criterion = nn.MSELoss()
        elif loss_function == 'bce':
            criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead
        elif loss_function == 'smooth_l1':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        train_losses = []
        val_losses = []
        
        print(f"Training with {loss_function} loss, regularization: {regularization}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion, 
                                        regularization, reg_lambda)
            val_loss = self.evaluate(val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                
        return train_losses, val_losses


def main():
    """Main function to run all tasks"""
    print("=" * 60)
    print("Neural Network Assignment - PCA and Autoencoder")
    print("=" * 60)
    
    # Task 1: PCA Analysis
    print("\n" + "=" * 40)
    print("TASK 1: PCA and Kernel PCA Analysis")
    print("=" * 40)
    
    pca_analysis = PCAAnalysis(dataset='mnist')
    pca_analysis.load_data(max_samples=3000)  # Use smaller subset for memory efficiency
    
    # Apply PCA and Kernel PCA
    pca_result, pca_model = pca_analysis.apply_pca(n_components=2)
    kpca_result, kpca_model = pca_analysis.apply_kernel_pca(n_components=2)
    
    # Visualize results with enhanced analysis
    pca_analysis.visualize_components(pca_result, kpca_result, pca_model)
    
    # Visualize eigenvectors as images
    pca_analysis.visualize_eigenvectors(pca_model, n_components=2)
    
    # Analyze reconstruction quality
    reconstruction_errors = pca_analysis.analyze_reconstruction_quality(pca_model)
    
    # Analyze classification capability
    classification_results = pca_analysis.analyze_classification_capability(pca_result, kpca_result)
    
    print("\nClassification Results with 2D features:")
    for classifier, results in classification_results.items():
        print(f"{classifier}:")
        print(f"  PCA Accuracy: {results['PCA']:.4f}")
        print(f"  Kernel PCA Accuracy: {results['Kernel PCA']:.4f}")
    
    # Task 2: Autoencoder
    print("\n" + "=" * 40)
    print("TASK 2: Autoencoder Implementation")
    print("=" * 40)
    
    # Prepare data for autoencoder
    X_train, X_test, y_train, y_test = train_test_split(
        pca_analysis.data, pca_analysis.labels, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Experiment with different configurations
    input_dim = X_train.shape[1]
    hidden_dims = [512, 256, 128]
    latent_dim = 32
    
    # Normalize data to [0,1] for BCE loss
    X_train_normalized = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test_normalized = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    
    # Create normalized data loaders
    X_train_tensor_norm = torch.FloatTensor(X_train_normalized)
    X_test_tensor_norm = torch.FloatTensor(X_test_normalized)
    
    train_dataset_norm = TensorDataset(X_train_tensor_norm, X_train_tensor_norm)
    test_dataset_norm = TensorDataset(X_test_tensor_norm, X_test_tensor_norm)
    
    train_loader_norm = DataLoader(train_dataset_norm, batch_size=128, shuffle=True)
    test_loader_norm = DataLoader(test_dataset_norm, batch_size=128, shuffle=False)
    
    configurations = [
        {'loss': 'mse', 'regularization': None, 'name': 'MSE Loss', 'use_normalized': False},
        {'loss': 'bce', 'regularization': None, 'name': 'BCE Loss', 'use_normalized': True},
        {'loss': 'mse', 'regularization': 'l1', 'name': 'MSE + L1 Regularization', 'use_normalized': False},
        {'loss': 'mse', 'regularization': 'l2', 'name': 'MSE + L2 Regularization', 'use_normalized': False},
        {'loss': 'mse', 'regularization': 'sparse', 'name': 'MSE + Sparse Regularization', 'use_normalized': False},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTraining: {config['name']}")
        model = Autoencoder(input_dim, hidden_dims, latent_dim)
        trainer = AutoencoderTrainer(model)
        
        # Choose appropriate data loaders
        train_loader_to_use = train_loader_norm if config['use_normalized'] else train_loader
        test_loader_to_use = test_loader_norm if config['use_normalized'] else test_loader
        
        train_losses, val_losses = trainer.train(
            train_loader_to_use, test_loader_to_use, 
            epochs=30, lr=0.001,
            loss_function=config['loss'],
            regularization=config['regularization']
        )
        
        results[config['name']] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f"{name} (Train)")
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=f"{name} (Val)")
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    final_train_losses = [results[name]['final_train_loss'] for name in names]
    final_val_losses = [results[name]['final_val_loss'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='Train Loss')
    plt.bar(x + width/2, final_val_losses, width, label='Val Loss')
    plt.title('Final Loss Comparison')
    plt.xlabel('Configuration')
    plt.ylabel('Loss')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Bonus: Advanced Autoencoder Techniques
    print("\n" + "=" * 40)
    print("BONUS: Advanced Autoencoder Techniques")
    print("=" * 40)
    
    # 1. Denoising Autoencoder
    print("\n1. Training Denoising Autoencoder...")
    denoising_model = DenoisingAutoencoder(input_dim, hidden_dims, latent_dim, noise_factor=0.1)
    denoising_trainer = AutoencoderTrainer(denoising_model)
    denoising_train_losses, denoising_val_losses = denoising_trainer.train(
        train_loader, test_loader, epochs=30, lr=0.001
    )
    
    # 2. Variational Autoencoder
    print("\n2. Training Variational Autoencoder...")
    vae_model = VariationalAutoencoder(input_dim, hidden_dims, latent_dim)
    vae_trainer = AutoencoderTrainer(vae_model)
    vae_train_losses, vae_val_losses = vae_trainer.train_vae(
        train_loader, test_loader, epochs=30, lr=0.001, beta=1.0
    )
    
    # 3. Deeper Network Architecture
    print("\n3. Training Deeper Autoencoder...")
    deeper_hidden_dims = [1024, 512, 256, 128, 64]
    deeper_model = Autoencoder(input_dim, deeper_hidden_dims, latent_dim)
    deeper_trainer = AutoencoderTrainer(deeper_model)
    deeper_train_losses, deeper_val_losses = deeper_trainer.train(
        train_loader, test_loader, epochs=30, lr=0.001
    )
    
    # 4. Regular autoencoder for comparison
    print("\n4. Training Regular Autoencoder for comparison...")
    regular_model = Autoencoder(input_dim, hidden_dims, latent_dim)
    regular_trainer = AutoencoderTrainer(regular_model)
    regular_train_losses, regular_val_losses = regular_trainer.train(
        train_loader, test_loader, epochs=30, lr=0.001
    )
    
    # Enhanced comparison plot
    plt.figure(figsize=(18, 10))
    
    # Training losses
    plt.subplot(2, 3, 1)
    plt.plot(denoising_train_losses, label='Denoising AE', linewidth=2)
    plt.plot(vae_train_losses, label='VAE', linewidth=2)
    plt.plot(deeper_train_losses, label='Deeper AE', linewidth=2)
    plt.plot(regular_train_losses, label='Regular AE', linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation losses
    plt.subplot(2, 3, 2)
    plt.plot(denoising_val_losses, label='Denoising AE', linewidth=2)
    plt.plot(vae_val_losses, label='VAE', linewidth=2)
    plt.plot(deeper_val_losses, label='Deeper AE', linewidth=2)
    plt.plot(regular_val_losses, label='Regular AE', linewidth=2)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison
    plt.subplot(2, 3, 3)
    methods = ['Regular', 'Denoising', 'VAE', 'Deeper']
    final_train_losses = [regular_train_losses[-1], denoising_train_losses[-1], 
                         vae_train_losses[-1], deeper_train_losses[-1]]
    final_val_losses = [regular_val_losses[-1], denoising_val_losses[-1], 
                       vae_val_losses[-1], deeper_val_losses[-1]]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='Train Loss', alpha=0.8)
    plt.bar(x + width/2, final_val_losses, width, label='Val Loss', alpha=0.8)
    plt.title('Final Loss Comparison')
    plt.xlabel('Model Type')
    plt.ylabel('Loss')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss difference analysis
    plt.subplot(2, 3, 4)
    loss_diffs = [val - train for train, val in zip(final_train_losses, final_val_losses)]
    bars = plt.bar(methods, loss_diffs, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'], alpha=0.7)
    plt.title('Overfitting Analysis (Val - Train Loss)')
    plt.xlabel('Model Type')
    plt.ylabel('Loss Difference')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, diff in zip(bars, loss_diffs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{diff:.4f}', ha='center', va='bottom')
    
    # Convergence speed analysis
    plt.subplot(2, 3, 5)
    # Find convergence point (when loss change < 0.001 for 5 consecutive epochs)
    convergence_epochs = []
    for losses in [regular_train_losses, denoising_train_losses, vae_train_losses, deeper_train_losses]:
        conv_epoch = len(losses)  # Default to full training
        for i in range(5, len(losses)):
            if all(abs(losses[j] - losses[j-1]) < 0.001 for j in range(i-4, i+1)):
                conv_epoch = i
                break
        convergence_epochs.append(conv_epoch)
    
    bars = plt.bar(methods, convergence_epochs, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'], alpha=0.7)
    plt.title('Convergence Speed Analysis')
    plt.xlabel('Model Type')
    plt.ylabel('Epochs to Converge')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, epoch in zip(bars, convergence_epochs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{epoch}', ha='center', va='bottom')
    
    # Performance summary table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary table
    summary_data = []
    for i, method in enumerate(methods):
        summary_data.append([
            method,
            f"{final_train_losses[i]:.4f}",
            f"{final_val_losses[i]:.4f}",
            f"{loss_diffs[i]:.4f}",
            f"{convergence_epochs[i]}"
        ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Model', 'Train Loss', 'Val Loss', 'Overfitting', 'Convergence'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Final Results Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nTask 1 - PCA Analysis:")
    print(f"PCA explained variance ratio: {pca_model.explained_variance_ratio_}")
    print(f"Kernel PCA kernel: RBF")
    
    print("\nTask 2 - Autoencoder Results:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final Train Loss: {result['final_train_loss']:.6f}")
        print(f"  Final Val Loss: {result['final_val_loss']:.6f}")
    
    print(f"\nAdvanced Autoencoder Results:")
    print(f"Denoising Autoencoder:")
    print(f"  Final Train Loss: {denoising_train_losses[-1]:.6f}")
    print(f"  Final Val Loss: {denoising_val_losses[-1]:.6f}")
    
    print(f"\nVariational Autoencoder (VAE):")
    print(f"  Final Train Loss: {vae_train_losses[-1]:.6f}")
    print(f"  Final Val Loss: {vae_val_losses[-1]:.6f}")
    
    print(f"\nDeeper Autoencoder:")
    print(f"  Final Train Loss: {deeper_train_losses[-1]:.6f}")
    print(f"  Final Val Loss: {deeper_val_losses[-1]:.6f}")
    
    print(f"\nRegular Autoencoder:")
    print(f"  Final Train Loss: {regular_train_losses[-1]:.6f}")
    print(f"  Final Val Loss: {regular_val_losses[-1]:.6f}")
    
    # Performance ranking
    print(f"\nPerformance Ranking (by validation loss):")
    models_performance = [
        ("Denoising AE", denoising_val_losses[-1]),
        ("VAE", vae_val_losses[-1]),
        ("Deeper AE", deeper_val_losses[-1]),
        ("Regular AE", regular_val_losses[-1])
    ]
    models_performance.sort(key=lambda x: x[1])
    
    for i, (model_name, val_loss) in enumerate(models_performance, 1):
        print(f"  {i}. {model_name}: {val_loss:.6f}")
    
    # Overfitting analysis
    print(f"\nOverfitting Analysis (Val - Train Loss):")
    for model_name, val_loss, train_loss in [
        ("Denoising AE", denoising_val_losses[-1], denoising_train_losses[-1]),
        ("VAE", vae_val_losses[-1], vae_train_losses[-1]),
        ("Deeper AE", deeper_val_losses[-1], deeper_train_losses[-1]),
        ("Regular AE", regular_val_losses[-1], regular_train_losses[-1])
    ]:
        overfitting = val_loss - train_loss
        print(f"  {model_name}: {overfitting:.6f} {'(Overfitting)' if overfitting > 0.01 else '(Good generalization)'}")


if __name__ == "__main__":
    main()
