# Wasserstein GAN mit Gradient Penalty (WGAN-GP)
## Eine wissenschaftliche Zusammenfassung mit Implementierungsdetails

### Theoretische Grundlage

Wasserstein GANs nutzen die **Wasserstein-Distanz** als Metrik zwischen Wahrscheinlichkeitsverteilungen, die unter der Voraussetzung berechnet werden kann, dass der Diskriminator eine 1-Lipschitz-Funktion ist.

### Mathematische Formulierung mit Implementierung

#### 1. Wasserstein-Verlustfunktion

Die grundlegende Verlustfunktion des WGAN-Diskriminators:

$$L_W = \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{x \sim P_g}[D(x)]$$

```python
# Berechnung des Wasserstein-Verlusts
y_real = discriminator(real_data)  # Bewertung realer Daten
y_fake = discriminator(fake_data)  # Bewertung generierter Daten
loss_d = -(torch.mean(y_real) - torch.mean(y_fake))  # Negativ für Minimierung
```

#### 2. Gradient Penalty

Der Gradient Penalty erzwingt die 1-Lipschitz-Bedingung:

$$L_{GP} = \lambda \cdot \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

```python
def calc_gradient_penalty(self, real_data, fake_data, device, pac=1):
    # Batch-Größe unter Berücksichtigung des PAC-Faktors
    batch_size = real_data.size(0) // pac
    
    # Interpolierte Punkte zwischen realen und gefälschten Daten erzeugen
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, real_data.size(1))
    
    # Wenn PAC > 1, reshape für die "Paketierung"
    if pac > 1:
        alpha = alpha.unsqueeze(1).expand(batch_size, pac, real_data.size(1))
        alpha = alpha.contiguous().view(-1, real_data.size(1))
    
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # Diskriminator-Ausgabe für interpolierte Punkte
    disc_interpolates = self(interpolates)
    
    # Gradienten bezüglich der Eingabe berechnen
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient Penalty: (||∇D(x)||₂ - 1)²
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty
```

#### 3. Training des Diskriminators

Die vollständige Trainingsschleife mit separaten backward()-Aufrufen:

```python
# Training-Iteration des Diskriminators
# Wasserstein-Verlust berechnen
y_fake = discriminator(fake_cat)
y_real = discriminator(real_cat)
loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

# Gradient Penalty berechnen
pen = discriminator.calc_gradient_penalty(
    real_cat, fake_cat, device, pac
)

# Getrennte backward()-Aufrufe für bessere Kontrolle und Stabilität
optimizerD.zero_grad(set_to_none=False)
pen.backward(retain_graph=True)      # Gradienten für Gradient Penalty
loss_d.backward()                    # Gradienten für Wasserstein-Verlust
optimizerD.step()                    # Parameter-Update
```

### Gradientennorm und ihre Bedeutung

Die Gradientennorm misst die lokale Änderungsrate des Diskriminators:

$$\|\nabla_x D(x)\|_2 = \sqrt{\sum_{i=1}^n \left(\frac{\partial D(x)}{\partial x_i}\right)^2}$$

```python
# Berechnung der Gradientennorm in PyTorch
gradients = torch.autograd.grad(...)[0]
gradient_norm = gradients.norm(2, dim=1)  # L2-Norm entlang der Feature-Dimension
```

Diese Norm hat direkte Auswirkungen auf das Trainingsverhalten:
1. **Stabile Gradienten**: Verhindert verschwindende/explodierende Gradienten
2. **Konsistente Updates**: Sorgt für gleichmäßige Aktualisierungen des Generators
3. **Bessere Konvergenz**: Reduziert Oszillationen im Training

### Empirisch beobachtete Verbesserungen

WGAN-GP gegenüber Standard-GAN und WGAN mit Weight Clipping zeigt:
- Reduzierte Anfälligkeit für Mode Collapse
- Stabileres Trainingsverhalten über verschiedene Architekturen hinweg
- Höhere Qualität der generierten Samples

### Referenz

Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved training of Wasserstein GANs. *Advances in Neural Information Processing Systems*, 30.