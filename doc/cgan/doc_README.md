# Die Funktionsweise des GAN für tabellarische Datensynthese

## Überblick
Das GAN (Generative Adversarial Network) für die Synthese tabellarischer Daten im CTGAN (Conditional Tabular GAN) Modell basiert auf einem Wettbewerb zwischen zwei neuronalen Netzwerken: Generator und Diskriminator. Ziel ist es, synthetische Daten zu erzeugen, die statistisch ähnlich zu den Originaldaten sind.

## Grundlegende Komponenten

### 1. Datenaufbereitung
- **DataTransformer**: Wandelt Rohdaten in ein Format um, das das GAN verarbeiten kann
  - Kontinuierliche Spalten werden mit einem BayesianGMM (Gaussian Mixture Model) modelliert und normalisiert
  - Diskrete Spalten werden mit One-Hot-Encoding kodiert

### 2. Kernkomponenten
- **Generator**: Erzeugt synthetische Daten aus Rauschen und Bedingungsvektoren
- **Diskriminator**: Versucht, echte von synthetischen Daten zu unterscheiden
- **DataSampler**: Stellt Trainingsdaten und Bedingungsvektoren bereit

## Die Fit-Funktion im Detail

### 1. Initialisierungsphase
- **Validierung der Eingabedaten**:
  ```python
  self._validate_discrete_columns(train_data, discrete_columns)
  self._validate_null_data(train_data, discrete_columns)
  ```
  - Überprüft, ob die angegebenen diskreten Spalten tatsächlich in den Daten existieren
  - Stellt sicher, dass keine Null-Werte in kontinuierlichen Spalten vorhanden sind

### 2. Datenaufbereitung
- **Daten transformieren**:
  ```python
  self._transformer = DataTransformer()
  self._transformer.fit(train_data, discrete_columns)
  train_data = self._transformer.transform(train_data)
  ```
  - Erstellt einen DataTransformer, der die Daten in ein geeignetes Format umwandelt
  - Kontinuierliche Spalten werden normalisiert und mit GMM modelliert
  - Diskrete Spalten werden one-hot-encodiert

### 3. Modellinitialisierung
- **Sampler einrichten**:
  ```python
  self._data_sampler = DataSampler(
      train_data, self._transformer.output_info_list, self._log_frequency
  )
  ```
  - Speichert Informationen über die Datenverteilung und stellt Sampling-Funktionen bereit

- **Netzwerke erzeugen**:
  ```python
  self._generator = Generator(
      self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
  ).to(self._device)

  discriminator = Discriminator(
      data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
  ).to(self._device)
  ```
  - Generator erzeugt aus Rauschen und bedingtem Vektor synthetische Daten
  - Diskriminator unterscheidet zwischen echten und gefälschten Daten

- **Optimierer initialisieren**:
  ```python
  optimizerG = optim.Adam(
      self._generator.parameters(),
      lr=self._generator_lr,
      betas=(0.5, 0.9),
      weight_decay=self._generator_decay,
  )

  optimizerD = optim.Adam(
      discriminator.parameters(),
      lr=self._discriminator_lr,
      betas=(0.5, 0.9),
      weight_decay=self._discriminator_decay,
  )
  ```
  - Adam-Optimierer für Generator und Diskriminator
  - Spezifische Hyperparameter (Lernrate, Gewichtsverfall) für beide Netzwerke

- **Weight Decay ≠ Lernraten-Anpassung**: 
  - `weight_decay` reduziert Gewichte (L2-Regularisierung)
  - Hat keinen Einfluss auf die Lernrate selbst
  
- **Aktuelles Setup**:
  - `optimizerG.step()` wendet nur die aktuelle, feste Lernrate an
  - Keine dynamische Anpassung der Lernrate implementiert

- **Möglichkeiten für dynamische Lernraten**:
  1. Learning Rate Scheduler hinzufügen:
     ```python
     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG)
     # Nach jeder Epoche:
     scheduler.step(loss_value)
     ```
  
  2. Manuelles Anpassen basierend auf G/D-Verhältnis:
     ```python
     # Nach x Epochen:
     for param_group in optimizerG.param_groups:
         param_group['lr'] = neue_lernrate
     ```
     siehe dazu auch doc_TRAIN.md

- **Aktuell verwendete Parameter**:
  - Adam-Optimizer mit fester Startlernrate `self._discriminator_lr`
  - Beta-Parameter (0.5, 0.9) für Momentum
  - Weight Decay `self._discriminator_decay` für Regularisierung

### 4. Trainingsschleife
```python
for i in epoch_iterator:  # Epoche
    for id_ in range(steps_per_epoch):  # Batch
        # Diskriminator-Training
        for n in range(self._discriminator_steps):
            # [Diskriminator-Trainingsschritte]
        
        # Generator-Training
        # [Generator-Trainingsschritte]
```

#### 4.1 Diskriminator-Training
```python
# Noise und Bedingungsvektor erzeugen
fakez = torch.normal(mean=mean, std=std)
condvec = self._data_sampler.sample_condvec(self._batch_size)

# Bedingten Noise-Vektor erstellen
if condvec is not None:
    c1, m1, col, opt = condvec
    c1 = torch.from_numpy(c1).to(self._device)
    m1 = torch.from_numpy(m1).to(self._device)
    fakez = torch.cat([fakez, c1], dim=1)

# Echte Daten aus der gleichen Kategorie samplen
if condvec is not None:
    c2 = c1[perm]  # Permutierte Bedingungsvektoren
    real = self._data_sampler.sample_data(train_data, self._batch_size, col[perm], opt[perm])

# Synthetische Daten erzeugen
fake = self._generator(fakez)
fakeact = self._apply_activate(fake)  # Aktivierungsfunktionen anwenden

# Diskriminator auf echte und synthetische Daten anwenden
y_fake = discriminator(fake_cat)
y_real = discriminator(real_cat)

# Wasserstein-Verlust und Gradienten-Penalty berechnen
pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac)
loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

# Optimierungsschritt für Diskriminator
optimizerD.zero_grad(set_to_none=False)
pen.backward(retain_graph=True)
loss_d.backward()
optimizerD.step()
```

#### 4.2 Generator-Training
```python
# Neue Noise und Bedingungsvektoren erzeugen
fakez = torch.normal(mean=mean, std=std)
condvec = self._data_sampler.sample_condvec(self._batch_size)

# Bedingte Vektoren einfügen, falls vorhanden
if condvec is not None:
    c1, m1, col, opt = condvec
    c1 = torch.from_numpy(c1).to(self._device)
    m1 = torch.from_numpy(m1).to(self._device)
    fakez = torch.cat([fakez, c1], dim=1)

# Synthetische Daten erzeugen
fake = self._generator(fakez)
fakeact = self._apply_activate(fake)

# Diskriminator auf synthetische Daten anwenden
if c1 is not None:
    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
else:
    y_fake = discriminator(fakeact)

# Cross-Entropy für diskrete Spalten berechnen
if condvec is None:
    cross_entropy = 0
else:
    cross_entropy = self._cond_loss(fake, c1, m1)

# Generator-Verlust berechnen
loss_g = -torch.mean(y_fake) + cross_entropy

# Optimierungsschritt für Generator
optimizerG.zero_grad(set_to_none=False)
loss_g.backward()
optimizerG.step()
```

## Mathematische Analyse der Loss-Funktionen

### Diskriminator-Loss

```python
loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
```

- **Mathematische Formel**: $L_D = -(E_{x \sim P_{data}}[D(x)] - E_{z \sim P_z}[D(G(z))])$
- **Erklärung**:
  - WGAN-Loss (Wasserstein GAN)
  - Maximiert den Unterschied zwischen der Bewertung echter Daten und generierter Daten
  - Je größer der Unterschied, desto besser kann der Diskriminator unterscheiden
  - Das Minuszeichen bedeutet: Der Optimierer minimiert, wir wollen aber maximieren

### Gradient-Penalty

```python
# Berechnung in der calc_gradient_penalty Funktion
gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
gradient_penalty = ((gradients_view) ** 2).mean() * lambda_
```

- **Mathematische Formel**: $GP = \lambda \cdot E_{x \sim P_{penalty}}[(\|\nabla_x D(x)\|_2 - 1)^2]$
- **Erklärung**:
  - WGAN-GP (WGAN mit Gradient Penalty)
  - Bestraft Gradienten, die von 1 abweichen (Lipschitz-Bedingung)
  - Stellt die Stabilität des Diskriminators sicher
  - Verhindert das Vanishing-Gradient-Problem

### Generator-Loss

```python
loss_g = -torch.mean(y_fake) + cross_entropy
```

- **Mathematische Formel**: $L_G = -E_{z \sim P_z}[D(G(z))] + L_{cond}$
- **Erklärung**:
  - Erster Teil: Standard-WGAN-Loss für den Generator
  - Ziel: Erreichen, dass der Diskriminator generierte Daten als echt bewertet
  - Je höher die Bewertung der synthetischen Daten, desto kleiner der Loss

### Conditional Loss (Cross-Entropy)

```python
# In der _cond_loss Methode
tmp = functional.cross_entropy(
    data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
)
loss.append(tmp)
# ...
return (loss * m).sum() / data.size()[0]
```

- **Mathematische Formel**: $L_{cond} = \frac{1}{N} \sum_{i} \sum_{j} m_i \cdot CE(G(z)_{ij}, c_{ij})$
  - wobei $CE$ die Cross-Entropy-Funktion ist
- **Erklärung**:
  - Bestraft den Generator, wenn er diskrete Kategorien falsch erzeugt
  - Stellt sicher, dass der Generator die konditionellen Eigenschaften berücksichtigt
  - Wichtig für die Erzeugung strukturierter, kategorischer Daten

## Warum verbessert die Gewichtsanpassung das Gesamtmodell?

### Wechselwirkung zwischen Diskriminator und Generator

1. **Adversarielles Gleichgewicht**:
   - Der Diskriminator lernt besser zu unterscheiden → verbessert seine Feedback-Qualität
   - Der Generator muss sich an das verbesserte Feedback anpassen → erzeugt realistischere Daten
   - Diese konkurrierende Dynamik treibt beide Komponenten zur Verbesserung an

2. **Wasserstein-Distanz als Trainingsmetrik**:
   - Misst den Abstand zwischen realer und generierter Datenverteilung
   - Bietet ein stabiles Gradientensignal, selbst wenn die Verteilungen nicht überlappen
   - Führt zu einem glatteren Konvergenzverhalten als bei herkömmlichen GANs

3. **Gradient-Penalty sichert Stabilität**:
   - Verhindert zu steile Gradienten und damit Instabilität
   - Erzwingt gleichmäßigere Lernfortschritte
   - Ermöglicht tiefere und ausdrucksstärkere Diskriminator-Netzwerke

4. **Konditionierungsmechanismus**:
   - Der bedingte Verlust (Cross-Entropy) steuert den Generator gezielt
   - Kategorische Merkmale werden korrekt reproduziert
   - Tabellarische Strukturen und Abhängigkeiten bleiben erhalten

5. **PAC-Mechanismus (Packed Averaging Classification)**:
   - Bündelt mehrere Samples zusammen für stabileres Training
   - Verbessert die Gradientenschätzung, besonders bei kleinem Batch
   - Erhöht die statistische Effizienz des Lernprozesses

## Zusammenfassung

- CTGAN ist ein spezialisiertes GAN für tabellarische Daten
- Kombiniert WGAN-GP mit bedingter Erzeugung
- Transformiert Daten mit GMM für kontinuierliche und One-Hot-Encoding für diskrete Variablen
- Balanciert Diskriminator- und Generator-Training durch abwechselnde Optimierungsschritte
- Dual-Loss-Ansatz: Wasserstein-Distanz für realistische Daten und Cross-Entropy für korrekte Kategorien
- Gradient-Penalty sorgt für Stabilität während des Trainingsprozesses

Der inkrementelle Verbesserungsprozess durch den Wettbewerb zwischen Generator und Diskriminator, gepaart mit den speziellen Loss-Funktionen für tabellarische Daten, ermöglicht die Erzeugung synthetischer Daten mit ähnlichen statistischen Eigenschaften wie die Originaldaten.