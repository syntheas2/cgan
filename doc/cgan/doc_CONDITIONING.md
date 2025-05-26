# Konditionierung in CTGAN: Zusammenfassung

## Funktionsweise der Konditionierung

- CTGAN erlaubt das Generieren von Daten mit spezifischen Werten in kategorischen Spalten
- Während des Trainings werden alle kategorischen Spalten zur Konditionierung verwendet
- Bei der Generierung kann auf eine einzelne Spalte konditioniert werden
- Es müssen nicht alle Spalten bei der Generierung angegeben werden

## Effizienz und Overhead

- Komplexe Konditionierungsmechanismen während des Trainings sind notwendig:
  - Erfassen von Abhängigkeiten zwischen Spalten
  - Stabilisierung des GAN-Trainings
  - Vermeidung von Mode-Collapse
- Der Overhead ist minimal im Vergleich zu den Vorteilen:
  - Bessere Qualität der generierten Daten
  - Flexibilität bei der Datengenerierung
  - Möglichkeit, die Konditionierungsspalte zu wechseln ohne Neutraining

## Anwendung

- Einfache API für die Generierung:
  ```python
  # Mit Konditionierung
  model.sample(n=1000, condition_column='spalte', condition_value='wert')
  
  # Ohne Konditionierung
  model.sample(n=1000)
  ```
- Interne Mechanismen wandeln die angegebene Spalte und den Wert in entsprechende IDs um
- Optimal für Anwendungsfälle, bei denen Daten mit bestimmten Eigenschaften generiert werden sollen