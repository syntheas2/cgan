# Optimales G/D-Verhältnis für GAN-Training

## Mathematische Betrachtung des Generator-Diskriminator-Verhältnisses

- **Ideales G/D-Verhältnis**: Ein Wert nahe **1,0** wird angestrebt
- **Mathematische Begründung**: 
  - Bei G/D ≈ 1,0 herrscht Nash-Gleichgewicht zwischen Generator und Diskriminator
  - L_G ≈ |L_D| bedeutet ausgewogenes Lernverhalten

## Interpretation der Abweichungen

- **G/D >> 1,0** (z.B. > 3,0):
  - Generator lernt zu langsam
  - Diskriminator dominiert den Lernprozess
  - $\frac{|L_G|}{|L_D|} \gg 1$ deutet auf Diskriminator-Überlegenheit hin

- **G/D << 1,0** (z.B. < 0,3):
  - Generator dominiert das Training
  - Diskriminator kann nicht mithalten
  - $\frac{|L_G|}{|L_D|} \ll 1$ signalisiert sogenanntes "Mode Collapse"-Risiko

## Mathematische Optimierung

- **Zielformel**: $\frac{|L_G|}{|L_D|} \approx 1 \pm 0.3$
- **Anpassungsregel**: Bei Abweichung des G/D-Verhältnisses:
  - Lernrate $\eta_G = \eta_G \cdot \sqrt{\frac{|L_D|}{|L_G|}}$
  - Lernrate $\eta_D = \eta_D \cdot \sqrt{\frac{|L_G|}{|L_D|}}$

## Praktische Anwendung

- G/D-Verhältnis regelmäßig in TensorBoard überwachen
- Bei anhaltenden Abweichungen vom optimalen Bereich (0,7-1,3) Lernraten anpassen
- Das Verhältnis ist besonders in frühen Trainingsphasen kritisch