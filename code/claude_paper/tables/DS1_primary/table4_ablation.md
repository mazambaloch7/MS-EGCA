| ID | Config | Sobel | Attn | DeiT | XAttn | Acc(%) | F1(%) | dF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | Backbone only | - | None | - | - | 98.3300 | 98.3200 | 0.0000 |
| A1 | Backbone + SE-Net | - | SE | - | - | 98.1100 | 98.1100 | -0.2100 |
| A2 | Backbone + CBAM | - | CBAM | - | - | 98.5100 | 98.5100 | 0.1900 |
| A3 | Backbone + ECA | - | ECA | - | - | 98.3000 | 98.2900 | -0.0300 |
| A4 | Backbone + CoordAtt | - | CA | - | - | 98.1800 | 98.1800 | -0.1400 |
| A5 | Backbone + Sobel | Y | None | - | - | 98.3700 | 98.3700 | 0.0500 |
| A6 | Backbone + Sobel + MS-EGCA | Y | MS_EGCA | - | - | 98.4400 | 98.4400 | 0.1200 |
| A7 | Backbone + Sobel + MS-EGCA + DeiT | Y | MS_EGCA | Y | - | 98.4400 | 98.4400 | 0.1200 |
| A8 | Proposed (Full) | Y | MS_EGCA | Y | Y | 98.4400 | 98.4400 | 0.1200 |