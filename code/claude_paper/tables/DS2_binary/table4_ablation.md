| ID | Config | Sobel | Attn | DeiT | XAttn | Acc(%) | F1(%) | dF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | Backbone only | - | None | - | - | 99.1800 | 99.1800 | 0.0000 |
| A1 | Backbone + SE-Net | - | SE | - | - | 98.5600 | 98.5600 | -0.6200 |
| A2 | Backbone + CBAM | - | CBAM | - | - | 98.7100 | 98.7100 | -0.4700 |
| A3 | Backbone + ECA | - | ECA | - | - | 98.9500 | 98.9500 | -0.2300 |
| A4 | Backbone + CoordAtt | - | CA | - | - | 98.6300 | 98.6300 | -0.5500 |
| A5 | Backbone + Sobel | Y | None | - | - | 98.7100 | 98.7100 | -0.4700 |
| A6 | Backbone + Sobel + MS-EGCA | Y | MS_EGCA | - | - | 99.1000 | 99.1000 | -0.0800 |
| A7 | Backbone + Sobel + MS-EGCA + DeiT | Y | MS_EGCA | Y | - | 99.1000 | 99.1000 | -0.0800 |
| A8 | Proposed (Full) | Y | MS_EGCA | Y | Y | 99.1000 | 99.1000 | -0.0800 |