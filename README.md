[[Paper]]() [[Colab]]() [[Kaggle]]() [[Model Card]]()

ruDALLE aspect ratio images
---
Use any custom aspect ratio for image generation by ruDALLE Malevich XL

### Installing

```
pip install rudalle==1.0.0
git clone https://github.com/shonenkov-AI/rudalle-aspect-ratio
```

### Quick Start

Horizontal images:
```python3
import sys
sys.path.insert(0, './rudalle-aspect-ratio')
from rudalle_aspect_ratio import RuDalleAspectRatio, get_rudalle_model
from rudalle import get_vae, get_tokenizer
from rudalle.pipelines import show

device = 'cuda'
dalle = get_rudalle_model('Surrealist_XL', fp16=True, device=device)
vae, tokenizer = get_vae().to(device), get_tokenizer()
rudalle_ar = RuDalleAspectRatio(
    dalle=dalle, vae=vae, tokenizer=tokenizer,
    aspect_ratio=32/9, bs=4, device=device
)
_, result_pil_images = rudalle_ar.generate_images('готический квартал', 2048, 0.975, 4)
show(result_pil_images, 4)
```
![](./pics/h_example.jpg)

Vertical images:
```python3
rudalle_ar = RuDalleAspectRatio(
    dalle=dalle, vae=vae, tokenizer=tokenizer,
    aspect_ratio=9/32, bs=4, device=device
)
_, result_pil_images = rudalle_ar.generate_images('голубой цветок', 2048, 0.975, 4)
show(result_pil_images, 4)
```

![](./pics/w_example.jpg)


# Author:
```
@misc{shonenkov2022rudalle_aspect_ration,
      title={ruDALLE aspect ratio images: technical report},
      author={Alex Shonenkov},
      year={2022},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
