# cogvideox-2b-img2vid


https://github.com/user-attachments/assets/925e733c-cab6-4b09-ad92-f3b8324679c9

### ComfyUI Example
<p>
    <img src="resources/cogvideox-2b-img2vid-workflow.png" width="800" height="400" title="preview"/>
</p>
<p>
    <a href="resources/cogvideox-2b-img2vid-workflow.json">JSON Workflow Example</a>
</p>
ComfyUI custom node can be found <a href="https://github.com/Nim-Video/ComfyUI-CogVideoXWrapper">here</a>.

## Setup and Inference Instructions

### Prerequisites

1. **System Update and Dependencies Installation**
   ```bash
   sudo apt-get update
   sudo apt-get install git-lfs cbm ffmpeg
   sudo apt-get upgrade ffmpeg
   ```

2. **Create and Activate Conda Environment**
   ```bash
   conda create --name py310 python=3.10
   conda activate py310
   pip install ipykernel
   python -m ipykernel install --user --name py310 --display-name "py310"
   ```

Clone repo 
```bash
git clone https://github.com/svjack/cogvideox-2b-img2vid.git
cd cogvideox-2b-img2vid
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```
  
### Simple examples
#### Inference with cli
```bash
python -m inference.cli_demo \
    --image_path "resources/truck.jpg" \
    --prompt "A truck is driving through a dirt road, showcasing its capability for off-roading." \
    --model_path NimVideo/cogvideox-2b-img2vid
```

#### Inference with Gradio
```bash
python -m inference.gradio_web_demo \
    --model_path NimVideo/cogvideox-2b-img2vid
```

## I2V Reference Images

### Cards
- **IMG_bat_gen.webp**
  <div style="display: flex; flex-direction: column; align-items: center;">
    <img src="IMG_bat_gen.webp" alt="IMG_bat_gen.webp" style="max-width: 100%;">
    <p><strong>Prompt:</strong> A film style shot. On the moon, this item on the moon surface. The background is that Earth looms large in the foreground.</p>
  </div>
- **image_hs_card_gen.webp**
  <div style="display: flex; flex-direction: column; align-items: center;">
    <img src="image_hs_card_gen.webp" alt="image_hs_card_gen.webp" style="max-width: 100%;">
    <p><strong>Prompt:</strong> In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.</p>
  </div>


# Use This repo to make Genshin Impact product video (https://github.com/svjack/LTX-Video) 

## Bat_card_LTX_Video.mp4
<div style="margin-bottom: 10px;">
  <video controls autoplay src="https://github.com/user-attachments/assets/b20fbcc7-3cf5-4c1b-a923-932a8ea9cd0a" style="width: 1024px; height: 800px;"></video>
</div>
<p><strong>Prompt:</strong> A film style shot. On the moon, this item on the moon surface. The background is that Earth looms large in the foreground.</p>

## Hs_card_LTX_Video.mp4
<div style="margin-bottom: 10px;">
  <video controls autoplay src="https://github.com/user-attachments/assets/fe423aaf-81b9-4dba-bca9-e69a1f60b1bc" style="width: 1024px; height: 800px;"></video>
</div>
<p><strong>Prompt:</strong> In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.</p>

# NimVideo/cogvideox1.5-5b-prompt-camera-motion demo 
```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX1.5-5B-I2V", torch_dtype=torch.bfloat16
)

pipe.load_lora_weights("NimVideo/cogvideox1.5-5b-prompt-camera-motion", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])
#pipe.to("cuda")

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

height = 768 
width = 1360
image = load_image("resources/truck.jpg").resize((width, height))

prompt = "Camera is moving to the right. A trunk driving on the road."

video_generate = pipe(
    image=image,
    prompt=prompt,
    height=height, 
    width=width, 
    num_inference_steps=50,  
    num_frames=16,  
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42), 
).frames[0]

output_path = "truck_right.mp4"
export_to_video(video_generate, output_path, fps=8)

from IPython import display 
display.Video(output_path, width=512, height=512
             )
```









<br/><br/>

## Acknowledgements
Original code and models [CogVideoX](https://github.com/THUDM/CogVideo/tree/main).  
