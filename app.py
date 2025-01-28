from flask import Flask, render_template, request, jsonify, send_from_directory, session
import json
import asyncio
import aiohttp
import base64
from datetime import datetime, UTC
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# System message for chat context
SYSTEM_MESSAGE = """You are an expert AI art generation consultant specialized in crafting precise JSON configurations for image generation. Your role is to translate user visions into optimal technical parameters while maintaining a conversational, helpful tone.

CORE RESPONSIBILITIES:

1. Gather Requirements
- Understand the desired image concept, style, and mood
- Ask targeted questions about key visual elements
- Clarify artistic preferences (realistic, stylized, abstract, etc.)
- Consider technical constraints if mentioned by user

2. Prompt Construction Rules:
- CLARITY OVER LENGTH: Keep prompts concise and specific
- Use 15-25 words maximum for the main subject and style
- Structure prompts in this priority order:
  1. Core subject/concept (2-3 words)
  2. Primary style (1-2 words)
  3. Essential composition elements (2-3 words)
  4. Key details (3-4 words)
  5. Color/lighting (3-4 words)
- Avoid:
  * Redundant descriptions
  * Overly flowery language
  * Multiple adjectives for the same element
  * Long narrative descriptions

3. Blur Prevention Guidelines:
- Higher guidance_scale doesn't always mean better results
- For detailed subjects (like pixel art):
  * Keep prompts shorter and more direct
  * Use stronger negative prompts to prevent blur
  * Lower guidance_scale (4.5-5.5) often produces clearer results
  * Pair with higher sharpness (2) when clarity is crucial
- For soft/ethereal subjects:
  * Can use more descriptive language
  * Higher guidance_scale (6.0-7.0)
  * Lower sharpness (1)

4. Parameter Selection Based on Style:
Photorealistic:
- Steps: 25-30
- guidance_scale: ~7.0
- Sampler: DPM++ 2M Karras
- guidance_embed: 2.5-3.0
- sharpness: 2

Artistic/Stylized:
- Steps: 20-25
- guidance_scale: ~5.5
- Sampler: Euler a
- guidance_embed: 1.8-2.2
- sharpness: 1-2

Minimal/Logo:
- Steps: 15-20
- guidance_scale: ~4.5
- Sampler: Euler a
- guidance_embed: 1.5-2.0
- sharpness: 2

Fluid/Gradient:
- Steps: 20-25
- guidance_scale: ~6.5
- Sampler: Euler a
- guidance_embed: 2.0-2.5
- sharpness: 1

Example Prompts:
GOOD: "pixel art Death Star, centered, detailed surface panels, starry space background, metallic grays"
BAD: "A pixel art representation of the Death Star in the vastness of space with detailed surface features and dramatic lighting creating depth..."

INTERACTION GUIDELINES:
- Maintain a helpful, conversational tone while gathering requirements
- Ask no more than one clarifying question at a time
- Only provide the JSON response when you have sufficient information
- If user provides example images, analyze their style characteristics
- Explain parameter choices if asked
- If user indicates poor results, focus on adjusting key parameters (guidance_scale, steps, sampler) rather than minor tweaks
- For compute-constrained scenarios, recommend lower steps and simpler samplers

RESPONSE FORMAT:
Always provide the final response as valid JSON with this EXACT structure:
{
  "prompt": "Detailed generation prompt",
  "width": 512,
  "height": 512,
  "steps": <25-40>,             
  "guidance_scale": <6.0-8.0>,   
  "guidance_embed": <2.0-3.5>,   
  "sampler": "<Euler a OR DPM++ 2M Karras>",  
  "negative_prompt": "Elements to avoid",
  "aesthetic_score": <6.5-9.0>,  
  "sharpness": <1.5 OR 2>,       
  "refiner_start": <0.5-0.8>,    
  "model": "flux_1_schnell_q8p.ckpt",
  "batch_size": 1,
  "batch_count": 1,
  "seed": -1,
  "seed_mode": "Scale Alike",
  "clip_skip": 1,
  "preserve_original_after_inpaint": true,
  "target_width": <same as width>,
  "target_height": <same as height>,
  "upscaler": null,
  "strength": 1,
  "mask_blur": 1.5,
  "tiled_decoding": false,
  "resolution_dependent_shift": true,
  "speed_up_with_guidance_embed": true,
  "zero_negative_prompt": false,
  "t5_text_encoder_decoding": true
}

Remember:
- Quality depends on prompt structure and parameter harmony rather than maximizing all values
- Clearer prompts generally produce better results than longer ones
- Always match parameters to the desired style
- When in doubt, err on the side of simplicity"""

# Load configuration presets
with open('presets.json', 'r') as f:
    PRESETS = json.load(f)['presets']

# Image size options
IMAGE_SIZES = {
    'small': 256,
    'medium': 512,
    'large': 768
}

# Ensure output directory exists
OUTPUT_DIR = Path('static/output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_recent_images(limit=12):
    """Load most recent images with their metadata."""
    images = []
    for img_path in sorted(OUTPUT_DIR.glob('*.png'), reverse=True)[:limit]:
        meta_path = img_path.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            images.append({
                'path': f'output/{img_path.name}',
                'prompt': metadata['configuration']['prompt'],
                'size': f"{metadata['configuration']['imageSize']}x{metadata['configuration']['imageSize']}",
                'time': metadata['generationTime'],
                'preset': metadata.get('configuration', {}).get('preset', 'Custom'),
                'timestamp': metadata['timestamp']
            })
    return images

async def generate_image_with_custom_body(request_body):
    """Generate image using DrawThings API with custom request body."""
    logger.info("\n🎨 Starting image generation with custom configuration...")
    logger.info(f"• Prompt: \"{request_body.get('prompt', 'No prompt provided')}\"")
    
    start_time = datetime.now(UTC)
    logger.info(f"\nConnecting to DrawThings API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://127.0.0.1:7860/sdapi/v1/txt2img',
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise Exception(f"API error: {error_text}")
                
                data = await response.json()
                
                if not data.get('images'):
                    logger.error("No images in response")
                    raise Exception("No images in response")
                
                logger.info("\nReceived response from API")
                
                # Save image and metadata
                timestamp = datetime.now(UTC)
                prompt_text = request_body.get('prompt', 'custom_generation')[:30]
                base_name = f"{prompt_text.replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Save image
                image_data = base64.b64decode(data['images'][0])
                image_path = OUTPUT_DIR / f"{base_name}.png"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"✓ Saved image: {image_path.name}")
                
                # Save metadata
                generation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
                metadata = {
                    "version": "1.0",
                    "timestamp": timestamp.isoformat(),
                    "imageFile": image_path.name,
                    "generationTime": f"{int(generation_time/1000//60)}:{int(generation_time/1000%60):02d}",
                    "configuration": {
                        "preset": "Custom",
                        "imageSize": request_body.get('width', 512),
                        "prompt": request_body.get('prompt', 'No prompt provided'),
                        **request_body
                    }
                }
                
                meta_path = OUTPUT_DIR / f"{base_name}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"✓ Saved metadata: {meta_path.name}")
                
                total_time = f"{int(generation_time/1000//60)}:{int(generation_time/1000%60):02d}"
                logger.info(f"\n✨ Generation completed in {total_time}")
                
                return {
                    'success': True,
                    'image_path': f'static/output/{image_path.name}',
                    'metadata': metadata
                }
                
    except Exception as e:
        logger.error(f"\n❌ Error occurred: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

async def generate_image(prompt, size, preset_name):
    """Generate image using DrawThings API."""
    # Get preset configuration
    preset = next((p for p in PRESETS if p['name'] == preset_name), PRESETS[0])
    
    # Log generation start
    logger.info("\n🎨 Starting image generation...")
    logger.info(f"• Prompt: \"{prompt}\"")
    logger.info(f"• Image Size: {IMAGE_SIZES[size]}x{IMAGE_SIZES[size]}")
    logger.info(f"• Preset: {preset_name}")
    
    # Base configuration
    config = {
        "prompt": prompt,
        "imageSize": IMAGE_SIZES[size],
        "steps": preset['config']['steps'],
        "guidance_scale": preset['config']['guidance_scale'],
        "sampler": preset['config']['sampler'],
        "negative_prompt": preset['config']['negative_prompt'],
        "aesthetic_score": preset['config']['aesthetic_score'],
        "model": "flux_1_schnell_q8p.ckpt",
        "batch_size": 1,
        "batch_count": 1,
        "seed": -1
    }
    
    # Build request body
    request_body = {
        "prompt": config["prompt"],
        "negative_prompt": config["negative_prompt"],
        "width": config["imageSize"],
        "height": config["imageSize"],
        "batch_size": config["batch_size"],
        "batch_count": config["batch_count"],
        "steps": config["steps"],
        "model": config["model"],
        "sampler": config["sampler"],
        "seed": config["seed"],
        "guidance_scale": config["guidance_scale"],
        "aesthetic_score": config["aesthetic_score"],
        "seed_mode": "Scale Alike",
        "guidance_embed": 3.5,
        "clip_skip": 1,
        "preserve_original_after_inpaint": True,
        "refiner_start": 0.8500000238418579,
        "target_width": config["imageSize"],
        "target_height": config["imageSize"],
        "upscaler": None,
        "strength": 1,
        "mask_blur": 1.5,
        "tiled_decoding": False,
        "resolution_dependent_shift": True,
        "speed_up_with_guidance_embed": True,
        "sharpness": 0,
        "zero_negative_prompt": False,
        "t5_text_encoder_decoding": True
    }
    
    if app.debug:
        logger.info("\nRequest body:")
        logger.info(json.dumps(request_body, indent=2))
    
    start_time = datetime.now(UTC)
    logger.info(f"\nConnecting to DrawThings API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://127.0.0.1:7860/sdapi/v1/txt2img',
                json=request_body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    raise Exception(f"API error: {error_text}")
                
                data = await response.json()
                
                if not data.get('images'):
                    logger.error("No images in response")
                    raise Exception("No images in response")
                
                logger.info("\nReceived response from API")
                
                # Save image and metadata
                timestamp = datetime.now(UTC)
                base_name = f"{prompt[:30].replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Save image
                image_data = base64.b64decode(data['images'][0])
                image_path = OUTPUT_DIR / f"{base_name}.png"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"✓ Saved image: {image_path.name}")
                
                # Save metadata
                generation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
                metadata = {
                    "version": "1.0",
                    "timestamp": timestamp.isoformat(),
                    "imageFile": image_path.name,
                    "generationTime": f"{int(generation_time/1000//60)}:{int(generation_time/1000%60):02d}",
                    "configuration": {
                        **config,
                        "preset": preset_name
                    }
                }
                
                meta_path = OUTPUT_DIR / f"{base_name}.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"✓ Saved metadata: {meta_path.name}")
                
                total_time = f"{int(generation_time/1000//60)}:{int(generation_time/1000%60):02d}"
                logger.info(f"\n✨ Generation completed in {total_time}")
                
                return {
                    'success': True,
                    'image_path': f'output/{image_path.name}',
                    'metadata': metadata
                }
                
    except Exception as e:
        logger.error(f"\n❌ Error occurred: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def create_chat_completion(messages):
    """Create a chat completion using OpenAI's API"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate image configuration"""
    try:
        message = request.json.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Initialize or get chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = [
                {"role": "system", "content": SYSTEM_MESSAGE}
            ]

        # Add user message to history
        session['chat_history'].append({"role": "user", "content": message})

        # Get AI response
        response = create_chat_completion(session['chat_history'])
        session['chat_history'].append({"role": "assistant", "content": response})
        session.modified = True

        # Check if response is JSON (final configuration)
        try:
            json_response = json.loads(response)
            if ('prompt' in json_response and 
                'width' in json_response and 
                'height' in json_response and 
                'steps' in json_response):  # Basic validation of required fields
                return jsonify({
                    'message': response,
                    'isComplete': True,
                    'configuration': json_response  # Pass the entire response as the configuration
                })
        except json.JSONDecodeError:
            pass

        return jsonify({
            'message': response,
            'isComplete': False
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    # Clear any existing chat history
    session.pop('chat_history', None)
    return render_template(
        'index.html',
        presets=PRESETS,
        sizes=IMAGE_SIZES,
        recent_images=load_recent_images()
    )

@app.route('/generate', methods=['POST'])
def generate():
    custom_json = request.form.get('custom_json', '').strip()
    
    result = None
    
    if custom_json:
        try:
            # Use custom JSON as request body
            request_body = json.loads(custom_json)
            result = asyncio.run(generate_image_with_custom_body(request_body))
        except json.JSONDecodeError:
            return jsonify({'success': False, 'error': 'Invalid JSON format'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        # Use standard form parameters
        prompt = request.form.get('prompt', '').strip()
        size = request.form.get('size', 'medium')
        preset = request.form.get('preset', PRESETS[0]['name'])
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Either prompt or custom JSON is required'})
        
        if size not in IMAGE_SIZES:
            return jsonify({'success': False, 'error': 'Invalid size'})
        
        if not any(p['name'] == preset for p in PRESETS):
            return jsonify({'success': False, 'error': 'Invalid preset'})
        
        result = asyncio.run(generate_image(prompt, size, preset))
    
    if result['success']:
        return jsonify({
            'success': True,
            'image_path': result['image_path'],
            'metadata': result['metadata']
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        })

@app.route('/recent_images')
def recent_images():
    return jsonify(load_recent_images())

@app.route('/delete_image', methods=['POST'])
def delete_image():
    """Delete an image and its associated metadata file."""
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'error': 'Image path is required'}), 400

        image_path = data['image_path']
        if not image_path.startswith('output/'):
            return jsonify({'error': 'Invalid image path'}), 400

        # Get the full paths for both image and metadata files
        image_file = OUTPUT_DIR / image_path.replace('output/', '')
        json_file = image_file.with_suffix('.json')

        # Check if files exist
        if not image_file.exists() or not json_file.exists():
            return jsonify({'error': 'Image or metadata file not found'}), 404

        # Delete both files
        image_file.unlink()
        json_file.unlink()

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
