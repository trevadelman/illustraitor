# illustr/ai/tor

AI image generation powered by DrawThings API, featuring an intelligent chat assistant for prompt crafting and multiple generation modes.

## ⚠️ Important: DrawThings Requirement

This application requires the DrawThings application to be installed and running locally with HTTP server enabled:

1. Download and install DrawThings from the official source
2. Launch DrawThings
3. Enable the HTTP server in DrawThings settings:
   - Go to Settings
   - Enable "HTTP Server"
   - Ensure it's running on port 7860 (default)
   - Keep DrawThings running while using this application

The illustraitor application connects to DrawThings' local API to generate images. Without DrawThings running with HTTP server enabled, image generation will not work.

## Features

### 🎨 Generation Modes
- **Simple Mode**: Quick access to basic image generation with preset styles
- **Advanced Mode**: 
  - AI Chat Assistant for guided prompt creation
  - Custom JSON configuration for fine-tuned control

### 🖼️ Image Customization
- Multiple size options: 256px, 512px, 768px
- 9 carefully crafted style presets:
  - Photo Realistic
  - Animated Style
  - Small Logo
  - Quick Draft
  - Fluid Gradient
  - Neon Dreams
  - Retro Wave
  - Glass Morphism
  - Paper Cut

### 🤖 AI Assistant
- Intelligent chat interface powered by GPT-4
- Expert guidance for prompt crafting
- Automatic parameter optimization based on style
- Real-time JSON configuration generation

### 📊 Gallery & Management
- Recent generations gallery
- Detailed image metadata viewing
- Image deletion capability
- Full-screen modal view with advanced details

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- DrawThings API running locally
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd illustraitor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

5. Ensure DrawThings API is running locally on port 7860

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for chat assistance

### Image Sizes
Available sizes are configured in `app.py`:
- Small: 256x256
- Medium: 512x512
- Large: 768x768

### Style Presets
Presets are defined in `presets.json` with the following parameters:
- steps
- guidance_scale
- guidance_embed
- sampler
- negative_prompt
- aesthetic_score
- sharpness
- refiner_start

## Usage

### Starting the Server
```bash
python app.py
```
The server will start on http://localhost:8000

### Generation Modes

#### Simple Mode
1. Enter your prompt
2. Select image size
3. Choose a style preset
4. Click "Generate Image"

#### Advanced Mode - AI Chat
1. Switch to Advanced Mode
2. Describe your image idea to the AI assistant
3. Refine based on AI suggestions
4. Use generated configuration

#### Advanced Mode - Custom JSON
1. Switch to Advanced Mode
2. Select "Custom JSON"
3. Input or paste your configuration
4. Click "Generate Image"

### Image Management
- View generated images in the gallery
- Click images to view full details
- Delete unwanted images
- Export metadata for reference

## API Documentation

### Main Endpoints

#### GET /
- Returns the main application interface

#### POST /generate
- Generates an image based on provided parameters
- Parameters:
  - prompt (string)
  - size (string): small/medium/large
  - preset (string): preset name
  - custom_json (string, optional): custom configuration

#### POST /chat
- Interacts with AI assistant
- Parameters:
  - message (string): user message

#### GET /recent_images
- Returns recent image generations

#### POST /delete_image
- Deletes an image and its metadata
- Parameters:
  - image_path (string)

## Technical Details

### Image Generation Parameters

#### Photorealistic Style
```json
{
  "steps": 25,
  "guidance_scale": 7.0,
  "guidance_embed": 3.0,
  "sampler": "DPM++ 2M Karras",
  "sharpness": 2
}
```

#### Artistic Style
```json
{
  "steps": 20-25,
  "guidance_scale": 5.0-6.5,
  "guidance_embed": 2.0-2.5,
  "sampler": "Euler a",
  "sharpness": 1
}
```

#### Logo Design
```json
{
  "steps": 25,
  "guidance_scale": 6.5,
  "guidance_embed": 2.5,
  "sampler": "Euler a",
  "sharpness": 1
}
```

### File Structure
```
illustraitor/
├── app.py              # Main Flask application
├── presets.json        # Style preset configurations
├── static/
│   ├── style.css       # Application styling
│   └── output/         # Generated images and metadata
└── templates/
    └── index.html      # Main application interface
```

### Dependencies
- Flask: Web framework
- OpenAI: Chat assistance
- aiohttp: Async HTTP client
- python-dotenv: Environment management

## Performance

- Average generation time: 2:15-2:25 minutes
- Supported image sizes: up to 768x768
- Concurrent generations: Single queue
- Storage: Local filesystem

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT
